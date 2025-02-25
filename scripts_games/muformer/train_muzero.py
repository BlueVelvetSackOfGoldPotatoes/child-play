#!/usr/bin/env python3
"""
train_muzero_data.py

Generates a dataset of (ascii_state, policy, value, env_name) by training
a small MuZero agent (with an MLP) on multiple ASCII-based games in sequence,
for data collection only.

Environments must implement:
  - reset_board() or reset_game() -> to reset
  - get_text_state() -> returns a string of ASCII board
  - apply_move(...) or guess(...) -> to make a move
  - game_over -> boolean
  - check_win() or final outcomes
  - name -> string identifying the game
  - board_size or action space if relevant

We store the final dataset in a JSON file, e.g. "multigame_muzero_dataset.json".
"""

import os
import json
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import argparse

# Import your environment classes:
from tictactoe import TicTacToe
from connectfour import ConnectFour
from shapes import Shapes
from battleship import BattleShip

# ------------------ Environment Registry ------------------
ENV_REGISTRY = {
    "tictactoe": TicTacToe,
    "connectfour": ConnectFour,
    "shapes": Shapes,
    "battleship": BattleShip
}

# ------------------ Minimal MuZero Implementation ------------------

class MuZeroNet(nn.Module):
    """
    Simple demonstration network for MuZero data generation:
      - MLP representation
      - policy & value heads
      - dynamics + reward head
    """
    def __init__(self, hidden_size=64, action_dim=9):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim

        self.repr_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
        self.dynamics_fc = nn.Sequential(
            nn.Linear(hidden_size + action_dim, hidden_size),
            nn.ReLU()
        )
        self.reward_head = nn.Linear(hidden_size, 1)

    def representation(self, board_vec):
        x = self.repr_fc(board_vec)
        return x

    def prediction(self, latent):
        policy_logits = self.policy_head(latent)
        value = torch.tanh(self.value_head(latent))  # in [-1,1]
        return policy_logits, value

    def dynamics(self, latent, action_onehot):
        inp = torch.cat([latent, action_onehot], dim=-1)
        next_latent = self.dynamics_fc(inp)
        reward = torch.tanh(self.reward_head(next_latent))
        return next_latent, reward


class MCTSNode:
    def __init__(self, latent, env, prior=1.0):
        self.latent = latent
        self.env = env  # a cloned environment
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_terminal = env.game_over

    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class MuZeroAgent:
    def __init__(self, hidden_size=64, lr=1e-3):
        self.hidden_size = hidden_size
        self.lr = lr
        # We'll create the MuZeroNet once we know the action_dim
        self.model = None
        self.optimizer = None
        self.num_simulations = 15
        self.c_puct = 1.25

    def initialize_model(self, action_dim):
        """Create the MuZeroNet with the given action dimension."""
        self.model = MuZeroNet(hidden_size=self.hidden_size, action_dim=action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    # ------------------ Board encoding per environment ------------------ #

    def board_to_tensor_tictactoe(self, env):
        # 3x3 board => flatten => X=+1, O=-1, empty=0
        board_list = []
        symbol_map = {"X": 1.0, "O": -1.0, " ": 0.0}
        for row in env.board:
            for c in row:
                board_list.append(symbol_map[c])
        return torch.tensor([board_list], dtype=torch.float32)

    def board_to_tensor_connectfour(self, env):
        # 7x7 board => flatten => X=+1, O=-1, .=0
        board_list = []
        symbol_map = {"X": 1.0, "O": -1.0, ".": 0.0}
        for row in env.board:
            for c in row:
                board_list.append(symbol_map.get(c, 0.0))
        return torch.tensor([board_list], dtype=torch.float32)

    def board_to_tensor_shapes(self, env):
        # NxN of '0'/'1' => convert each char to float
        board_list = []
        for row in env.board:
            for c in row:
                board_list.append(float(c))
        return torch.tensor([board_list], dtype=torch.float32)

    def board_to_tensor_battleship(self, env):
        # Flatten guess_board (P1 or P2 perspective)
        symbol_map = {"~": 0.0, "S": 1.0, "X": 2.0, "O": 3.0}
        if env.current_player == "P1":
            board = env.guess_board_p1
        else:
            board = env.guess_board_p2

        board_list = []
        for row in board:
            for c in row:
                board_list.append(symbol_map.get(c, 0.0))
        return torch.tensor([board_list], dtype=torch.float32)

    def board_to_tensor(self, env_name, env):
        if env_name == "tictactoe":
            return self.board_to_tensor_tictactoe(env)
        elif env_name == "connectfour":
            return self.board_to_tensor_connectfour(env)
        elif env_name == "shapes":
            return self.board_to_tensor_shapes(env)
        elif env_name == "battleship":
            return self.board_to_tensor_battleship(env)
        else:
            raise ValueError(f"Unknown env_name: {env_name}")

    # ------------------ Valid moves per environment ------------------ #

    def get_valid_moves_tictactoe(self, env):
        moves = []
        for r in range(env.board_size):
            for c in range(env.board_size):
                if env.board[r][c] == " ":
                    moves.append((r, c))
        return moves

    def get_valid_moves_connectfour(self, env):
        # columns 0..6 if top cell is "."
        valid_moves = []
        for col in range(env.cols):
            if env.board[0][col] == ".":
                valid_moves.append(col)
        return valid_moves

    def get_valid_moves_shapes(self, env):
        # env.answer_options => e.g. [square, cross, circle, etc.]
        return list(range(len(env.answer_options)))

    def get_valid_moves_battleship(self, env):
        # row*size + col for squares not guessed
        size = env.board_size
        if env.current_player == "P1":
            guess_board = env.guess_board_p1
        else:
            guess_board = env.guess_board_p2

        valid_moves = []
        for r in range(size):
            for c in range(size):
                if guess_board[r][c] not in ("X", "O"):  # "~" or "S" => can guess
                    valid_moves.append(r*size + c)
        return valid_moves

    def get_valid_moves(self, env_name, env):
        if env_name == "tictactoe":
            return self.get_valid_moves_tictactoe(env)
        elif env_name == "connectfour":
            return self.get_valid_moves_connectfour(env)
        elif env_name == "shapes":
            return self.get_valid_moves_shapes(env)
        elif env_name == "battleship":
            return self.get_valid_moves_battleship(env)
        else:
            raise ValueError(f"Unknown env {env_name}")

    # ------------------ Action <-> Index conversions ------------------ #

    def action_to_index(self, env_name, action, env):
        if env_name == "tictactoe":
            r, c = action
            return r * env.board_size + c
        elif env_name == "connectfour":
            return action  # col in [0..6]
        elif env_name == "shapes":
            return action  # an int index
        elif env_name == "battleship":
            size = env.board_size
            return action  # already row*size + col
        else:
            raise ValueError(f"Unknown env_name: {env_name}")

    def action_to_onehot(self, env_name, action, action_dim, env):
        idx = self.action_to_index(env_name, action, env)
        onehot = torch.zeros(1, action_dim)
        onehot[0, idx] = 1.0
        return onehot

    def apply_action_env(self, env_name, env, action):
        """Actually make the move in the environment."""
        if env_name == "tictactoe":
            result, valid = env.apply_move(action)  # (r,c)
            return result, valid
        elif env_name == "connectfour":
            # guess(...) with col
            player_index = 0 if env.current_player == "P1" else 1
            result, valid = env.guess(player_index, action, None)
            return result, valid
        elif env_name == "shapes":
            result, valid = env.guess(action)
            return result, valid
        elif env_name == "battleship":
            player_index = 0 if env.current_player == "P1" else 1
            result, valid = env.guess(player_index, action, None)
            return result, valid
        else:
            raise ValueError(f"Unsupported environment: {env_name}")

    # ------------------ MCTS ------------------ #

    def run_mcts(self, env_name, env, action_dim):
        root_env = copy.deepcopy(env)
        board_vec = self.board_to_tensor(env_name, root_env)
        with torch.no_grad():
            latent_root = self.model.representation(board_vec)
        root = MCTSNode(latent_root, root_env, prior=1.0)
        self.expand_node(env_name, root, action_dim)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            # 1) Traverse
            while node.children and not node.is_terminal:
                move, node = self.select_child(node)
                search_path.append(node)
            # 2) Expand
            if not node.is_terminal:
                self.expand_node(env_name, node, action_dim)
                leaf_value = self.evaluate_leaf(node)
            else:
                leaf_value = 0.0
            # 3) Backprop
            for n in search_path:
                n.visit_count += 1
                n.value_sum += leaf_value

        move_visits = {mv: ch.visit_count for mv, ch in root.children.items()}
        total = sum(move_visits.values()) + 1e-8
        policy = {mv: v/total for mv,v in move_visits.items()}
        return policy

    def expand_node(self, env_name, node, action_dim):
        if node.is_terminal:
            return
        env = node.env
        valid_moves = self.get_valid_moves(env_name, env)

        with torch.no_grad():
            policy_logits, _ = self.model.prediction(node.latent)
        policy_probs = torch.softmax(policy_logits, dim=-1)[0]  # (action_dim,)

        for move in valid_moves:
            action_idx = self.action_to_index(env_name, move, env)
            prior = policy_probs[action_idx].item()
            action_1h = torch.zeros(1, action_dim)
            action_1h[0, action_idx] = 1.0
            with torch.no_grad():
                next_latent, _ = self.model.dynamics(node.latent, action_1h)
            next_env = copy.deepcopy(env)
            self.apply_action_env(env_name, next_env, move)
            child = MCTSNode(next_latent, next_env, prior=prior)
            node.children[move] = child

    def select_child(self, node):
        best_score = float("-inf")
        best_move = None
        best_child = None
        for mv, child in node.children.items():
            q = child.q_value()
            u = self.c_puct * child.prior * ((node.visit_count**0.5)/(1+child.visit_count))
            score = q + u
            if score > best_score:
                best_score = score
                best_move = mv
                best_child = child
        return best_move, best_child

    def evaluate_leaf(self, node):
        with torch.no_grad():
            _, value = self.model.prediction(node.latent)
        return value.item()

    def select_move(self, env_name, env, action_dim):
        policy = self.run_mcts(env_name, env, action_dim)
        best_move = max(policy, key=policy.get)
        return best_move, policy


def compute_outcome(env_name, env, last_result):
    """
    Return +1 if P1 eventually wins, -1 if P1 loses, 0 if tie or no decisive result.
    This is a simplistic approach for demonstration.
    """
    if last_result == "Win":
        # Typically, the environment's 'current_player' might have just changed,
        # so you might need to track who actually made the winning move.
        # For simplicity: we assume the last move was by 1 - env.current_player.
        # If that was P1 => outcome=+1, else -1
        # But it depends on how your environment toggles current_player.
        # We'll guess if env.current_player == "P1", that means P2's move caused the win => outcome = -1
        if env.current_player == "P1":
            return -1.0
        else:
            return +1.0
    elif last_result in ("Tie", "Draw"):
        return 0.0
    elif last_result == "Loss":
        # For Shapes: 'Loss' means current_player guessed incorrectly
        # If env.current_player == "P1", that's a negative outcome for P1
        return -1.0 if env.current_player == "P1" else +1.0
    return 0.0


def run_selfplay_episode(agent, env_name):
    EnvClass = ENV_REGISTRY[env_name]
    env = EnvClass()
    if hasattr(env, "reset_board"):
        env.reset_board()
    elif hasattr(env, "reset_game"):
        env.reset_game()

    # Decide action_dim
    if env_name == "tictactoe":
        action_dim = 3*3
    elif env_name == "connectfour":
        # For demonstration, we use 7 for columns
        action_dim = 7
    elif env_name == "shapes":
        action_dim = len(env.answer_options)  # typically 4, but can vary
    elif env_name == "battleship":
        size = env.board_size
        action_dim = size * size
    else:
        raise ValueError(f"Unknown env {env_name}")

    # Initialize MuZero model if not done or if action_dim changed
    if agent.model is None or agent.model.action_dim != action_dim:
        agent.initialize_model(action_dim)

    trajectory = []
    while not env.game_over:
        ascii_state = env.get_text_state()
        move, policy = agent.select_move(env_name, env, action_dim)
        policy_vec = [0.0]*action_dim
        for mv, p in policy.items():
            idx = agent.action_to_index(env_name, mv, env)
            policy_vec[idx] = p

        step_data = {
            "ascii_state": ascii_state,
            "policy": policy_vec
        }

        last_result, valid = agent.apply_action_env(env_name, env, move)
        trajectory.append(step_data)
        if not valid:
            # Shouldn't happen if MCTS enumerates valid moves
            break

    # game is over, compute outcome from P1 perspective
    outcome = compute_outcome(env_name, env, last_result)
    # fill value for each step
    for step in trajectory:
        step["value"] = outcome

    return trajectory, outcome


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=str, default="tictactoe,connectfour,shapes,battleship",
                        help="Comma-separated list of environment names to train on.")
    parser.add_argument("--episodes_per_game", type=int, default=10, help="Number of self-play episodes per game.")
    parser.add_argument("--output_file", type=str, default="multigame_muzero_dataset.json")
    args = parser.parse_args()

    game_list = args.games.split(",")

    agent = MuZeroAgent(hidden_size=64, lr=1e-3)
    dataset = []

    for env_name in game_list:
        print(f"=== Generating data for {env_name} ===")
        for ep in range(args.episodes_per_game):
            ep_data, outcome = run_selfplay_episode(agent, env_name)
            for step in ep_data:
                dataset.append({
                    "env_name": env_name,
                    "ascii_state": step["ascii_state"],
                    "policy": step["policy"],
                    "value": step["value"]
                })
            print(f"Game {env_name}, episode {ep+1}, outcome={outcome}")

    # Save dataset
    with open(args.output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to {args.output_file} with {len(dataset)} samples.")


if __name__ == "__main__":
    main()


"""
python train_muzero_data.py \
  --games "tictactoe,connectfour,shapes,battleship" \
  --episodes_per_game 10 \
  --output_file multigame_muzero_dataset.json
"""