#!/usr/bin/env python3
"""
run_pipeline.py

Loads the trained Transformer policy-value model (fine-tuned from
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) and performs inference
on a newly instantiated game environment (TicTacToe, ConnectFour, Shapes, or BattleShip).
It then prints out the environment's ASCII state, the predicted policy distribution,
and the predicted value.

You can optionally override the default random initial state by interacting with
the environment prior to calling get_text_state(), if you like.

Usage:
  python run_pipeline.py --model_dir transformer_policy_value --env tictactoe
  python run_pipeline.py --model_dir transformer_policy_value --env connectfour
  python run_pipeline.py --model_dir transformer_policy_value --env shapes
  python run_pipeline.py --model_dir transformer_policy_value --env battleship
"""

import argparse
import os
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Import your environment classes:
from tictactoe import TicTacToe
from connectfour import ConnectFour
from shapes import Shapes
from battleship import BattleShip

# Registry so we can instantiate by name
ENV_REGISTRY = {
    "tictactoe": TicTacToe,
    "connectfour": ConnectFour,
    "shapes": Shapes,
    "battleship": BattleShip
}

class TransformerPolicyValueNet(torch.nn.Module):
    """
    Transformer-based model that maps an ASCII game state to:
      - policy_logits (up to 'max_action_dim')
      - value (scalar)
    Initialized from deepseek-ai/DeepSeek-R1-Distill-Qwen-7B, then fine-tuned.
    """
    def __init__(self, hf_model_name, max_action_dim=49):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        self.transformer = AutoModel.from_pretrained(hf_model_name)
        hidden_size = self.transformer.config.hidden_size

        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, max_action_dim)
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, ascii_batch):
        """
        ascii_batch: list of strings (ASCII states)
        Returns:
          policy_logits (B, max_action_dim)
          value (B,)
        """
        enc = self.tokenizer(ascii_batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, T, H)
        pooled = hidden_states.mean(dim=1)         # average pooling -> (B, H)

        policy_logits = self.policy_head(pooled)   # (B, max_action_dim)
        value = torch.tanh(self.value_head(pooled))# (B, 1), in [-1,1]
        return policy_logits, value.squeeze(dim=-1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="transformer_policy_value",
                        help="Directory containing the fine-tuned HF model and policy_value_head.pt")
    parser.add_argument("--max_action_dim", type=int, default=49,
                        help="Max action dimension used during training")
    parser.add_argument("--env", type=str, default="tictactoe",
                        help="Which environment to instantiate and run (tictactoe, connectfour, shapes, battleship)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load the HF base model from 'model_dir'
    base_model = AutoModel.from_pretrained(args.model_dir).to(device)

    # 2) Create the policy-value net and load the trained heads
    net = TransformerPolicyValueNet(args.model_dir, max_action_dim=args.max_action_dim).to(device)
    net.transformer = base_model  # overwrite the transformer's weights with the loaded base
    head_path = os.path.join(args.model_dir, "policy_value_head.pt")
    net.load_state_dict(torch.load(head_path, map_location=device), strict=False)
    net.eval()

    # 3) Instantiate the requested environment
    if args.env not in ENV_REGISTRY:
        print(f"Unknown environment '{args.env}'. Available: {list(ENV_REGISTRY.keys())}")
        return

    EnvClass = ENV_REGISTRY[args.env]
    env = EnvClass()  # This may randomize the state (shapes, battleship), or be empty (tictactoe, connectfour)
    # If needed, we can call env.reset_board() or env.reset_game()

    # 4) Get the ASCII representation from the environment
    ascii_state = env.get_text_state()

    # We'll define the actual_action_dim for slicing policy output.
    # This is environment-specific.
    if args.env == "tictactoe":
        actual_action_dim = 9  # 3x3
    elif args.env == "connectfour":
        actual_action_dim = 7  # columns
    elif args.env == "shapes":
        # It's typically len(env.answer_options), but let's do a safe upper bound:
        actual_action_dim = len(env.answer_options)
    elif args.env == "battleship":
        # If it's a 5x5 default => 25
        actual_action_dim = env.board_size * env.board_size
    else:
        actual_action_dim = args.max_action_dim  # fallback

    # 5) Inference
    with torch.no_grad():
        policy_logits, value = net([ascii_state])
        # policy_logits shape: (1, max_action_dim)
        # We only need the first 'actual_action_dim' entries
        policy_logits = policy_logits[:, :actual_action_dim]
        policy_probs = F.softmax(policy_logits, dim=-1)[0]  # shape (actual_action_dim,)
        value = value.item()  # scalar

    # 6) Print results
    print(f"=== Environment: {args.env} ===")
    print("ASCII state:")
    print(ascii_state)
    print("------")
    print(f"Predicted Value (P1 perspective, in [-1,1]): {value:.4f}")
    print("Predicted Policy Distribution:")
    for i, p in enumerate(policy_probs.tolist()):
        print(f"  Action {i}: {p:.4f}")

    best_action = torch.argmax(policy_probs).item()
    print(f"\nBest action index = {best_action}, Probability={policy_probs[best_action]:.4f}")

if __name__ == "__main__":
    main()
