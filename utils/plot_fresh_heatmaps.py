"""Move heatmaps for the models evaluated on the 2026 ChildPlay items.

The ledger stores the model's replies, not the random opponent's moves.
Each game is replayed with that episode's opponent seed, and only moves
the game accepted are counted. The two panels match the appendix figures:
the model on the left and the random opponent on the right.

Args:
    None. Tic-tac-toe, Connect-Four, and the other tasks are read from
    fresh_experiments/seed20261111. Battleship is read from
    fresh_experiments/seed20261111_bs_clause.

Returns:
    None. Writes one PDF per model and task under paper/, and
    fresh_experiments/seed20261111/fresh_move_summary.json.
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from fresh_run import apply_move, parse_int, parse_pair
from scripts_games.battleship import BattleShip
from scripts_games.connectfour import ConnectFour
from scripts_games.tictactoe import TicTacToe

ROOT = Path(__file__).resolve().parents[1]
SEED_DIR = ROOT / "fresh_experiments" / "seed20261111"
CLAUSE_DIR = ROOT / "fresh_experiments" / "seed20261111_bs_clause"
OUT_DIR = ROOT / "paper"
BATTLESHIP_CONDITIONS = (
    "claude-fable-5-2",
    "claude-opus-5-5",
    "gpt-6-astra",
    "gpt-6.1-sol",
    "gpt-6-luna-t0",
    "gpt-6-luna-t0.5",
    "gpt-6-luna-t1",
)

ORDER = [
    ("claude-fable-5-2", "Fable 5.2", None),
    ("claude-opus-5-5", "Opus 5.5", None),
    ("gpt-6-astra", "Astra", None),
    ("gpt-6.1-sol", "Sol", None),
    ("gpt-6-luna-t0", "Luna", 0.0),
    ("gpt-6-luna-t0.5", "Luna", 0.5),
    ("gpt-6-luna-t1", "Luna", 1.0),
]
TASKS = ("tictactoe", "connectfour", "battleship")
NAMES = {
    "tictactoe": "tic-tac-toe",
    "connectfour": "Connect-Four",
    "battleship": "Battleship",
}


def _read_jsonl(path):
    """Read one JSON object from each non-empty line.

    Args:
        path: JSONL file.

    Returns:
        list[dict]: The rows in file order.
    """
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_rows():
    """Read the 2026 ledger, with Battleship taken from the clause rerun.

    Tic-tac-toe, Connect-Four, shapes, LCL, and every other task stay on
    fresh_experiments/seed20261111/results.jsonl. Battleship rows are
    replaced by fresh_experiments/seed20261111_bs_clause/results.jsonl,
    the rerun of the same fleets after ``else you lose.`` was added.
    The published ledger is only read.

    Returns:
        list[dict]: One row per episode.

    Raises:
        ValueError: The clause file is not 100 Battleship rows for each
        of the seven 2026 conditions.
    """
    published = _read_jsonl(SEED_DIR / "results.jsonl")
    clause = _read_jsonl(CLAUSE_DIR / "results.jsonl")
    if any(row.get("task") != "battleship" for row in clause):
        raise ValueError("the clause ledger contains a row that is not battleship")
    counts = defaultdict(int)
    for row in clause:
        counts[row["condition"]] += 1
    if set(counts) != set(BATTLESHIP_CONDITIONS) or any(counts[name] != 100 for name in BATTLESHIP_CONDITIONS):
        raise ValueError(f"clause battleship counts are {dict(counts)}")
    return [row for row in published if row.get("task") != "battleship"] + clause


def load_stimuli():
    """Read the frozen boards and opponent seeds.

    Returns:
        dict: Stimuli keyed by task.
    """
    return json.loads((SEED_DIR / "stimuli.json").read_text())


def prepare_game(task, episode):
    """Build the game and the opponent used for this episode.

    Args:
        task: tictactoe, connectfour, or battleship.
        episode: Frozen episode dict.

    Returns:
        tuple: The game and a zero-argument opponent callable.
    """
    rng = random.Random(episode["opponent_seed"])
    if task == "tictactoe":
        game = TicTacToe(options={"board_size": 3})
        game.reset_board()

        def opponent():
            empties = [(r, c) for r in range(3) for c in range(3) if game.board[r][c] == " "]
            return rng.choice(empties) if empties else None

        return game, opponent
    if task == "connectfour":
        game = ConnectFour(options={"board_size": 7})
        game.reset_board()

        def opponent():
            columns = [col for col in range(game.cols) if game.board[0][col] == "."]
            return rng.choice(columns) if columns else None

        return game, opponent
    game = BattleShip()
    game.ship_board_p1 = [list(row) for row in episode["ship_p1"]]
    game.ship_board_p2 = [list(row) for row in episode["ship_p2"]]
    game.ships_p1 = [tuple(cell) for cell in episode["ships_p1"]]
    game.ships_p2 = [tuple(cell) for cell in episode["ships_p2"]]
    size = len(game.ship_board_p1)
    game.board_size = size
    game.guess_board_p1 = [["~" for _ in range(size)] for _ in range(size)]
    game.guess_board_p2 = [["~" for _ in range(size)] for _ in range(size)]
    game.game_over = False
    game.current_player = "P1"

    def opponent():
        cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if game.guess_board_p2[r][c] == "~"
        ]
        return rng.choice(cells) if cells else None

    return game, opponent


def parse_model_move(task, text):
    """Parse one stored model reply the way the runner did.

    Args:
        task: Game name.
        text: Raw reply stored in the ledger.

    Returns:
        The move, or None when the reply is not a legal format.
    """
    if task == "connectfour":
        return parse_int(text)
    size = 3 if task == "tictactoe" else 5
    return parse_pair(text, size)


def replay(task, episode, texts):
    """Recover accepted model and opponent moves for one episode.

    The turn horizon matches the runner, so a game that stopped on the
    safety limit is not given an extra opponent move.

    Args:
        task: Game name.
        episode: Frozen episode dict.
        texts: Model replies in the order they were sent.

    Returns:
        tuple[list, list, str]: Accepted model moves, accepted opponent
        moves, and the replayed winner.
    """
    game, opponent = prepare_game(task, episode)
    horizons = {"tictactoe": 9, "connectfour": 49, "battleship": 50}
    model_moves = []
    opponent_moves = []
    winner = None
    turns = 0
    for text in texts:
        if game.game_over or turns >= horizons[task]:
            break
        turns += 1
        move = parse_model_move(task, text)
        if move is None:
            winner = "P2"
            break
        status = apply_move(game, 0, move)
        if status == "invalid":
            winner = "P2" if game.current_player == "P1" else "P1"
            break
        model_moves.append(move)
        if status == "win":
            winner = "P1" if game.current_player == "P1" else "P2"
            break
        if status == "tie":
            winner = "tie"
            break
        if turns >= horizons[task]:
            break
        turns += 1
        reply = opponent()
        if reply is None:
            winner = "tie"
            break
        status = apply_move(game, 1, reply)
        if status == "invalid":
            winner = "P2" if game.current_player == "P1" else "P1"
            break
        opponent_moves.append(reply)
        if status == "win":
            winner = "P1" if game.current_player == "P1" else "P2"
            break
        if status == "tie":
            winner = "tie"
            break
    if winner is None:
        winner = "tie"
    return model_moves, opponent_moves, winner


def accumulate(task, moves, board):
    """Add accepted moves to a count grid.

    Args:
        task: Game name.
        moves: Column integers, or row-column pairs.
        board: Array updated in place.

    Returns:
        None.
    """
    for move in moves:
        if task == "connectfour":
            board[0, move] += 1
        else:
            row, col = move
            board[row, col] += 1


def percentages(board):
    """Convert counts to a share of that player's moves.

    Args:
        board: Count array.

    Returns:
        numpy.ndarray: Percentages. An empty board stays at zero.
    """
    total = board.sum()
    if total == 0:
        return board
    return board / total * 100


def draw(task, label, temperature, model_board, opponent_board, path):
    """Write one two-panel move heatmap.

    Args:
        task: Game name.
        label: Model name printed in the title.
        temperature: Temperature, or None for a single evaluation.
        model_board: Model move counts.
        opponent_board: Opponent move counts.
        path: PDF path.

    Returns:
        None.
    """
    panels = [percentages(model_board), percentages(opponent_board)]
    titles = [f"{label}", "Random opponent"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.2))
    for axis, panel, title in zip(axes, panels, titles):
        sns.heatmap(panel, ax=axis, annot=True, cmap="coolwarm", fmt=".1f", cbar=False)
        axis.set_title(title)
        axis.set_xlabel("Column")
        if task == "connectfour":
            axis.set_yticks([])
        else:
            axis.set_ylabel("Row")
    setting = "single evaluation" if temperature is None else f"t = {temperature:g}"
    fig.suptitle(f"Distribution of moves in {NAMES[task]}, {label}, {setting}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main(complete=False):
    """Replay the ledger, write heatmaps, and write the move summary.

    Returns:
        None.
    """
    global OUT_DIR
    if complete:
        from utils.paper_dataset import load_dataset, LABELS, CONDITIONS
        rows = load_dataset()
        order = [(c['id'], LABELS[c['id']], c['temperature']) for c in CONDITIONS]
        OUT_DIR = ROOT / 'paper' / 'complete'
    else:
        rows = load_rows()
        order = ORDER
    stimuli = load_stimuli()
    episodes = {task: {item["id"]: item for item in stimuli[task]} for task in TASKS}
    buckets = defaultdict(lambda: {"model": None, "opponent": None, "replies": 0, "illegal": 0, "games": 0, "mismatch": 0})
    for row in rows:
        task = row["task"]
        if task not in TASKS:
            continue
        key = (task, row["condition"])
        bucket = buckets[key]
        if bucket["model"] is None:
            if task == "connectfour":
                shape = (1, 7)
            else:
                shape = (3, 3) if task == "tictactoe" else (5, 5)
            bucket["model"] = np.zeros(shape)
            bucket["opponent"] = np.zeros(shape)
        episode = episodes[task][int(row["episode"])]
        model_moves, opponent_moves, winner = replay(task, episode, row["moves"])
        accumulate(task, model_moves, bucket["model"])
        accumulate(task, opponent_moves, bucket["opponent"])
        bucket["games"] += 1
        bucket["replies"] += len(row["moves"])
        bucket["illegal"] += int(row["invalid"])
        bucket["mismatch"] += int(winner != row["winner"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}
    mismatches = 0
    for condition, label, temperature in order:
        for task in TASKS:
            bucket = buckets[(task, condition)]
            mismatches += bucket["mismatch"]
            slug = f"fresh_{task}_{condition}_heatmap.pdf"
            draw(task, label, temperature, bucket["model"], bucket["opponent"], OUT_DIR / slug)
            summary[f"{task}|{condition}"] = {
                "games": bucket["games"],
                "replies": bucket["replies"],
                "illegal_games": bucket["illegal"],
                "mismatch": bucket["mismatch"],
                "illegal_game_percent": round(100 * bucket["illegal"] / bucket["games"], 1) if bucket["games"] else None,
            }
    summary_path = OUT_DIR / 'move_summary.json' if complete else SEED_DIR / 'fresh_move_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2))
    print("mismatches", mismatches)
    print("wrote", len(summary), "heatmaps")


if __name__ == "__main__":
    main()
