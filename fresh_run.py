#!/usr/bin/env python3
"""Run a fresh ChildPlay set on the current Claude and GPT models.

The published 2024 logs are not replayed. One seeded draw builds the boards,
ship layouts, shape placements, and Lego constructs, and every model sees that
same draw. Opponent moves come from a per-episode seed, so a model that plays
the same moves meets the same opponent.

Claude Fable 5.2 and Opus 5.5 reject temperature. They are called once with
output effort set to low, which is what keeps thinking tokens off. gpt-6-astra
and gpt-6.1-sol accept only the default temperature and are called once at low
reasoning effort. gpt-6-luna accepts a temperature sweep, so it is run at 0,
0.5, and 1. The GPT dollar caps sum to $23 so parallel runs stay inside the
$25 credit.

Args:
    None. Configuration is the constants below and the command-line flags
    parsed in main.

Returns:
    None. Writes stimuli, a JSONL ledger, and summary.json under
    fresh_experiments/seed<SEED>/.
"""

import argparse
import contextlib
import copy
import io
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from lcl import LCLGame
from scripts_games.battleship import BattleShip
from scripts_games.connectfour import ConnectFour
from scripts_games.outcome import apply_move
from scripts_games.shapes import Shapes
from scripts_games.tictactoe import TicTacToe

SEED = 20261111
N_EPISODES = 100
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "fresh_experiments" / f"seed{SEED}"

RATES = {
    "gpt-4o-2024-08-06": (2.5 / 1e6, 10.0 / 1e6),
    "gpt-4o-mini-2024-07-18": (0.15 / 1e6, 0.60 / 1e6),
    "claude-fable-5-2": (8.0 / 1e6, 40.0 / 1e6),
    "claude-opus-5-5": (4.0 / 1e6, 20.0 / 1e6),
    "gpt-6-astra": (10.0 / 1e6, 50.0 / 1e6),
    "gpt-6.1-sol": (2.0 / 1e6, 10.0 / 1e6),
    "gpt-6-luna": (0.10 / 1e6, 0.50 / 1e6),
}
CAPS = {
    "gpt-4o-2024-08-06": 8.0,
    "gpt-4o-mini-2024-07-18": 1.0,
    "claude-fable-5-2": 40.0,
    "claude-opus-5-5": 25.0,
    "gpt-6-astra": 16.0,
    "gpt-6.1-sol": 5.0,
    "gpt-6-luna": 2.0,
}
CONDITIONS = [
    {"id": "claude-fable-5-2", "provider": "anthropic", "model": "claude-fable-5-2", "temperature": None},
    {"id": "claude-opus-5-5", "provider": "anthropic", "model": "claude-opus-5-5", "temperature": None},
    {"id": "gpt-6-astra", "provider": "openai", "model": "gpt-6-astra", "temperature": None},
    {"id": "gpt-6.1-sol", "provider": "openai", "model": "gpt-6.1-sol", "temperature": None},
    {"id": "gpt-6-luna-t0", "provider": "openai", "model": "gpt-6-luna", "temperature": 0.0},
    {"id": "gpt-6-luna-t0.5", "provider": "openai", "model": "gpt-6-luna", "temperature": 0.5},
    {"id": "gpt-6-luna-t1", "provider": "openai", "model": "gpt-6-luna", "temperature": 1.0},
]
AVAILABLE_LEGACY_CONDITIONS = [
    {"id": "gpt4o-t0", "provider": "openai", "model": "gpt-4o-2024-08-06", "temperature": 0.0},
    {"id": "gpt4o-mini-t0", "provider": "openai", "model": "gpt-4o-mini-2024-07-18", "temperature": 0.0},
]


def select_conditions(names):
    """Resolve an explicit model-condition selection without silent omissions."""
    available = {c["id"]: c for c in CONDITIONS + AVAILABLE_LEGACY_CONDITIONS}
    requested = names.split(",")
    unknown = set(requested) - available.keys()
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    return [available[name] for name in dict.fromkeys(requested)]

TASKS = ["shapes", "tictactoe", "lcl_validity", "lcl_construct", "connectfour", "battleship"]

_WRITE_LOCK = threading.Lock()
_BUDGET_LOCK = threading.Lock()
_SPENT = defaultdict(float)
_STOPPED = set()
_OPENAI = None


def _openai():
    """Return a shared OpenAI client.

    Returns:
        OpenAI: Client using OPENAI_API_KEY from the environment.
    """
    global _OPENAI
    if _OPENAI is None:
        _OPENAI = OpenAI(timeout=120, max_retries=0)
    return _OPENAI


def blocked(model):
    """Say whether a model has hit its dollar cap.

    Args:
        model: API model id.

    Returns:
        bool: True when no further calls should be sent for this model.
    """
    with _BUDGET_LOCK:
        return model in _STOPPED or _SPENT[model] >= CAPS[model]


def charge(model, cost):
    """Add a measured call cost and trip the cap when it is reached.

    Args:
        model: API model id.
        cost: Dollars for the call that just finished.

    Returns:
        float: Running spend for this model in dollars.
    """
    with _BUDGET_LOCK:
        _SPENT[model] += cost
        if _SPENT[model] >= CAPS[model]:
            _STOPPED.add(model)
        return _SPENT[model]


def claude_cost(model, usage):
    """Price one Claude Messages response.

    Args:
        model: Claude model id, used to pick the list rate.
        usage: Usage object from the Messages API.

    Returns:
        float: Dollars, including cache writes and reads when present.
    """
    in_rate, out_rate = RATES[model]
    fresh = usage.get("input_tokens", 0) or 0
    read = usage.get("cache_read_input_tokens", 0) or 0
    write = usage.get("cache_creation_input_tokens", 0) or 0
    read_mult = 0.025 if "fable" in model else 0.05
    output = usage.get("output_tokens", 0) or 0
    return fresh * in_rate + read * in_rate * read_mult + write * in_rate * 1.25 + output * out_rate


def gpt_cost(model, usage):
    """Price one OpenAI chat completion.

    Args:
        model: OpenAI model id.
        usage: Usage object from the chat completion.

    Returns:
        float: Dollars at the short-context list rate.
    """
    in_rate, out_rate = RATES[model]
    return usage.prompt_tokens * in_rate + usage.completion_tokens * out_rate


def call_model(condition, system, user, max_tokens):
    """Send one benchmark prompt and return the text plus measured cost.

    Retries rate limits and transport failures. A 400 from a bad parameter
    is raised, because retrying it would repeat the same rejection.

    Args:
        condition: Condition dict with provider, model, and temperature.
        system: System prompt, the game rules.
        user: User prompt, the current board.
        max_tokens: Hard ceiling on completion tokens, including thinking.

    Returns:
        tuple[str, dict]: The assistant text and a usage dict with keys
        input_tokens, output_tokens, and cost_usd.
    """
    model = condition["model"]
    last_error = None
    for attempt in range(5):
        if blocked(model):
            raise RuntimeError("budget cap reached")
        try:
            if condition["provider"] == "anthropic":
                text, usage, cost = _call_claude(model, system, user, max_tokens)
            else:
                text, usage, cost = _call_openai(condition, system, user, max_tokens)
            charge(model, cost)
            usage["cost_usd"] = cost
            return text, usage
        except RuntimeError:
            raise
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:500]
            if exc.code in (429, 500, 503, 529):
                last_error = RuntimeError(f"HTTP {exc.code}: {body}")
                time.sleep(min(2 ** attempt, 20))
                continue
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"retries exhausted: {last_error}")


def _call_claude(model, system, user, max_tokens):
    """Call the Claude Messages API once.

    Args:
        model: Claude model id.
        system: System prompt.
        user: User prompt.
        max_tokens: Completion ceiling.

    Returns:
        tuple[str, dict, float]: Text, raw usage, and dollars.
    """
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": "low"},
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "content-type": "application/json",
            "x-api-key": os.environ["CLAUDE_CLOUD_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        data = json.load(response)
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    usage = data.get("usage", {})
    return text.strip(), usage, claude_cost(model, usage)


def anthropic_limit_message():
    """Return the Claude account limit text when the key cannot be used.

    One short call is the only way to distinguish a monthly account cap
    from a rejected parameter. A usage-limit response is not retried.

    Returns:
        str | None: The provider message, or None when the call is accepted.
    """
    try:
        _call_claude("claude-fable-5-2", "Reply with the single word ok.", "ok", 16)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode(errors="replace")[:500]
        if exc.code == 400 and ("usage limits" in text or "regain access" in text):
            return text
        raise RuntimeError(f"HTTP {exc.code}: {text}") from exc
    return None


def _call_openai(condition, system, user, max_tokens):
    """Call the OpenAI chat completions API once.

    Args:
        condition: Condition dict. Luna carries a temperature. Sol does not.
        system: System prompt.
        user: User prompt.
        max_tokens: Completion ceiling, mapped to max_completion_tokens.

    Returns:
        tuple[str, dict, float]: Text, a plain usage dict, and dollars.
    """
    kwargs = {
        "model": condition["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_completion_tokens": max_tokens,
    }
    if condition["model"] == "gpt-6-luna":
        kwargs["temperature"] = condition["temperature"]
        kwargs["reasoning_effort"] = "none"
    elif condition["model"] in ("gpt-6-astra", "gpt-6.1-sol"):
        kwargs["reasoning_effort"] = "low"
    else:
        kwargs["temperature"] = condition["temperature"]
    completion = _openai().chat.completions.create(**kwargs)
    text = completion.choices[0].message.content or ""
    usage_obj = completion.usage
    usage = {
        "input_tokens": usage_obj.prompt_tokens,
        "output_tokens": usage_obj.completion_tokens,
    }
    return text.strip(), usage, gpt_cost(condition["model"], usage_obj)


def system_prompt(game):
    """Build the system prompt the existing harness sends.

    Args:
        game: A game instance with name and prompt.

    Returns:
        str: The system message.
    """
    return f"You are a player in a game of {game.name}. {game.prompt}."


def user_prompt(state):
    """Build the user turn the existing harness sends.

    Args:
        state: Text board from the game.

    Returns:
        str: The user message.
    """
    return f"Player 1 (LLM), it's your turn. Here's the current game state:\n{state}\nMy move is: "


def parse_pair(text, size):
    """Parse a row and column from a model reply.

    Args:
        text: Raw model text.
        size: Board width. Values outside it still parse, so the game can
            score them as illegal moves.

    Returns:
        tuple[int, int] | None: The two integers, or None when the reply is
        not exactly two integers.
    """
    parts = text.strip().split()
    if len(parts) != 2:
        return None
    try:
        row, col = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    return row, col


def parse_int(text):
    """Parse a reply that must be a single integer.

    Args:
        text: Raw model text.

    Returns:
        int | None: The integer, or None when any other text is present.
    """
    try:
        return int(text.strip())
    except ValueError:
        return None


def _blank_record(condition, task, episode_id):
    """Start an episode ledger row.

    Args:
        condition: Condition dict.
        task: Task name.
        episode_id: Episode id within the task.

    Returns:
        dict: Row with token counters at zero.
    """
    return {
        "condition": condition["id"],
        "model": condition["model"],
        "temperature": condition["temperature"],
        "task": task,
        "episode": episode_id,
        "winner": None,
        "correct": None,
        "invalid": False,
        "moves": [],
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "error": None,
    }


def _add_usage(record, usage):
    """Fold one call's usage into an episode row.

    Args:
        record: Episode ledger row.
        usage: Usage dict from call_model.

    Returns:
        None.
    """
    record["input_tokens"] += usage.get("input_tokens", 0) or 0
    record["output_tokens"] += usage.get("output_tokens", 0) or 0
    record["cost_usd"] += usage.get("cost_usd", 0.0) or 0.0


def ask_turn(condition, system, user, max_tokens, record):
    """Request one move and attach its cost to the episode.

    Args:
        condition: Condition dict.
        system: System prompt.
        user: User prompt.
        max_tokens: Completion ceiling.
        record: Episode ledger row updated in place.

    Returns:
        str | None: Model text, or None when the budget cap blocks the call.
    """
    if blocked(condition["model"]):
        record["error"] = "budget"
        return None
    text, usage = call_model(condition, system, user, max_tokens)
    _add_usage(record, usage)
    record["moves"].append(text)
    return text


def play_shapes(stimuli, condition, episode):
    """Score one frozen shape board.

    Args:
        stimuli: Full stimuli dict, used for the system prompt.
        condition: Condition dict.
        episode: One shapes item with text, options, and the true shape.

    Returns:
        dict: Episode ledger row.
    """
    record = _blank_record(condition, "shapes", episode["id"])
    text = ask_turn(condition, stimuli["prompts"]["shapes"], user_prompt(episode["text"]), 64, record)
    if text is None:
        return record
    choice = parse_int(text)
    options = episode["options"]
    if choice is None or not 0 <= choice < len(options):
        record["invalid"] = True
        record["winner"] = "P2"
        record["correct"] = False
        return record
    record["correct"] = options[choice] == episode["shape"]
    record["winner"] = "P1" if record["correct"] else "P2"
    return record


def _play_alternating(condition, record, game, system, legal_opponent, parse_model, max_turns):
    """Play one model-versus-seeded-random game to the end.

    Args:
        condition: Condition dict.
        record: Episode ledger row updated in place.
        game: Fresh game already set to the frozen opening.
        system: System prompt.
        legal_opponent: Callable returning the opponent's move, or None
            when the opponent has no legal move.
        parse_model: Callable mapping model text to a move or None.
        max_turns: Safety stop so a repeating model cannot spin.

    Returns:
        dict: The same ledger row.
    """
    turns = 0
    while not game.game_over and turns < max_turns:
        turns += 1
        if game.current_player == "P1":
            text = ask_turn(condition, system, user_prompt(game.get_text_state(0)), 256, record)
            if text is None:
                return record
            move = parse_model(text)
            if move is None:
                record["invalid"] = True
                record["winner"] = "P2"
                return record
            status = apply_move(game, 0, move)
        else:
            move = legal_opponent()
            if move is None:
                record["winner"] = "tie"
                return record
            status = apply_move(game, 1, move)
        if status == "invalid":
            record["invalid"] = True
            record["winner"] = "P2" if game.current_player == "P1" else "P1"
            return record
        if status == "win":
            record["winner"] = "P1" if game.current_player == "P1" else "P2"
            return record
        if status == "tie":
            record["winner"] = "tie"
            return record
    if record["winner"] is None:
        record["winner"] = "tie"
    return record


def play_tictactoe(stimuli, condition, episode):
    """Play one tic-tac-toe game from an empty board.

    Args:
        stimuli: Full stimuli dict.
        condition: Condition dict.
        episode: Episode dict with an opponent seed.

    Returns:
        dict: Episode ledger row.
    """
    record = _blank_record(condition, "tictactoe", episode["id"])
    game = TicTacToe(options={"board_size": 3})
    game.reset_board()
    rng = random.Random(episode["opponent_seed"])

    def opponent():
        empties = [(r, c) for r in range(3) for c in range(3) if game.board[r][c] == " "]
        return rng.choice(empties) if empties else None

    return _play_alternating(
        condition, record, game, stimuli["prompts"]["tictactoe"], opponent, lambda text: parse_pair(text, 3), 9
    )


def play_connectfour(stimuli, condition, episode):
    """Play one Connect Four game from an empty board.

    Args:
        stimuli: Full stimuli dict.
        condition: Condition dict.
        episode: Episode dict with an opponent seed.

    Returns:
        dict: Episode ledger row.
    """
    record = _blank_record(condition, "connectfour", episode["id"])
    game = ConnectFour(options={"board_size": 7})
    game.reset_board()
    rng = random.Random(episode["opponent_seed"])

    def opponent():
        columns = [col for col in range(game.cols) if game.board[0][col] == "."]
        return rng.choice(columns) if columns else None

    return _play_alternating(
        condition, record, game, stimuli["prompts"]["connectfour"], opponent, parse_int, 49
    )


def play_battleship(stimuli, condition, episode):
    """Play one battleship game from a frozen ship layout.

    The safety horizon is two shots for every cell. Each loop turn is
    one player, so a horizon of one board ended unfinished games as ties.

    Args:
        stimuli: Full stimuli dict.
        condition: Condition dict.
        episode: Episode dict with both fleets and an opponent seed.

    Returns:
        dict: Episode ledger row.
    """
    record = _blank_record(condition, "battleship", episode["id"])
    game = BattleShip()
    game.ship_board_p1 = copy.deepcopy(episode["ship_p1"])
    game.ship_board_p2 = copy.deepcopy(episode["ship_p2"])
    game.ships_p1 = [tuple(cell) for cell in episode["ships_p1"]]
    game.ships_p2 = [tuple(cell) for cell in episode["ships_p2"]]
    size = len(game.ship_board_p1)
    game.board_size = size
    game.guess_board_p1 = [["~" for _ in range(size)] for _ in range(size)]
    game.guess_board_p2 = [["~" for _ in range(size)] for _ in range(size)]
    game.game_over = False
    game.current_player = "P1"
    rng = random.Random(episode["opponent_seed"])

    def opponent():
        cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if game.guess_board_p2[r][c] == "~"
        ]
        return rng.choice(cells) if cells else None

    return _play_alternating(
        condition,
        record,
        game,
        stimuli["prompts"]["battleship"],
        opponent,
        lambda text: parse_pair(text, size),
        size * size * 2,
    )


def play_lcl_validity(stimuli, condition, episode):
    """Ask whether one frozen Lego construct is valid.

    Args:
        stimuli: Full stimuli dict.
        condition: Condition dict.
        episode: Episode dict with pieces and the true validity bit.

    Returns:
        dict: Episode ledger row.
    """
    record = _blank_record(condition, "lcl_validity", episode["id"])
    prompt = stimuli["prompts"]["lcl_validity"] + str(episode["pieces"])
    text = ask_turn(condition, "You answer questions about Lego structures.", prompt, 256, record)
    if text is None:
        return record
    answer = text.strip().lower()
    if answer not in ("valid", "invalid"):
        record["invalid"] = True
        record["correct"] = False
        record["winner"] = "P2"
        return record
    record["correct"] = (answer == "valid") == bool(episode["valid"])
    record["winner"] = "P1" if record["correct"] else "P2"
    return record


def play_lcl_construct(stimuli, condition, episode):
    """Ask for a valid 3-piece Lego construct and score it with the game rules.

    Args:
        stimuli: Full stimuli dict.
        condition: Condition dict.
        episode: Episode dict. The id distinguishes the sample. The prompt
            itself does not change.

    Returns:
        dict: Episode ledger row.
    """
    record = _blank_record(condition, "lcl_construct", episode["id"])
    text = ask_turn(condition, "You build Lego structures.", stimuli["prompts"]["lcl_construct"], 256, record)
    if text is None:
        return record
    pieces = _parse_pieces(text)
    if pieces is None:
        record["invalid"] = True
        record["correct"] = False
        record["winner"] = "P2"
        return record
    record["correct"] = len(pieces) == 3 and bool(stimuli["lcl_checker"].is_valid_construct(pieces))
    record["winner"] = "P1" if record["correct"] else "P2"
    record["pieces"] = pieces
    return record


def _parse_pieces(text):
    """Parse a model construct into a list of (x, y, color) tuples.

    Args:
        text: Raw model text, expected to be a Python list of triples.

    Returns:
        list[tuple] | None: The pieces, or None when the text is not that list.
    """
    candidate = text.strip()
    if not candidate.startswith("[") or not candidate.endswith("]"):
        return None
    try:
        parsed = json.loads(candidate.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            parsed = eval_literal(candidate)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(parsed, list) or not parsed:
        return None
    pieces = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return None
        x, y, color = item
        if type(x) is not int or type(y) is not int or not isinstance(color, str):
            return None
        pieces.append((x, y, color))
    return pieces


def eval_literal(text):
    """Parse a Python literal without executing it.

    Args:
        text: Source text.

    Returns:
        The literal value.
    """
    import ast
    return ast.literal_eval(text)


PLAYERS = {
    "shapes": play_shapes,
    "tictactoe": play_tictactoe,
    "connectfour": play_connectfour,
    "battleship": play_battleship,
    "lcl_validity": play_lcl_validity,
    "lcl_construct": play_lcl_construct,
}


def build_stimuli(n, seed):
    """Draw a fresh shared set.

    Global random is seeded per item because the game classes draw with the
    process random module. The returned object does not keep those generators.

    Args:
        n: Number of episodes per task. Shapes uses n boards for each of
            square, triangle, and cross.
        seed: Master seed.

    Returns:
        dict: Prompts and episodes. A live LCLGame is attached under
        lcl_checker for scoring and is not serialized.
    """
    shapes_game = Shapes(options={"board_size": 15}, shape="square")
    ttt = TicTacToe(options={"board_size": 3})
    c4 = ConnectFour(options={"board_size": 7})
    battle = BattleShip()
    checker = LCLGame()
    shapes = []
    for shape_index, shape in enumerate(["square", "triangle", "cross"]):
        for index in range(n):
            random.seed(seed + 1000 * shape_index + index)
            game = Shapes(options={"board_size": 15}, shape=shape)
            shapes.append({
                "id": f"{shape}-{index}",
                "shape": shape,
                "text": game.get_text_state(0),
                "options": list(game.answer_options),
            })
    fleets = []
    for index in range(n):
        random.seed(seed + 30000 + index)
        game = BattleShip()
        fleets.append({
            "id": index,
            "opponent_seed": seed + 40000 + index,
            "ship_p1": game.ship_board_p1,
            "ship_p2": game.ship_board_p2,
            "ships_p1": game.ships_p1,
            "ships_p2": game.ships_p2,
        })
    validity = []
    with contextlib.redirect_stdout(io.StringIO()):
        for index in range(n):
            random.seed(seed + 50000 + index)
            want_valid = index < n // 2
            pieces = checker.generate_valid_or_invalid_construct(5, valid=want_valid)
            validity.append({
                "id": index,
                "pieces": pieces,
                "valid": bool(checker.is_valid_construct(pieces)),
            })
    construct_prompt = (
        "A description of a Lego structure consists of a list of tuples, "
        "[(x1, y1, 'color1'), (x2, y2, 'color2')], where each tuple shows the coordinates "
        "and colors of a piece. Such a structure is valid if all Lego pieces are connected "
        "but not overlapping. A Lego piece is connected through interlocking pegs, not by "
        "merely touching sides. Two Lego pieces overlap when they share the same y-coordinate "
        "and any part of their length has the same x-coordinate. Produce a description of a valid "
        "structure using 3 Lego pieces. Reply only with the Lego structure description "
        "following the format [(x1, y1, 'color1'), (x2, y2, 'color2'), ...], write nothing else "
        "but the structure."
    )
    validity_prompt = (
        "You will receive a description of a Lego structure, for instance, [(x1, y1, 'color1'), "
        "(x2, y2, 'color2')], which lists the coordinates and colors of two pieces. A construct is "
        "valid if all Lego pieces are connected but not overlapping. A Lego piece is connected through "
        "interlocking pegs, not by merely touching sides. Two Lego pieces overlap when they share the "
        "same y-coordinate and any part of their length has the same x-coordinate. If the following "
        "structure is valid then reply with valid, otherwise reply with invalid (do not write anything else): "
    )
    return {
        "seed": seed,
        "n": n,
        "prompts": {
            "shapes": system_prompt(shapes_game),
            "tictactoe": system_prompt(ttt),
            "connectfour": system_prompt(c4),
            "battleship": system_prompt(battle),
            "lcl_construct": construct_prompt,
            "lcl_validity": validity_prompt,
        },
        "shapes": shapes,
        "tictactoe": [{"id": i, "opponent_seed": seed + 10000 + i} for i in range(n)],
        "connectfour": [{"id": i, "opponent_seed": seed + 20000 + i} for i in range(n)],
        "battleship": fleets,
        "lcl_validity": validity,
        "lcl_construct": [{"id": i} for i in range(n)],
        "lcl_checker": checker,
    }


def stimuli_for_disk(stimuli):
    """Copy stimuli without the live checker object.

    Args:
        stimuli: In-memory stimuli dict.

    Returns:
        dict: JSON-serializable stimuli.
    """
    stored = dict(stimuli)
    stored.pop("lcl_checker", None)
    return stored


def load_or_build(n, seed):
    """Load the frozen set, building it the first time.

    Args:
        n: Episodes per task.
        seed: Master seed.

    Returns:
        dict: Stimuli, with a live LCLGame attached for scoring.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "stimuli.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored.get("seed") != seed or stored.get("n") != n:
            raise RuntimeError(f"{path} was built with a different seed or size")
    else:
        stored = stimuli_for_disk(build_stimuli(n, seed))
        path.write_text(json.dumps(stored))
    stored["lcl_checker"] = LCLGame()
    return stored


def ledger_path():
    """Return the JSONL ledger path.

    Returns:
        Path: Append-only episode results.
    """
    return OUT_DIR / "results.jsonl"


def done_keys():
    """Read episode keys already written.

    Returns:
        set[str]: Keys of the form condition|task|episode.
    """
    path = ledger_path()
    if not path.exists():
        return set()
    found = set()
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            found.add(f"{row['condition']}|{row['task']}|{row['episode']}")
    return found


def append_row(row):
    """Append one episode row.

    Args:
        row: Ledger dict. The live pieces list is converted by the json encoder.

    Returns:
        None.
    """
    with _WRITE_LOCK:
        with ledger_path().open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()


def summarize():
    """Aggregate the ledger into win rates and spend.

    Returns:
        dict: Per condition and task, plus spend by model.
    """
    buckets = defaultdict(lambda: {
        "n": 0, "p1": 0, "p2": 0, "tie": 0, "invalid": 0, "correct": 0,
        "errors": 0, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0,
    })
    path = ledger_path()
    if path.exists():
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                bucket = buckets[f"{row['condition']}|{row['task']}"]
                bucket["n"] += 1
                bucket["cost_usd"] += row.get("cost_usd") or 0.0
                bucket["input_tokens"] += row.get("input_tokens") or 0
                bucket["output_tokens"] += row.get("output_tokens") or 0
                if row.get("error"):
                    bucket["errors"] += 1
                if row.get("invalid"):
                    bucket["invalid"] += 1
                if row.get("correct") is True:
                    bucket["correct"] += 1
                winner = row.get("winner")
                if winner == "P1":
                    bucket["p1"] += 1
                elif winner == "P2":
                    bucket["p2"] += 1
                elif winner == "tie":
                    bucket["tie"] += 1
    spend = defaultdict(float)
    for key, bucket in buckets.items():
        model = key.split("|", 1)[0]
        for condition in CONDITIONS:
            if condition["id"] == model:
                model = condition["model"]
        spend[model] += bucket["cost_usd"]
    report = {
        "tasks": {key: bucket for key, bucket in sorted(buckets.items())},
        "spend_usd": dict(spend),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2))
    return report


def load_battleship_clause(source, destination):
    """Load the frozen boards and attach the loss clause to Battleship only.

    The published seed directory is read and never written. The copy keeps
    every fleet. Only the Battleship system prompt changes, from the live
    game class, which ends in ``else you lose.``

    Args:
        source: Directory that holds the published stimuli.json.
        destination: New directory for this rerun. Created if missing.

    Returns:
        dict: Stimuli with a live LCL checker and the clause prompt.

    Raises:
        RuntimeError: The live prompt has no clause, the published prompt
            already has one, or the destination prompt does not match.
    """
    destination.mkdir(parents=True, exist_ok=True)
    dest_path = destination / "stimuli.json"
    published = json.loads((source / "stimuli.json").read_text())
    if dest_path.exists():
        stored = json.loads(dest_path.read_text())
    else:
        probe = BattleShip()
        clause = system_prompt(probe)
        if "else you lose." not in clause:
            raise RuntimeError("live battleship prompt has no loss clause")
        old = published["prompts"]["battleship"]
        if "else you lose." in old:
            raise RuntimeError("published battleship prompt already has the loss clause")
        if "row col values." not in old:
            raise RuntimeError("published battleship prompt does not end at row col values")
        stored = dict(published)
        stored["prompts"] = dict(published["prompts"])
        stored["prompts"]["battleship"] = clause
        dest_path.write_text(json.dumps(stored))
    if stored["prompts"]["battleship"] == published["prompts"]["battleship"]:
        raise RuntimeError("clause copy still has the published battleship prompt")
    if "else you lose." not in stored["prompts"]["battleship"]:
        raise RuntimeError("clause copy is missing else you lose")
    if len(stored["battleship"]) != len(published["battleship"]):
        raise RuntimeError("clause copy changed the number of frozen fleets")
    stored["lcl_checker"] = LCLGame()
    return stored


def load_lcl_rules(source, destination):
    """Reuse frozen constructs while making their exact geometry explicit."""
    published = json.loads((source / "stimuli.json").read_text())
    rules = (
        "Coordinates x and y must be integers and the color must be a string. "
        "Every brick occupies the four consecutive x positions x, x+1, x+2, x+3 "
        "at its y level. Bricks overlap only if their y values are equal and "
        "their occupied x positions intersect. Bricks connect only if their y "
        "values differ by exactly one and their occupied x positions intersect. "
        "All bricks must form one connected component; side touching does not connect them. "
    )
    stored = dict(published)
    stored["prompts"] = dict(published["prompts"])
    stored["prompts"]["lcl_validity"] = (
        rules + "Decide whether the following list of (x, y, color) tuples is valid. "
        "Reply with only valid or invalid, without any explanation: "
    )
    stored["prompts"]["lcl_construct"] = (
        rules + "Produce one valid construct using exactly three bricks. "
        "Reply only with a list of three tuples in the format "
        "[(x1, y1, 'color1'), (x2, y2, 'color2'), (x3, y3, 'color3')]. "
        "Write no explanation or code fence."
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "stimuli.json"
    if path.exists() and json.loads(path.read_text()) != stored:
        raise RuntimeError("existing LCL run uses different stimuli or instructions")
    if not path.exists():
        path.write_text(json.dumps(stored))
    stored["lcl_checker"] = LCLGame()
    return stored


def run(workers, tasks, stimuli=None, episode_limit=None, prior_spend=None):
    """Play every unfinished episode.

    Claude conditions are left out when the account usage limit rejects
    a probe. Those episodes stay unwritten and can be resumed later.
    A turn blocked by the dollar cap is not a finished episode, so that
    row is left unwritten too.

    Args:
        workers: Concurrent episodes.
        tasks: Task names to run, in order.
        stimuli: Preloaded stimuli. When omitted, the published seed is loaded.
        episode_limit: When set, only episode ids below this value are played.
        prior_spend: Dollars already spent per model, counted toward the caps.

    Returns:
        dict: The summary written at the end.
    """
    load_dotenv(ROOT / ".env")
    if stimuli is None:
        stimuli = load_or_build(N_EPISODES, SEED)
    conditions = CONDITIONS
    if any(condition["provider"] == "anthropic" for condition in CONDITIONS):
        limit = anthropic_limit_message()
        if limit:
            print(f"claude unavailable, skipping fable and opus: {limit}", flush=True)
            conditions = [condition for condition in CONDITIONS if condition["provider"] != "anthropic"]
    prior = summarize()
    for model, amount in prior["spend_usd"].items():
        _SPENT[model] = amount
        if amount >= CAPS[model]:
            _STOPPED.add(model)
    if prior_spend:
        with _BUDGET_LOCK:
            for model, amount in prior_spend.items():
                _SPENT[model] += float(amount)
                if model in CAPS and _SPENT[model] >= CAPS[model]:
                    _STOPPED.add(model)
    finished = done_keys()
    jobs = []
    for task in tasks:
        for condition in conditions:
            for episode in stimuli[task]:
                if episode_limit is not None and int(str(episode["id"]).rsplit("-", 1)[-1]) >= episode_limit:
                    continue
                key = f"{condition['id']}|{task}|{episode['id']}"
                if key not in finished:
                    jobs.append((task, condition, episode))
    print(f"episodes to run: {len(jobs)}", flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(PLAYERS[task], stimuli, condition, episode) for task, condition, episode in jobs]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as exc:
                print(f"episode failed: {exc}", flush=True)
                continue
            if row.get("error") == "budget":
                print(
                    f"budget stop {row.get('condition')} {row.get('task')} episode {row.get('episode')}",
                    flush=True,
                )
                continue
            append_row(row)
            completed += 1
            if completed % 20 == 0:
                report = summarize()
                spent = ", ".join(f"{name} ${amount:.2f}" for name, amount in sorted(report["spend_usd"].items()))
                print(f"finished {completed}/{len(jobs)} spend {spent}", flush=True)
    return summarize()


def self_test():
    """Check that the fresh set is stable and that random games finish.

    Returns:
        None. Raises AssertionError when a check fails.
    """
    first = build_stimuli(4, SEED)
    second = build_stimuli(4, SEED)
    assert stimuli_for_disk(first) == stimuli_for_disk(second)
    assert parse_pair("1 1", 3) == (1, 1)
    assert parse_pair("1 1 extra", 3) is None
    assert parse_int("4") == 4
    assert parse_int("4.") is None
    left = play_tictactoe_random(first["tictactoe"][0]["opponent_seed"])
    right = play_tictactoe_random(first["tictactoe"][0]["opponent_seed"])
    assert left == right
    pieces = _parse_pieces("[(0, 0, 'red'), (0, 1, 'blue')]")
    assert pieces == [(0, 0, "red"), (0, 1, "blue")]
    assert first["lcl_checker"].is_valid_construct(pieces) is True
    assert battleship_repeat_keeps_the_hit() == "X"
    print("self-test ok", flush=True)


def battleship_repeat_keeps_the_hit():
    """Fire twice at the same ship cell and return the mark that remains.

    The first shot records a hit. The second shot is illegal. The hit has
    to stay on the board, because clearing it unsinks the ship.

    Returns:
        str: The guess-board mark at that cell after the rejected shot.
    """
    game = BattleShip()
    size = game.board_size
    game.ship_board_p2 = [["~" for _ in range(size)] for _ in range(size)]
    game.guess_board_p1 = [["~" for _ in range(size)] for _ in range(size)]
    game.ship_board_p2[0][0] = "S"
    game.ship_board_p2[0][1] = "S"
    game.ships_p2 = [(0, 0), (0, 1)]
    game.ships_p1 = [(1, 1)]
    game.ship_board_p1 = [["~" for _ in range(size)] for _ in range(size)]
    game.ship_board_p1[1][1] = "S"
    game.game_over = False
    game.current_player = "P1"
    assert apply_move(game, 0, (0, 0)) == "continue"
    assert game.guess_board_p1[0][0] == "X"
    game.current_player = "P1"
    assert apply_move(game, 0, (0, 0)) == "invalid"
    return game.guess_board_p1[0][0]


def play_tictactoe_random(seed):
    """Play a fully random tic-tac-toe game.

    Args:
        seed: Opponent and first-player seed. Both sides share it so the
            game is a pure function of the seed.

    Returns:
        str: Winner label, P1, P2, or tie.
    """
    game = TicTacToe(options={"board_size": 3})
    game.reset_board()
    rng = random.Random(seed)
    while not game.game_over:
        empties = [(r, c) for r in range(3) for c in range(3) if game.board[r][c] == " "]
        player = 0 if game.current_player == "P1" else 1
        status = apply_move(game, player, rng.choice(empties))
        if status == "win":
            return "P1" if player == 0 else "P2"
        if status == "tie":
            return "tie"
    return "tie"


def main():
    """Parse flags and either self-test, summarize, or run the benchmark.

    Returns:
        None.
    """
    global OUT_DIR, CONDITIONS
    parser = argparse.ArgumentParser(description="Fresh ChildPlay run for Fable, Opus, Sol, and Luna.")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--conditions", help="Comma-separated condition IDs for this run")
    parser.add_argument("--battleship-clause", action="store_true")
    parser.add_argument("--lcl-rules", action="store_true")
    args = parser.parse_args()
    if args.conditions:
        CONDITIONS = select_conditions(args.conditions)
    if args.battleship_clause and args.lcl_rules:
        raise SystemExit("choose one prompt revision per run")
    if args.self_test:
        self_test()
        return
    stimuli = None
    prior_spend = None
    if args.lcl_rules:
        if not args.out_dir:
            raise SystemExit("an LCL prompt revision needs --out-dir")
        source = OUT_DIR
        destination = Path(args.out_dir)
        if destination.resolve() == source.resolve():
            raise SystemExit("refusing to overwrite original stimuli")
        stimuli = load_lcl_rules(source, destination)
        OUT_DIR = destination
    elif args.battleship_clause:
        if not args.out_dir:
            raise SystemExit("a battleship clause rerun needs --out-dir")
        destination = Path(args.out_dir)
        if destination.resolve() == OUT_DIR.resolve():
            raise SystemExit("refusing to write the clause rerun over the published seed")
        source = OUT_DIR
        OUT_DIR = destination
        stimuli = load_battleship_clause(source, destination)
        published_summary = source / "summary.json"
        if published_summary.exists():
            prior_spend = json.loads(published_summary.read_text()).get("spend_usd", {})
        ending = stimuli["prompts"]["battleship"][-48:]
        print(f"battleship prompt ends: {ending}", flush=True)
        CAPS["gpt-6-astra"] = 29.0
        CAPS["gpt-6.1-sol"] = 8.0
        print(
            "caps: " + ", ".join(f"{name}={amount:.1f}" for name, amount in CAPS.items()),
            flush=True,
        )
    elif args.out_dir:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.summarize:
        report = summarize()
        print(json.dumps(report["spend_usd"], indent=2))
        return
    tasks = [task for task in args.tasks.split(",") if task]
    report = run(
        args.workers,
        tasks,
        stimuli=stimuli,
        episode_limit=args.limit or None,
        prior_spend=prior_spend,
    )
    print(json.dumps(report["spend_usd"], indent=2), flush=True)


if __name__ == "__main__":
    main()
