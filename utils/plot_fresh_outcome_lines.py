"""Appendix outcome figures for the 2024 sweep and the 2026 models.

The temperature plots contain only GPT-3.5, GPT-4, GPT-4o, and GPT-4o-mini.
Fable, Opus, Astra, Sol, and Luna are the same line charts in a separate
figure. The best 2024 result on each panel is a tick on the right axis,
not another line. Outcome panels carry the standard error of a 0/1 trial,
SE (%) = sigma / sqrt(n) * 100 with n = 100. Tactical rates use before/after move checks and explicit episode boundaries.
No values are inferred from published figure strokes.

Returns:
    None. Writes the PDFs included by the appendix.
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import count_blocking_moves_connect_four as connectfour_counter
import count_blocking_moves_tictactoe as tictactoe_counter
from plot_fresh_heatmaps import load_rows, load_stimuli, replay
from utils.board_move_metrics import episodes_from_messages

ROOT = Path(__file__).resolve().parents[1]
OLD_DIR = ROOT / "experiment_board_games"
OUT_DIR = ROOT / "paper"
OLD_MODELS = ["gpt3_5", "gpt4", "gpt4o", "gpt4o_mini"]
OLD_CONDITIONS = ["temp_0", "temp_0.5", "temp_1", "temp_1.5"]
OLD_COLORS = {
    "gpt3_5": "#4c72b0",
    "gpt4": "#dd8452",
    "gpt4o": "#55a868",
    "gpt4o_mini": "#c44e52",
}
FRESH_AXIS = [
    ("claude-fable-5-2", "Fable 5.2"),
    ("claude-opus-5-5", "Opus 5.5"),
    ("gpt-6-astra", "Astra"),
    ("gpt-6.1-sol", "Sol"),
    ("gpt-6-luna-t0", "Luna 0"),
    ("gpt-6-luna-t0.5", "Luna 0.5"),
    ("gpt-6-luna-t1", "Luna 1"),
]
MODEL_COLOR = "#086788"
RANDOM_COLOR = "#FF7F0E"
REFERENCE_NAMES = {
    "gpt3_5": "GPT-3.5",
    "gpt4": "GPT-4",
    "gpt4o": "GPT-4o",
    "gpt4o_mini": "GPT-4o-mini",
}
METRICS = ["Wins", "Wins (Random Player)", "Incorrect Moves", "Ties"]
MEASURES = ["Average Moves", "Missed Wins", "Missed Blocks"]
EXPECTED = {
    ("battleship", "claude-fable-5-2"): (0, 100),
    ("battleship", "claude-opus-5-5"): (2, 98),
    ("battleship", "gpt-6-astra"): (63, 37),
    ("battleship", "gpt-6.1-sol"): (56, 44),
    ("battleship", "gpt-6-luna-t0"): (1, 99),
    ("battleship", "gpt-6-luna-t0.5"): (1, 99),
    ("battleship", "gpt-6-luna-t1"): (1, 99),
    ("tictactoe", "claude-fable-5-2"): (35, 65),
    ("tictactoe", "claude-opus-5-5"): (85, 15),
    ("tictactoe", "gpt-6-astra"): (73, 27),
    ("tictactoe", "gpt-6.1-sol"): (62, 38),
    ("tictactoe", "gpt-6-luna-t0"): (68, 26),
    ("tictactoe", "gpt-6-luna-t0.5"): (69, 26),
    ("tictactoe", "gpt-6-luna-t1"): (73, 22),
    ("connectfour", "claude-fable-5-2"): (37, 63),
    ("connectfour", "claude-opus-5-5"): (21, 79),
    ("connectfour", "gpt-6-astra"): (77, 23),
    ("connectfour", "gpt-6.1-sol"): (75, 25),
    ("connectfour", "gpt-6-luna-t0"): (80, 20),
    ("connectfour", "gpt-6-luna-t0.5"): (80, 20),
    ("connectfour", "gpt-6-luna-t1"): (78, 22),
}


def load_old_counts(game):
    """Read the published win, loss, tie, and incorrect-move counts.

    Args:
        game: battleship, tictactoe, or connectfour.

    Returns:
        pandas.DataFrame: One row per model and temperature.
    """
    rows = []
    for model in OLD_MODELS:
        for condition in OLD_CONDITIONS:
            path = OLD_DIR / f"experiment_{game}_{model}_oneshot_{condition}" / f"results_{game}.json"
            stored = json.loads(path.read_text())
            rows.append({
                "Model": model,
                "Temperature": float(condition.split("_")[1]),
                "Wins": stored["P1 Wins"],
                "Wins (Random Player)": stored["P2 Wins"],
                "Ties": stored["Ties"],
                "Incorrect Moves": stored["P1 Wrong Moves"],
            })
    return pd.DataFrame(rows)


def _binary_se_percent(flags):
    """Standard error of a 0/1 outcome, in percentage points.

    The sample standard deviation uses the n-1 divisor. The LCL table
    defines SE (%) = sigma / sqrt(n) * 100. At n = 100 that percentage
    has the same scale as a count of games.

    Args:
        flags: Per-trial outcomes, each 0 or 1.

    Returns:
        float: Standard error in percentage points.
    """
    values = [float(flag) for flag in flags]
    n = len(values)
    if n < 2:
        raise ValueError("a standard error needs at least two trials")
    mean = sum(values) / n
    sigma = (sum((value - mean) ** 2 for value in values) / (n - 1)) ** 0.5
    return sigma / (n ** 0.5) * 100.0


def _trial_outcome(messages):
    """Read one game's win, loss, tie, and invalid-move flags.

    The decisive sentence is the game-over line or the default-win line.
    A following line that says the game ended unexpectedly does not erase
    that sentence. The results summary leaves some of those default losses
    out of P2 Wins.

    Args:
        messages: Stored messages for one game.

    Returns:
        tuple[int, int, int, int]: Model win, random-player win, tie, invalid.
    """
    model_win = random_win = tie = invalid = 0
    for message in messages:
        lower = " ".join(str(message).split()).lower()
        if "wins by default" in lower:
            invalid = 1
            model_win = random_win = tie = 0
            before = lower.split("wins by default", 1)[0]
            if "random" in before:
                random_win = 1
            elif "llm" in before:
                model_win = 1
        elif lower.startswith("game over") and "unexpected" not in lower:
            model_win = random_win = tie = 0
            if "tie" in lower:
                tie = 1
            elif "llm" in lower and "wins" in lower:
                model_win = 1
            elif "random" in lower and "wins" in lower:
                random_win = 1
    return model_win, random_win, tie, invalid


def _prose_tick(label):
    """Replace a stored model id with the name used in the paper.

    Longer ids are matched first so that gpt4 does not consume gpt4o.

    Args:
        label: Legend text.

    Returns:
        str: The same label with model ids written as in the prose.
    """
    text = str(label)
    for model in sorted(REFERENCE_NAMES, key=len, reverse=True):
        name = REFERENCE_NAMES[model]
        if text == model:
            return name
        prefix = model + " "
        if text.startswith(prefix):
            return name + text[len(model):]
    return text


def load_old_outcome_trials(game):
    """Per-game 0/1 outcomes and their standard errors for the 2024 sweep.

    Wins, ties, and invalid moves are checked against the results summary.
    Random-player wins are counted from the stored messages. On two
    Battleship runs the summary does not partition the hundred games,
    because a default loss was closed as unexpected; the messages do
    partition, and those counts are the ones drawn.

    Args:
        game: battleship or connectfour.

    Returns:
        tuple[pandas.DataFrame, dict]: Counts, one row per model and
        temperature, and (model, temperature, metric) -> SE in percentage
        points.
    """
    summary = load_old_counts(game)
    rows = []
    errors = {}
    for model in OLD_MODELS:
        for condition in OLD_CONDITIONS:
            folder = OLD_DIR / f"experiment_{game}_{model}_oneshot_{condition}"
            messages = json.loads((folder / f"game_messages_{game}.json").read_text())
            flags = [_trial_outcome(game_messages) for game_messages in messages]
            if len(flags) != 100 or any(sum(flag[:3]) != 1 for flag in flags):
                raise ValueError(f"{game} {model} {condition} does not give 100 games one outcome each")
            totals = [sum(flag[index] for flag in flags) for index in range(4)]
            temperature = float(condition.split("_")[1])
            stored = summary[(summary["Model"] == model) & (summary["Temperature"] == temperature)].iloc[0]
            if (totals[0], totals[2], totals[3]) != (int(stored["Wins"]), int(stored["Ties"]), int(stored["Incorrect Moves"])):
                raise ValueError(f"{game} {model} {condition} messages {totals} disagree with the results summary")
            partitioned = int(stored["Wins"]) + int(stored["Wins (Random Player)"]) + int(stored["Ties"]) == 100
            if partitioned and totals[1] != int(stored["Wins (Random Player)"]):
                raise ValueError(f"{game} {model} {condition} random wins {totals[1]} disagree with the results summary")
            if not partitioned:
                print(
                    f"{game} {model} t={temperature:g}: random wins {totals[1]} from per-game messages; "
                    f"results summary P2 Wins is {int(stored['Wins (Random Player)'])}"
                )
            counts = {
                "Wins": totals[0],
                "Wins (Random Player)": totals[1],
                "Ties": totals[2],
                "Incorrect Moves": totals[3],
            }
            rows.append({"Model": model, "Temperature": temperature, **counts})
            flag_index = {"Wins": 0, "Wins (Random Player)": 1, "Ties": 2, "Incorrect Moves": 3}
            for metric, index in flag_index.items():
                errors[(model, temperature, metric)] = _binary_se_percent(flag[index] for flag in flags)
    return pd.DataFrame(rows), errors


def fresh_outcome_errors(rows, game, fresh):
    """Standard errors for the 2026 outcome lines.

    The trial sums are checked against the counts the line already draws.

    Args:
        rows: Ledger rows, one per episode.
        game: battleship or connectfour.
        fresh: Counts keyed by condition.

    Returns:
        dict: condition -> metric name -> SE in percentage points.
    """
    grouped = {}
    for row in rows:
        if row["task"] != game:
            continue
        bucket = grouped.setdefault(row["condition"], {metric: [] for metric in METRICS})
        bucket["Wins"].append(int(row["winner"] == "P1"))
        bucket["Wins (Random Player)"].append(int(row["winner"] == "P2"))
        bucket["Ties"].append(int(row["winner"] == "tie"))
        bucket["Incorrect Moves"].append(int(bool(row["invalid"])))
    errors = {}
    for condition, bucket in grouped.items():
        n = len(bucket["Wins"])
        if n != 100:
            raise ValueError(f"{game} {condition} has {n} trials, not 100")
        if sum(bucket["Wins"]) + sum(bucket["Wins (Random Player)"]) + sum(bucket["Ties"]) != n:
            raise ValueError(f"{game} {condition} outcomes do not partition")
        counted = {
            "Wins": fresh[condition]["wins"],
            "Wins (Random Player)": fresh[condition]["random"],
            "Ties": fresh[condition]["ties"],
            "Incorrect Moves": fresh[condition]["incorrect"],
        }
        if any(sum(bucket[metric]) != counted[metric] for metric in METRICS):
            raise ValueError(f"{game} {condition} trial sums disagree with the plotted counts")
        errors[condition] = {metric: _binary_se_percent(bucket[metric]) for metric in METRICS}
    return errors


def fresh_counts(rows, game):
    """Count wins, losses, ties, and rejected replies for one task.

    A rejected reply is the incorrect-move count, matching the 2024
    tally: the game ends on the first move it does not apply.

    Args:
        rows: Ledger rows.
        game: Task name.

    Returns:
        dict: condition -> wins, random wins, ties, incorrect moves.
    """
    grouped = {}
    for row in rows:
        if row["task"] != game:
            continue
        bucket = grouped.setdefault(row["condition"], {"wins": 0, "random": 0, "ties": 0, "incorrect": 0, "n": 0})
        bucket["n"] += 1
        bucket["wins"] += int(row["winner"] == "P1")
        bucket["random"] += int(row["winner"] == "P2")
        bucket["ties"] += int(row["winner"] == "tie")
        bucket["incorrect"] += int(bool(row["invalid"]))
        if row["winner"] not in ("P1", "P2", "tie"):
            raise ValueError(f"unexpected winner {row['winner']}")
    for condition, bucket in grouped.items():
        if bucket["wins"] + bucket["random"] + bucket["ties"] != bucket["n"]:
            raise ValueError(f"{game} {condition} does not partition into wins, losses, and ties")
        expected = EXPECTED.get((game, condition))
        if expected and (bucket["wins"], bucket["random"]) != expected:
            raise ValueError(f"{game} {condition} counts {bucket} disagree with the paper table {expected}")
    return grouped


def plot_outcome(old, errors, destination):
    """Draw the 2024 temperature sweep for the four outcome counts.

    Each point is a count out of 100 games. The bar is the standard error
    of that 0/1 outcome, in percentage points.

    Args:
        old: Counts, one row per model and temperature.
        errors: (model, temperature, metric) -> SE in percentage points.
        destination: PDF path.

    Returns:
        None.
    """
    previous = plt.rcParams.copy()
    plt.rcParams.update({"font.size": 18, "font.weight": "bold"})
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))
    legend_axis = None
    for index, (axis, metric) in enumerate(zip(axes.flatten(), METRICS)):
        sns.lineplot(
            data=old,
            x="Temperature",
            y=metric,
            hue="Model",
            hue_order=OLD_MODELS,
            style="Model",
            style_order=OLD_MODELS,
            markers=True,
            dashes=False,
            markersize=10,
            linewidth=2,
            palette=OLD_COLORS,
            ax=axis,
            legend=index == 0,
            errorbar=None,
        )
        for model in OLD_MODELS:
            part = old[old["Model"] == model].sort_values("Temperature")
            yerr = [errors[(model, float(temperature), metric)] for temperature in part["Temperature"]]
            axis.errorbar(
                part["Temperature"],
                part[metric],
                yerr=yerr,
                fmt="none",
                ecolor=OLD_COLORS[model],
                elinewidth=1.2,
                capsize=4,
                capthick=1.2,
                zorder=3,
            )
        axis.set_title(metric, fontsize=14, fontweight="bold")
        axis.set_xlabel("Temperature")
        axis.set_ylabel(metric)
        axis.set_ylim(0, 100)
        axis.set_xlim(-0.05, 1.55)
        if index == 0:
            legend_axis = axis
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_axis.legend_.remove()
    figure.legend(handles, [_prose_tick(label) for label in labels], title="Model", loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    plt.rcParams.update(previous)


def plot_fresh_outcome(fresh, old, errors, destination, conditions=None):
    """Draw the 2026 outcome counts as a line, with 2024 bests on the axis.

    The line joins Fable, Opus, Astra, Sol, and Luna's three temperatures.
    Each right-axis tick is one 2024 model at the temperature with the most wins.
    The bar is the standard error of that 0/1 outcome, in percentage points.

    Args:
        fresh: Counts keyed by condition.
        old: 2024 counts, one row per model and temperature.
        errors: condition -> metric name -> SE in percentage points.
        destination: PDF path.

    Returns:
        None.
    """
    previous = plt.rcParams.copy()
    plt.rcParams.update({"font.size": 14, "font.weight": "bold"})
    axis_conditions = conditions or FRESH_AXIS
    labels = [label for _condition, label in axis_conditions]
    chosen = _best_win_rows(old) if old is not None else []
    figure, axes = plt.subplots(4, 1, figsize=(11, 12))
    for axis, metric in zip(axes, METRICS):
        values = [fresh[condition][_metric_key(metric)] for condition, _label in axis_conditions]
        yerr = [errors[condition][metric] for condition, _label in axis_conditions]
        if conditions:
            axis.bar(range(len(values)), values, yerr=yerr, color=MODEL_COLOR, capsize=3)
        else:
            axis.errorbar(
            range(len(values)),
            values,
            yerr=yerr,
            color=MODEL_COLOR,
            marker="o",
            markersize=9,
            linewidth=2,
            capsize=4,
            elinewidth=1.2,
            capthick=1.2,
        )
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=11, fontweight="normal")
        axis.set_xlim(-0.35, len(labels) - 0.65)
        axis.set_ylim(0, 100)
        axis.set_title(metric, fontsize=14, fontweight="bold")
        axis.set_ylabel(metric)
        marks = [
            (float(row[metric]), f"{REFERENCE_NAMES[row['Model']]} t={row['Temperature']:g}", OLD_COLORS[row["Model"]])
            for row in chosen
        ]
        _reference_marks(axis, marks)
    figure.tight_layout(rect=(0, 0, 0.72, 1))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    plt.rcParams.update(previous)


def _best_win_rows(old):
    """Pick each 2024 model's temperature with the most wins.

    Ties go to the lower temperature. The other outcome counts on that
    row are the reference marks in the remaining panels.

    Args:
        old: Published counts, one row per model and temperature.

    Returns:
        list[pandas.Series]: One row per model, in the 2024 order.
    """
    chosen = []
    for model in OLD_MODELS:
        part = old[old["Model"] == model].sort_values(["Wins", "Temperature"], ascending=[False, True])
        chosen.append(part.iloc[0])
    return chosen


def _metric_key(metric):
    """Map a panel title to the fresh-count field.

    Args:
        metric: Panel title.

    Returns:
        str: Field name in a fresh count bucket.
    """
    return {
        "Wins": "wins",
        "Wins (Random Player)": "random",
        "Incorrect Moves": "incorrect",
        "Ties": "ties",
    }[metric]


def _count(game, logs):
    """Run the corrected episode-aware tactical counter.

    Args:
        game: tictactoe or connectfour.
        logs: Move dicts in the published schema.

    Returns:
        dict: average_moves, missed_wins, and missed_blocks by player.
    """
    counter = tictactoe_counter if game == "tictactoe" else connectfour_counter
    with contextlib.redirect_stdout(io.StringIO()):
        return counter.count_moves(logs)


def load_old_missed(game):
    """Read tactical rates from retained per-game board observations.

    No missing value is inferred from a plotted stroke. Rates are per
    retained game, and incomplete archives retain their actual sample size.
    """
    records = []
    size = 3 if game == 'tictactoe' else 7
    for model in OLD_MODELS:
        for condition in OLD_CONDITIONS:
            path = OLD_DIR / f'experiment_{game}_{model}_oneshot_{condition}' / f'game_messages_{game}.json'
            messages = json.loads(path.read_text())
            stats = _count(game, episodes_from_messages(messages, size, size))
            n = stats['total_games']
            wins = sum(_trial_outcome(item)[0] for item in messages)
            for player, label in ((0, 'Model'), (1, 'Random')):
                records.append({
                    'Model': model, 'Player': f'{model} {label}',
                    'Temperature': condition, 'Retained Games': n,
                    'Model Win Fraction': wins / n if n else float('nan'),
                    'Average Moves': stats['average_moves'][player],
                    'Missed Wins': stats['missed_wins'][player] / n if n else float('nan'),
                    'Missed Blocks': stats['missed_blocks'][player] / n if n else float('nan'),
                })
    return records


def fresh_logs(game, rows, episodes):
    """Rebuild accepted moves in the schema the missed-move counter reads.

    Args:
        game: tictactoe or connectfour.
        rows: Ledger rows for every condition.
        episodes: Frozen episodes keyed by id.

    Returns:
        dict: condition -> move dicts, in episode order.
    """
    grouped = {}
    for row in rows:
        if row["task"] != game:
            continue
        grouped.setdefault(row["condition"], []).append(row)
    logs = {}
    for condition, played in grouped.items():
        played.sort(key=lambda row: int(row["episode"]))
        condition_episodes = []
        for row in played:
            entries = []
            turn = 0
            model_moves, opponent_moves, winner = replay(game, episodes[int(row["episode"])], row["moves"])
            if winner != row["winner"]:
                raise ValueError(f"{game} episode {row['episode']} replayed as {winner}")
            for index, move in enumerate(model_moves):
                entries.append({"player": 0, "move": _stored_move(game, move), "turn": turn})
                turn += 1
                if index < len(opponent_moves):
                    entries.append({"player": 1, "move": _stored_move(game, opponent_moves[index]), "turn": turn})
                    turn += 1
            condition_episodes.append(entries)
        logs[condition] = condition_episodes
    return logs


def _stored_move(game, move):
    """Convert a replayed move to the published log schema.

    Args:
        game: tictactoe or connectfour.
        move: Column integer, or a row-column pair.

    Returns:
        int | list[int]: The move as the old counter unpacks it.
    """
    if game == "connectfour":
        return int(move)
    return [int(move[0]), int(move[1])]


def fresh_missed(game, logs):
    """Report missed wins and blocks per retained game for the 2026 conditions.

    Args:
        game: tictactoe or connectfour.
        logs: Accepted moves keyed by condition.

    Returns:
        dict: condition -> counter result.
    """
    result = {}
    for condition, entries in logs.items():
        stats = _count(game, entries)
        n = stats['total_games']
        for field in ('missed_wins', 'missed_blocks'):
            stats[field] = {p: count / n if n else float('nan') for p, count in stats[field].items()}
        result[condition] = stats
    return result


def plot_missed(old_records, destination):
    """Draw the 2024 missed-move temperature sweep.

    Args:
        old_records: Published series, one row per player.
        destination: PDF path.

    Returns:
        None.
    """
    frame = pd.DataFrame(old_records)
    temperatures = ["temp_0", "temp_0.5", "temp_1", "temp_1.5"]
    frame["Temperature"] = pd.Categorical(frame["Temperature"], temperatures, ordered=True)
    players = []
    for model in OLD_MODELS:
        players.extend([f"{model} Model", f"{model} Random"])
    figure, axes = plt.subplots(3, 1, figsize=(16, 11))
    legend_axis = None
    for index, (axis, measure) in enumerate(zip(axes, MEASURES)):
        sns.lineplot(
            data=frame,
            x="Temperature",
            y=measure,
            hue="Player",
            hue_order=players,
            style="Model",
            style_order=OLD_MODELS,
            markers=True,
            dashes=False,
            markersize=8,
            linewidth=2,
            palette="deep",
            ax=axis,
            legend=index == 0,
        )
        axis.set_title(measure + " per retained game", fontsize=14, fontweight="bold")
        axis.set_ylabel(measure, fontsize=12, fontweight="bold")
        if index < 2:
            axis.set_xlabel("")
            axis.tick_params(labelbottom=False)
        else:
            axis.set_xlabel("Temperature", fontsize=12, fontweight="bold")
            positions = [float(tick) for tick in axis.get_xticks()]
            if len(positions) != 4:
                raise ValueError(f"expected four temperature ticks, found {positions}")
            axis.set_xticks(positions)
            axis.set_xticklabels(["0", "0.5", "1", "1.5"], fontsize=12, fontweight="bold")
        if index == 0:
            legend_axis = axis
    handles, labels = legend_axis.get_legend_handles_labels()
    legend_axis.legend_.remove()
    figure.legend(handles, [_prose_tick(label) for label in labels], title="Player and Model", loc="center left", bbox_to_anchor=(1.0, 0.55), frameon=True, fontsize=9)
    figure.tight_layout(rect=(0, 0, 0.78, 1))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def plot_fresh_missed(fresh, old_records, destination, conditions=None):
    """Draw the 2026 missed-move counts as lines, with 2024 bests on the axis.

    Two lines are the model and the random opponent. Each right-axis tick
    is a 2024 model at the highest win fraction in a complete archive, and the
    other panels use that same temperature.

    Args:
        fresh: Counter results keyed by condition.
        old_records: Published 2024 series, one row per player.
        destination: PDF path.

    Returns:
        None.
    """
    previous = plt.rcParams.copy()
    plt.rcParams.update({"font.size": 13, "font.weight": "bold"})
    axis_conditions = conditions or FRESH_AXIS
    labels = [label for _condition, label in axis_conditions]
    chosen = _best_missed_rows(old_records) if old_records else []
    figure, axes = plt.subplots(3, 1, figsize=(13, 10))
    for axis, measure in zip(axes, MEASURES):
        model_values = [_player_value(fresh[condition], measure, 0) for condition, _label in axis_conditions]
        random_values = [_player_value(fresh[condition], measure, 1) for condition, _label in axis_conditions]
        if conditions:
            positions = list(range(len(labels)))
            axis.bar([x - 0.2 for x in positions], model_values, width=0.4, color=MODEL_COLOR, label="Model")
            axis.bar([x + 0.2 for x in positions], random_values, width=0.4, color=RANDOM_COLOR, label="Random opponent")
        else:
            axis.plot(range(len(labels)), model_values, color=MODEL_COLOR, marker="o", markersize=8, linewidth=2, label="Model")
            axis.plot(range(len(labels)), random_values, color=RANDOM_COLOR, marker="o", markersize=8, linewidth=2, label="Random opponent")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, fontsize=11, fontweight="normal")
        axis.set_xlim(-0.35, len(labels) - 0.65)
        ceiling = max(model_values + random_values + [row[measure] for row in chosen] + [1])
        axis.set_ylim(0, ceiling * 1.08)
        axis.set_title(measure + " per retained game", fontsize=14, fontweight="bold")
        axis.set_ylabel(measure, fontsize=12, fontweight="bold")
        marks = [
            (float(row[measure]), f"{REFERENCE_NAMES[row['Model']]} t={_temp_text(row['Temperature'])}", OLD_COLORS[row["Model"]])
            for row in chosen
        ]
        _reference_marks(axis, marks)
    handles = [
        Line2D([0], [0], color=MODEL_COLOR, marker="o", linewidth=2, label="Model"),
        Line2D([0], [0], color=RANDOM_COLOR, marker="o", linewidth=2, label="Random opponent"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.39, 1.02), ncol=2, frameon=True)
    figure.tight_layout(rect=(0, 0, 0.78, 0.96))
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)
    plt.rcParams.update(previous)


def _best_missed_rows(records):
    """Pick the highest win fraction among complete 100-game archives.

    The comparison is the model player. Ties go to the lower temperature.

    Args:
        records: One row per model, temperature, and player.

    Returns:
        list[dict]: One model-player row per model with a complete archive.
    """
    order = {condition: index for index, condition in enumerate(OLD_CONDITIONS)}
    chosen = []
    for model in OLD_MODELS:
        rows = [row for row in records if row["Model"] == model and row["Player"].endswith("Model") and row["Retained Games"] == 100]
        rows.sort(key=lambda row: (-row["Model Win Fraction"], order[row["Temperature"]]))
        if rows:
            chosen.append(rows[0])
    return chosen


def _temp_text(temperature):
    """Format a stored temperature for an axis label.

    Args:
        temperature: A float, or a condition such as temp_0.5.

    Returns:
        str: The temperature without a trailing zero.
    """
    if isinstance(temperature, str):
        temperature = float(temperature.split("_")[-1])
    return f"{temperature:g}"


def _reference_marks(axis, marks):
    """Put reference values on the right axis without drawing them as lines.

    Marks that share a value use one tick. Labels that would overlap are
    shifted vertically and joined to that tick.

    Args:
        axis: Matplotlib axis.
        marks: Tuples of y value, label, and color.

    Returns:
        None.
    """
    grouped = {}
    for value, label, color in marks:
        grouped.setdefault(round(float(value), 4), []).append((label, color))
    entries = []
    for value, texts in sorted(grouped.items()):
        for index, (label, color) in enumerate(texts):
            entries.append((value, label, color, index == 0))
    low, high = axis.get_ylim()
    gap = (high - low) * 0.12
    slots = []
    cursor = low
    for value, _label, _color, _tick in entries:
        slot = max(value, cursor)
        slots.append(slot)
        cursor = slot + gap
    if slots and slots[-1] > high:
        slots = [slot - (slots[-1] - high) for slot in slots]
    if slots and slots[0] < low:
        slots = [slot + (low - slots[0]) for slot in slots]
    transform = blended_transform_factory(axis.transAxes, axis.transData)
    for (value, label, color, tick), slot in zip(entries, slots):
        if tick:
            axis.plot([1.0, 1.015], [value, value], transform=transform, color="0.25", linewidth=1.4, clip_on=False, solid_capstyle="butt")
        if abs(slot - value) > gap * 0.3:
            axis.plot([1.015, 1.04], [value, slot], transform=transform, color="0.55", linewidth=0.6, clip_on=False)
        axis.text(1.055, slot, label, transform=transform, va="center", ha="left", fontsize=9, color=color, fontweight="normal", clip_on=False)


def _missed_key(measure):
    """Map a panel title to the counter field.

    Args:
        measure: Panel title.

    Returns:
        str: Field name.
    """
    return {"Average Moves": "average_moves", "Missed Wins": "missed_wins", "Missed Blocks": "missed_blocks"}[measure]


def _player_value(stats, measure, player):
    """Read one player's value from a counter result.

    Args:
        stats: Result dict from count_moves.
        measure: Panel title.
        player: 0 for the model and 1 for the random opponent.

    Returns:
        float: The plotted value.
    """
    return stats[_missed_key(measure)][player]


def main():
    """Write the appendix outcome figures.

    The temperature PDFs contain the 2024 sweep. The 2026 PDFs are line charts.

    Returns:
        None.
    """
    rows = load_rows()
    stimuli = load_stimuli()
    episodes = {
        game: {item["id"]: item for item in stimuli[game]}
        for game in ("tictactoe", "connectfour")
    }
    fresh_names = {
        "battleship": "battleship_2026_outcomes.pdf",
        "connectfour": "connectfour_2026_outcomes.pdf",
    }
    for game, filename in (
        ("battleship", "battleship_correct_incorrect_lines.pdf"),
        ("connectfour", "connectfour_correct_incorrect_lines.pdf"),
    ):
        fresh = fresh_counts(rows, game)
        old, errors = load_old_outcome_trials(game)
        fresh_errors = fresh_outcome_errors(rows, game, fresh)
        plot_outcome(old, errors, OUT_DIR / filename)
        plot_fresh_outcome(fresh, old, fresh_errors, OUT_DIR / fresh_names[game])
        print(game, {key: fresh[key] for key in fresh})
    missed_names = {
        "tictactoe": "tictactoe_2026_missed.pdf",
        "connectfour": "connectfour_2026_missed.pdf",
    }
    for game, filename in (
        ("tictactoe", "tictactoe_missed_wins_blocks_lines.pdf"),
        ("connectfour", "connect_four_missed_wins_blocks_lines.pdf"),
    ):
        fresh = fresh_missed(game, fresh_logs(game, rows, episodes[game]))
        old_records = load_old_missed(game)
        plot_missed(old_records, OUT_DIR / filename)
        plot_fresh_missed(fresh, old_records, OUT_DIR / missed_names[game])
        printable = {
            condition: {
                "moves": fresh[condition]["average_moves"],
                "missed_wins": fresh[condition]["missed_wins"],
                "missed_blocks": fresh[condition]["missed_blocks"],
            }
            for condition in fresh
        }
        print(game, json.dumps(printable))


if __name__ == "__main__":
    main()
