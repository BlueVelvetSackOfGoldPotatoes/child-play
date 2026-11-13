import io
import argparse
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fresh_heatmaps import load_rows

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Configuration ----------------------
MODEL_ORDER = ['gpt3_5', 'gpt4', 'gpt4o_mini', 'gpt4o']
FRESH_ORDER = ['fable', 'opus', 'astra', 'sol', 'luna']
MODEL_LABELS = {
    'gpt3_5': 'GPT-3.5',
    'gpt4': 'GPT-4',
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt4o': 'GPT-4o',
    'fable': 'Fable',
    'opus': 'Opus',
    'astra': 'Astra',
    'sol': 'Sol',
    'luna': 'Luna',
    # Possibly other name mappings:
    'oa_gpt-3.5-turbo-1106': 'GPT-3.5 Turbo',
    'oa_gpt-4-1106-preview': 'GPT-4 Preview',
    'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini',
    'oa:gpt-4o-2024-08-06': 'gpt4o'
}

TEMPERATURES = [0, 0.5, 1, 1.5]

BASE_PATH_SHAPES        = '../experiment_shapes'
BASE_PATH_LCL           = '../lcl_experiments'
BASE_PATH_BOARDGAMES    = '../experiment_board_games'
BASE_PATH_MOLECULE_APP  = '../molecule_app'

# Define custom palette using your provided hex codes.
FIGURE_DIR = Path(__file__).resolve().parents[1] / 'paper' / 'complete'

custom_palette = ["#DD1C1A", "#E6A800", "#086788", "#06AED5", "#FFF1D0"]

# ---------------------------------------------------------------------
# 1) Shapes
# ---------------------------------------------------------------------

def load_shapes_metrics():
    """
    For each model, find the single best temperature (overall),
    storing:
      - 'Correct Proportion': best ratio across square/triangle/cross
      - 'Best Temp': temperature that achieved that best ratio
    We read wins as d.get('P1 Wins', d.get('Wins', 0)) and
          losses as d.get('Losses', d.get('P2 Wins', 0)).
    """
    data = {m: {} for m in MODEL_ORDER}
    shape_folders = ['square', 'triangle', 'cross']
    
    for rm in MODEL_ORDER:
        best_ratio = 0.0
        best_temp = None
        
        for t in TEMPERATURES:
            total_wins = 0
            total_losses = 0
            for shape in shape_folders:
                path = os.path.join(
                    BASE_PATH_SHAPES,
                    rm.replace(":", "_"),
                    str(t).replace(".", "_"),
                    shape,
                    'results.json'
                )
                if not os.path.exists(path):
                    continue
                try:
                    with open(path, 'r') as f:
                        d = json.load(f)
                        wins = d.get('P1 Wins', d.get('Wins', 0))
                        losses = d.get('Losses', d.get('P2 Wins', 0))
                        total_wins += wins
                        total_losses += losses
                except:
                    pass
            
            attempts = total_wins + total_losses
            if attempts > 0:
                ratio = total_wins / attempts
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_temp = t
        
        data[rm]['Correct Proportion'] = best_ratio
        data[rm]['Best Temp'] = best_temp
    
    return data

def load_shapes_breakdown_by_best_temp(shapes_data):
    """
    For each model, uses that model's single best temperature.
    Returns shape-specific fractions that sum up to the model's
    overall 'Correct Proportion'.

    data_out[model][shape] = shape_wins / total_attempts (at best temp).
    """
    shape_folders = ['square', 'triangle', 'cross']
    data_out = {m: {} for m in MODEL_ORDER}
    
    for m in MODEL_ORDER:
        best_temp = shapes_data[m].get('Best Temp', None)
        if best_temp is None:
            # No data found for that model
            for shape in shape_folders:
                data_out[m][shape] = 0.0
            continue
        
        total_wins_all_shapes = 0
        total_attempts_all_shapes = 0
        shape_wins = {}
        
        # Count total wins/losses across all shapes at best_temp
        for shape in shape_folders:
            path = os.path.join(
                BASE_PATH_SHAPES,
                m.replace(":", "_"),
                str(best_temp).replace(".", "_"),
                shape,
                'results.json'
            )
            if not os.path.exists(path):
                shape_wins[shape] = 0
                continue
            
            try:
                with open(path, 'r') as f:
                    d = json.load(f)
                    wins = d.get('P1 Wins', d.get('Wins', 0))
                    losses = d.get('Losses', d.get('P2 Wins', 0))
                    
                    shape_wins[shape] = wins
                    total_wins_all_shapes += wins
                    total_attempts_all_shapes += (wins + losses)
            except:
                shape_wins[shape] = 0
        
        # Convert shape wins -> fraction of total attempts
        for shape in shape_folders:
            if total_attempts_all_shapes > 0:
                data_out[m][shape] = shape_wins[shape] / total_attempts_all_shapes
            else:
                data_out[m][shape] = 0.0
    
    return data_out

# ---------------------------------------------------------------------
# 2) LCL
# ---------------------------------------------------------------------
def _best_rate_by_temperature(frame, model_column, value_column, model_map):
    """Return each model's highest temperature-wise mean of a boolean column.

    Ties keep the lower temperature because the temperatures are scanned
    in ascending order and a new rate must be strictly greater.

    Args:
        frame: Table with a model column, a Temperature column, and the value.
        model_column: Column of model ids before they are mapped.
        value_column: Column of booleans, true when the trial is a success.
        model_map: Map from the ids stored in the table to MODEL_ORDER ids.

    Returns:
        dict: Model id to the best rate, as a fraction from 0 to 1.
    """
    if frame.empty or model_column not in frame.columns:
        return {}
    mapped = frame.copy()
    mapped[model_column] = mapped[model_column].map(model_map).fillna(mapped[model_column])
    best = {}
    for model, group in mapped.groupby(model_column):
        if model not in MODEL_ORDER:
            continue
        chosen = -1.0
        for _, temp_group in sorted(group.groupby('Temperature'), key=lambda item: float(item[0])):
            rate = temp_group[value_column].astype(str).isin(['True', 'true', '1']).mean()
            if rate > chosen:
                chosen = float(rate)
        best[model] = chosen
    return best


def load_lcl_metrics():
    """Load the best-temperature LCL validity and construct rates.

    Correct Proportion is the validity task. Valid Proportion is construct
    generation. Each metric keeps its own best temperature.

    Returns:
        dict: Model id to those two fractions.
    """
    data = {m: {} for m in MODEL_ORDER}
    model_map_lcl = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-3.5-turbo-1106': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }
    folder = os.path.join(BASE_PATH_LCL, 'data_lcl_experiments')
    try:
        df_valid = pd.read_csv(os.path.join(folder, 'df_validity.csv'))
    except Exception as e:
        print(f"Warning: Can't load df_validity.csv. Error: {e}")
        df_valid = pd.DataFrame(columns=['Model', 'Correct'])
    try:
        df_construct = pd.read_csv(os.path.join(folder, 'df_construct.csv'))
    except Exception as e:
        print(f"Warning: Can't load df_construct.csv. Error: {e}")
        df_construct = pd.DataFrame(columns=['Model', 'Valid'])
    correct = _best_rate_by_temperature(df_valid, 'Model', 'Correct', model_map_lcl)
    valid = _best_rate_by_temperature(df_construct, 'Model', 'Valid', model_map_lcl)
    for model in MODEL_ORDER:
        data[model]['Correct Proportion'] = correct.get(model, 0.0)
        data[model]['Valid Proportion'] = valid.get(model, 0.0)
    return data

# ---------------------------------------------------------------------
# 3) Board Games
# ---------------------------------------------------------------------
def load_boardgame_metrics(game):
    """
    For each model and for a given board game, picks the best temperature based on
    maximum 'P1 Wins'. Returns:
      - 'Correct Proportion' = best_wins / 100
      - 'Tie Proportion' = ties / 100
      - 'Wrong Moves Proportion' = wrong_moves / 100
    """
    data = {m: {} for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        best_wins = -1
        best_temp = None
        for t in TEMPERATURES:
            folder = f"experiment_{game}_{m}_oneshot_temp_{t}"
            path = os.path.join(BASE_PATH_BOARDGAMES, folder, f"results_{game}.json")
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r') as f:
                    d = json.load(f)
                    llm_wins = d.get('P1 Wins', 0)
                    if llm_wins > best_wins:
                        best_wins = llm_wins
                        best_temp = t
            except:
                pass
        
        if best_wins < 0:
            best_wins = 0
        ties = 0
        wrong_moves = 0
        if best_temp is not None:
            folder = f"experiment_{game}_{m}_oneshot_temp_{best_temp}"
            path = os.path.join(BASE_PATH_BOARDGAMES, folder, f"results_{game}.json")
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        d = json.load(f)
                        ties = d.get('Ties', 0)
                        wrong_moves = d.get('P1 Wrong Moves', 0)
                except:
                    pass
        
        # Convert from raw counts (out of 100 games) to proportion
        data[m]['Correct Proportion'] = best_wins / 100.0
        data[m]['Tie Proportion'] = ties / 100.0
        data[m]['Wrong Moves Proportion'] = wrong_moves / 100.0
    
    return data

# ---------------------------------------------------------------------
# 4) Molecule App
# ---------------------------------------------------------------------
def load_molecule_metrics():
    """
    For each model, reads the CSVs from the Molecule App and calculates:
      - 'Accuracy' (correct / total)
      - 'Avg Chem. Similarity'
      - 'Avg String Distance'
      - 'Incorrect SMILES Fraction'
    """
    data = {m: {} for m in MODEL_ORDER}
    model_map_mol = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }
    correct_total = {m: 0 for m in MODEL_ORDER}
    chem_similarity_total = {m: 0.0 for m in MODEL_ORDER}
    string_distance_total = {m: 0.0 for m in MODEL_ORDER}
    incorrect_trials_total = {m: 0 for m in MODEL_ORDER}
    count_total = {m: 0 for m in MODEL_ORDER}
    
    if not os.path.isdir(BASE_PATH_MOLECULE_APP):
        print("Warning: Molecule app folder not found.")
        return data
    
    files = [f for f in os.listdir(BASE_PATH_MOLECULE_APP)
             if f.startswith('benchmark_resultsoa:gpt') and f.endswith('.csv')]
    
    for fname in files:
        path = os.path.join(BASE_PATH_MOLECULE_APP, fname)
        try:
            df = pd.read_csv(path)
            if 'model' in df.columns:
                df['model'] = df['model'].map(model_map_mol).fillna(df['model'])
            for _, row in df.iterrows():
                rm = row.get('model', '')
                if rm not in MODEL_ORDER:
                    continue
                if row.get('correct', False):
                    correct_total[rm] += 1
                chem_similarity_total[rm] += row.get('chemical_similarity', 0.0)
                string_distance_total[rm] += row.get('string_distance', 0.0)
                if row.get('incorrect_smiles_count', 0) > 0:
                    incorrect_trials_total[rm] += 1
                count_total[rm] += 1
        except:
            pass
    
    for mm in MODEL_ORDER:
        c_total = count_total[mm]
        if c_total > 0:
            accuracy = correct_total[mm] / c_total
            avg_chem_similarity = chem_similarity_total[mm] / c_total
            avg_string_distance = string_distance_total[mm] / c_total
            incorrect_smiles_fraction = incorrect_trials_total[mm] / c_total
            data[mm]['Accuracy'] = accuracy
            data[mm]['Avg Chem. Similarity'] = avg_chem_similarity
            data[mm]['Avg String Distance'] = avg_string_distance
            data[mm]['Incorrect SMILES Fraction'] = incorrect_smiles_fraction
    
    return data


def _gts_point(frame):
    """Reduce one model-temperature block to the three plotted GtS metrics.

    Args:
        frame: Rows for one model at one temperature. Requires correct and
            chemical_similarity columns.

    Returns:
        dict: Accuracy, Avg Chem. Similarity, and Incorrect SMILES Fraction.
        Accuracy is correct replies over every reply.
    """
    count = len(frame)
    similarity = frame["chemical_similarity"].astype(float)
    return {
        "Accuracy": float(frame["correct"].astype(bool).sum()) / count,
        "Avg Chem. Similarity": float(similarity.mean()),
        "Incorrect SMILES Fraction": float((similarity == -1).mean()),
    }


def _paper_accuracy(frame):
    """Return the table accuracy, which leaves invalid SMILES out.

    Args:
        frame: Rows with correct and chemical_similarity columns.

    Returns:
        float: Correct / (correct + incorrect), or 0 when every reply is
        invalid. This is the fraction behind the Accuracy column, not a percent.
    """
    similarity = frame["chemical_similarity"].astype(float)
    graded = frame.loc[similarity != -1]
    if graded.empty:
        return 0.0
    return float(graded["correct"].astype(bool).sum()) / len(graded)


def load_gts_metrics():
    """Best-condition Guess-the-SMILES metrics for every model in the figure.

    A 2024 model keeps the lowest temperature with the most correct replies,
    which is the point already drawn for that model. A fresh model does the
    same across its calls. Duplicate rows in the published CSV files are
    dropped before the choice.

    Returns:
        dict: Model id to Accuracy, Avg Chem. Similarity, Incorrect SMILES
        Fraction, and Paper Accuracy. Paper Accuracy is the table definition
        at the temperature with the highest such accuracy.
    """
    published = _load_published_gts_frames()
    fresh = _load_fresh_gts_frames()
    data = {}
    for model in MODEL_ORDER:
        data[model] = _best_gts_condition(published[published["model"] == model])
    for model in FRESH_ORDER:
        data[model] = _best_gts_condition(fresh[fresh["model"] == model])
    return data


def _load_published_gts_frames():
    """Read the 2024 Guess-the-SMILES logs once.

    NUL bytes in those exports are stripped. Repeated copies of the same
    trial, which later files carry forward, are removed.

    Returns:
        DataFrame: One row per model, temperature, and trial.
    """
    frames = []
    folder = Path(__file__).resolve().parents[1] / "molecule_app"
    model_map = {
        "oa:gpt-3.5-turbo-0125": "gpt3_5",
        "oa:gpt-4-1106-preview": "gpt4",
        "oa:gpt-4o-2024-08-06": "gpt4o",
        "oa:gpt-4o-mini-2024-07-18": "gpt4o_mini",
    }
    for path in sorted(folder.glob("benchmark_resultsoa:gpt*.csv")):
        frame = pd.read_csv(io.BytesIO(path.read_bytes().replace(b"\x00", b"")))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["model"] = combined["model"].map(model_map)
    combined = combined.dropna(subset=["model"])
    combined["temperature"] = pd.to_numeric(combined["temperature"])
    combined["correct"] = combined["correct"].astype(str).str.lower().isin(["true", "1"])
    combined["chemical_similarity"] = pd.to_numeric(combined["chemical_similarity"])
    return combined.drop_duplicates(["model", "temperature", "trial"])


def _load_fresh_gts_frames():
    """Read the fresh Guess-the-SMILES ledger.

    Returns:
        DataFrame: One row per finished call. Empty when the run has not started.
    """
    path = Path(__file__).resolve().parents[1] / "fresh_experiments" / "seed20261111" / "gts_results.jsonl"
    rows = []
    order = {
        "claude-fable-5-2": ("fable", 0.0),
        "claude-opus-5-5": ("opus", 0.0),
        "gpt-6-astra": ("astra", 0.0),
        "gpt-6.1-sol": ("sol", 0.0),
        "gpt-6-luna-t0": ("luna", 0.0),
        "gpt-6-luna-t0.5": ("luna", 0.5),
        "gpt-6-luna-t1": ("luna", 1.0),
    }
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            model, temperature = order[row["condition"]]
            rows.append({
                "model": model,
                "temperature": temperature,
                "correct": bool(row["correct"]),
                "chemical_similarity": float(row["chemical_similarity"]),
            })
    return pd.DataFrame(rows, columns=["model", "temperature", "correct", "chemical_similarity"])


def _best_gts_condition(frame):
    """Choose one temperature and return its plotted metrics.

    The plotted temperature is the lowest one with the most correct replies.
    Paper Accuracy is chosen separately, at the temperature with the highest
    accuracy after invalid replies are removed. Those two temperatures differ
    for GPT-3.5 and GPT-4o in the published table.

    Args:
        frame: Rows for one model, possibly at several temperatures.

    Returns:
        dict: The three plotted metrics plus Paper Accuracy.
    """
    if frame.empty:
        return {
            "Accuracy": float("nan"),
            "Avg Chem. Similarity": float("nan"),
            "Incorrect SMILES Fraction": float("nan"),
            "Paper Accuracy": float("nan"),
        }
    chosen = None
    paper = -1.0
    for temperature, rows in sorted(frame.groupby("temperature"), key=lambda item: item[0]):
        point = _gts_point(rows)
        correct = point["Accuracy"] * len(rows)
        if chosen is None or correct > chosen[0]:
            chosen = (correct, point)
        paper = max(paper, _paper_accuracy(rows))
    chosen[1]["Paper Accuracy"] = paper
    return chosen[1]


def load_fresh_metrics():
    """Read the fresh ledger and reduce it the way the older loaders do.

    Fable, Opus, Astra, and Sol have one condition. Luna has three
    temperatures, and each task keeps the temperature with the most
    successes. Guess-the-SMILES was not rerun, so it is absent.
    Battleship rows come from the clause rerun through load_rows. Every
    other task stays on the published seed20261111 ledger.

    Returns:
        dict: Task name to a model-id dict of the same metric names the
        older loaders store. Shapes also includes a 'breakdown' entry.
    """
    conditions = {
        'fable': ['claude-fable-5-2'],
        'opus': ['claude-opus-5-5'],
        'astra': ['gpt-6-astra'],
        'sol': ['gpt-6.1-sol'],
        'luna': ['gpt-6-luna-t0', 'gpt-6-luna-t0.5', 'gpt-6-luna-t1'],
    }
    rows = {model: [] for model in FRESH_ORDER}
    for record in load_rows():
        for model, names in conditions.items():
            if record['condition'] in names:
                rows[model].append(record)
    board = {game: {} for game in ('tictactoe', 'connectfour', 'battleship')}
    shapes = {}
    breakdown = {}
    lcl = {}
    for model in FRESH_ORDER:
        for game, dest in board.items():
            best = None
            for condition in conditions[model]:
                played = [row for row in rows[model] if row['task'] == game and row['condition'] == condition]
                if not played:
                    continue
                wins = sum(row['winner'] == 'P1' for row in played)
                if best is None or wins > best[0]:
                    best = (
                        wins,
                        sum(row['winner'] == 'tie' for row in played),
                        sum(bool(row['invalid']) for row in played),
                        len(played),
                    )
            wins, ties, wrong, count = best
            dest[model] = {
                'Correct Proportion': wins / count,
                'Tie Proportion': ties / count,
                'Wrong Moves Proportion': wrong / count,
            }
        best_shapes = None
        for condition in conditions[model]:
            played = [row for row in rows[model] if row['task'] == 'shapes' and row['condition'] == condition]
            if not played:
                continue
            correct = sum(bool(row['correct']) for row in played)
            if best_shapes is None or correct > best_shapes[0]:
                best_shapes = (correct, played)
        correct, played = best_shapes
        shapes[model] = {'Correct Proportion': correct / len(played), 'Best Temp': None}
        counts = {name: [0, 0] for name in ('square', 'triangle', 'cross')}
        for row in played:
            kind = str(row['episode']).split('-')[0]
            counts[kind][1] += 1
            counts[kind][0] += int(bool(row['correct']))
        total = len(played)
        breakdown[model] = {name: counts[name][0] / total for name in counts}
        validity = _fresh_best(rows[model], conditions[model], 'lcl_validity')
        construct = _fresh_best(rows[model], conditions[model], 'lcl_construct')
        lcl[model] = {'Correct Proportion': validity, 'Valid Proportion': construct}
    return {
        'Shapes': shapes,
        'shapes_breakdown': breakdown,
        'LCL': lcl,
        'Tic-Tac-Toe': board['tictactoe'],
        'Connect-Four': board['connectfour'],
        'Battleship': board['battleship'],
    }


def _fresh_best(records, condition_names, task):
    """Return the highest success rate of one fresh task across conditions.

    Args:
        records: Ledger rows for one fresh model.
        condition_names: Condition ids, scanned from lowest temperature.
        task: Task name in the ledger.

    Returns:
        float: Best fraction correct, from 0 to 1.
    """
    chosen = -1.0
    for condition in condition_names:
        played = [row for row in records if row['task'] == task and row['condition'] == condition]
        if not played:
            continue
        rate = sum(bool(row['correct']) for row in played) / len(played)
        if rate > chosen:
            chosen = rate
    return chosen


# ---------------------------------------------------------------------
# Main Aggregation & Plotting
# ---------------------------------------------------------------------
def _save(figure_name):
    """Write one paper figure next to main_cell.tex and close it.

    Args:
        figure_name: PDF file name, without a directory.

    Returns:
        None.
    """
    destination = FIGURE_DIR / figure_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(destination, bbox_inches="tight")
    plt.close()


def main(include=None, historical=False, report=None):
    """Write the model-comparison figures.

    Args:
        include: Figure names to write, or None for every figure. Names
            are the task keys plus ``gts`` and ``stack``.

    Returns:
        None.
    """
    selected = None if include is None else set(include)
    # -------------------------
    # Load all the data
    # -------------------------
    global FIGURE_DIR
    if historical:
        FIGURE_DIR = Path(__file__).resolve().parents[1] / 'paper'
        shapes_data = load_shapes_metrics()                # picks ONE best temp; stores overall 'Correct Proportion'
        shapes_breakdown = load_shapes_breakdown_by_best_temp(shapes_data)
        
        lcl_data = load_lcl_metrics()
        tictactoe_data = load_boardgame_metrics('tictactoe')
        connectfour_data = load_boardgame_metrics('connectfour')
        battleship_data = load_boardgame_metrics('battleship')
        fresh = load_fresh_metrics()
        for model in FRESH_ORDER:
            shapes_data[model] = fresh['Shapes'][model]
            shapes_breakdown[model] = fresh['shapes_breakdown'][model]
            lcl_data[model] = fresh['LCL'][model]
            tictactoe_data[model] = fresh['Tic-Tac-Toe'][model]
            connectfour_data[model] = fresh['Connect-Four'][model]
            battleship_data[model] = fresh['Battleship'][model]
        gts_data = load_gts_metrics()
        plot_order = MODEL_ORDER + FRESH_ORDER
    else:
        from utils.paper_dataset import analyze
        report = report or analyze()
        FIGURE_DIR = Path(__file__).resolve().parents[1] / 'paper' / 'complete'
        plot_order = list(report['metrics'])
        shapes_data, shapes_breakdown, lcl_data = {}, {}, {}
        tictactoe_data, connectfour_data, battleship_data, gts_data = {}, {}, {}, {}
        for condition, block in report['metrics'].items():
            MODEL_LABELS[condition] = block['label']
            tasks = block['tasks']
            shapes_data[condition] = {'Correct Proportion': tasks['shapes']['score']}
            shapes_breakdown[condition] = {shape: rate / 3 for shape, rate in tasks['shapes']['by_shape'].items()}
            lcl_data[condition] = {'Validity': tasks['lcl_validity']['score'], 'Generation': tasks['lcl_construct']['score']}
            for task, destination in [('tictactoe', tictactoe_data), ('connectfour', connectfour_data), ('battleship', battleship_data)]:
                row = tasks[task]
                destination[condition] = {'Win': row['score'], 'Tie': row['ties'] / row['n'], 'Illegal reply': row['invalid'] / row['n']}
            row = tasks['gts']
            gts_data[condition] = {'Accuracy': row['score'], 'Avg Chem. Similarity': row['penalized_similarity'], 'Incorrect SMILES Fraction': row['invalid'] / row['n'], 'Paper Accuracy': row['score']}

    all_data = {
        'Shapes': shapes_data,
        'LCL': lcl_data,
        'Tic-Tac-Toe': tictactoe_data,
        'Connect-Four': connectfour_data,
        'Battleship': battleship_data,
    }
    figure_names = {
        'Shapes': 'model_comparison_shapes.pdf',
        'LCL': 'model_comparison_lcl.pdf',
        'Tic-Tac-Toe': 'model_comparison_tictactoe.pdf',
        'Connect-Four': 'model_comparison_connectfour.pdf',
        'Battleship': 'model_comparison_battleship.pdf',
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

    for game_name, game_dict in all_data.items():
        if selected is not None and game_name not in selected:
            continue
        if game_name == "Shapes":
            plt.figure(figsize=(12, 6))
            plt.title(("Correct Proportion per Model for the Shapes Game\n(Stacked by Shape Contribution at Best Temp)" if historical else "Shape recognition: accuracy by condition and class"),
                      fontsize=18, fontweight='bold')
            x = np.arange(len(plot_order))
            square_vals = np.array([shapes_breakdown[m].get('square', 0) for m in plot_order])
            triangle_vals = np.array([shapes_breakdown[m].get('triangle', 0) for m in plot_order])
            cross_vals = np.array([shapes_breakdown[m].get('cross', 0) for m in plot_order])
            if historical:
                plt.stackplot(
                    x,
                    square_vals,
                    triangle_vals,
                    cross_vals,
                    labels=["Square", "Triangle", "Cross"],
                    colors=[custom_palette[i] for i in range(3)]
                )
            else:
                bottom = np.zeros(len(plot_order))
                for shape, values, color in zip(['Square', 'Triangle', 'Cross'], [square_vals, triangle_vals, cross_vals], custom_palette):
                    plt.bar(x, values, bottom=bottom, label=shape, color=color)
                    bottom += values
            plt.xticks(x, [MODEL_LABELS[m] for m in plot_order], rotation=30, ha='right')
            plt.xlabel("Model", fontsize=14, fontweight='bold')
            plt.ylabel("Correct Proportion (stacked)", fontsize=14, fontweight='bold')
            plt.legend(loc='upper right')
            _save(figure_names[game_name])
        else:
            plt.figure(figsize=(12, 6))
            plt.title((f'Best Temperature per Model for {game_name}' if historical else f'{game_name}: observed outcomes by condition'), fontsize=18, fontweight='bold')
            x_labels = [MODEL_LABELS[m] for m in plot_order]
            metric_keys = sorted(game_dict[plot_order[0]].keys())
            color_index = 0
            all_y = []
            for metric_name in metric_keys:
                if metric_name == "Avg String Distance":
                    continue
                y_vals = [game_dict[m].get(metric_name, np.nan) for m in plot_order]
                all_y.extend([v for v in y_vals if not np.isnan(v)])
                if historical:
                    plt.plot(
                        x_labels,
                        y_vals,
                        marker='o',
                        label=metric_name,
                        color=custom_palette[color_index % len(custom_palette)]
                    )
                else:
                    width = 0.8 / len(metric_keys)
                    offset = (color_index - (len(metric_keys) - 1) / 2) * width
                    plt.bar(np.arange(len(plot_order)) + offset, y_vals, width=width, label=metric_name, color=custom_palette[color_index % len(custom_palette)])
                    plt.xticks(np.arange(len(plot_order)), x_labels)
                color_index += 1
            if all_y:
                global_min = min(all_y)
                global_max = max(all_y)
                margin = (global_max - global_min) * 0.1 if (global_max - global_min) != 0 else 0.1
                plt.ylim(global_min - margin, global_max + margin)
            else:
                plt.ylim(0, 1)
            plt.xticks(rotation=30, ha='right')
            plt.xlabel("Model", fontsize=14, fontweight='bold')
            plt.ylabel("Proportion / Fraction", fontsize=14, fontweight='bold')
            plt.legend(title="Metric", loc='upper right')
            _save(figure_names[game_name])

    if selected is None or "gts" in selected:
        plt.figure(figsize=(12, 6))
        plt.title("GtS: recorded responses by condition", fontsize=18, fontweight='bold')
        gts_metrics = ["Accuracy", "Avg Chem. Similarity", "Incorrect SMILES Fraction"]
        for index, metric_name in enumerate(gts_metrics):
            y_vals = [gts_data[model].get(metric_name, np.nan) for model in plot_order]
            if historical:
                plt.plot(
                    [MODEL_LABELS[model] for model in plot_order],
                    y_vals,
                    marker='o',
                    label=metric_name,
                    color=custom_palette[index % len(custom_palette)],
                )
            else:
                names = ['All-reply accuracy', 'Penalized fingerprint similarity', 'Invalid reply fraction']
                plt.bar(np.arange(len(plot_order)) + (index - 1) * 0.25, y_vals, width=0.25, label=names[index], color=custom_palette[index])
                plt.xticks(np.arange(len(plot_order)), [MODEL_LABELS[m] for m in plot_order])
        plt.xticks(rotation=30, ha='right')
        plt.xlabel("Model", fontsize=14, fontweight='bold')
        plt.ylabel("Proportion / Fraction", fontsize=14, fontweight='bold')
        plt.legend(title="Metric", loc='upper left', bbox_to_anchor=(1.02, 1))
        _save('model_comparison_molecules.pdf')

    if selected is not None and "stack" not in selected:
        return
    plt.figure(figsize=(12, 6))
    plt.title("Mean of seven task scores (all replies included)", fontsize=18, fontweight='bold')
    x = np.arange(len(plot_order))
    divisor = 7.0
    bands = [
        ("Battleship", np.array([all_data['Battleship'][m].get('Correct Proportion', all_data['Battleship'][m].get('Win', 0)) for m in plot_order]) / divisor),
        ("Tic-Tac-Toe", np.array([all_data['Tic-Tac-Toe'][m].get('Correct Proportion', all_data['Tic-Tac-Toe'][m].get('Win', 0)) for m in plot_order]) / divisor),
        ("Connect-Four", np.array([all_data['Connect-Four'][m].get('Correct Proportion', all_data['Connect-Four'][m].get('Win', 0)) for m in plot_order]) / divisor),
        ("Shapes", np.array([all_data['Shapes'][m].get('Correct Proportion', 0) for m in plot_order]) / divisor),
        ("LCL1", np.array([all_data['LCL'][m].get('Correct Proportion', all_data['LCL'][m].get('Validity', 0)) for m in plot_order]) / divisor),
        ("LCL2", np.array([all_data['LCL'][m].get('Valid Proportion', all_data['LCL'][m].get('Generation', 0)) for m in plot_order]) / divisor),
        ("GtS", np.array([gts_data[m].get('Paper Accuracy', 0) for m in plot_order]) / divisor),
    ]
    stack_colors = ["#7B2D8E", "#086788", "#E6A800", "#DD1C1A", "#FF7F0E", "#5B9279", "#06AED5"]
    if historical:
        plt.stackplot(x, *[band for _, band in bands], labels=[name for name, _ in bands], colors=stack_colors)
    else:
        bottom = np.zeros(len(plot_order))
        for (label, values), color in zip(bands, stack_colors):
            plt.bar(x, values, bottom=bottom, label=label, color=color)
            bottom += values
    plt.xticks(x, [MODEL_LABELS[m] for m in plot_order], rotation=30, ha='right')
    plt.xlabel("Model", fontsize=14, fontweight='bold')
    plt.ylabel("Average Score Contribution", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    _save('merged_metrics_per_model.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render the complete first-paper cohort.')
    parser.add_argument('--historical', action='store_true', help='Explicitly render archived historical comparisons instead')
    args = parser.parse_args()
    main(historical=args.historical)
