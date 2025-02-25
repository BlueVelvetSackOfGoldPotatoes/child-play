#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd

# ---------------------- Configuration ----------------------
MODEL_ORDER = ['gpt3_5', 'gpt4', 'gpt4o_mini', 'gpt4o']
MODEL_LABELS = {
    'gpt3_5': 'GPT-3.5',
    'gpt4': 'GPT-4',
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt4o': 'GPT-4o',
    # Possibly other name mappings:
    'oa_gpt-3.5-turbo-1106': 'GPT-3.5 Turbo',
    'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini',
    'oa:gpt-4o-2024-08-06': 'gpt4o'
}
TEMPERATURES = [0, 0.5, 1, 1.5]

BASE_PATH_SHAPES        = '../experiment_shapes'
BASE_PATH_LCL           = '../lcl_experiments'
BASE_PATH_BOARDGAMES    = '../experiment_board_games'
BASE_PATH_MOLECULE_APP  = '../molecule_app'

# ---------------------------------------------------------------------
# (Include the original functions here)
# ---------------------------------------------------------------------
def load_shapes_metrics():
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

def load_lcl_metrics():
    data = {m: {} for m in MODEL_ORDER}
    model_map_lcl = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-3.5-turbo-1106': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }
    try:
        df_valid = pd.read_csv(os.path.join(BASE_PATH_LCL, '../lcl_experiments/df_validity.csv'))
    except Exception as e:
        print(f"Warning: Can't load validity CSV. Error: {e}")
        df_valid = pd.DataFrame(columns=['Model', 'Correct'])
    try:
        df_construct = pd.read_csv(os.path.join(BASE_PATH_LCL, '../lcl_experiments/df_construct.csv'))
    except Exception as e:
        print(f"Warning: Can't load construct CSV. Error: {e}")
        df_construct = pd.DataFrame(columns=['Model', 'Valid'])
    if 'Model' in df_valid.columns:
        df_valid['Model'] = df_valid['Model'].map(model_map_lcl).fillna(df_valid['Model'])
    if 'Model' in df_construct.columns:
        df_construct['Model'] = df_construct['Model'].map(model_map_lcl).fillna(df_construct['Model'])
    grouped_correct = df_valid.groupby('Model')['Correct'].mean() if not df_valid.empty else {}
    grouped_valid = df_construct.groupby('Model')['Valid'].mean() if not df_construct.empty else {}
    for m in MODEL_ORDER:
        data[m] = {}
    for m in getattr(grouped_correct, 'index', []):
        if m in MODEL_ORDER:
            data[m]['Correct Proportion'] = grouped_correct.loc[m]
    for m in getattr(grouped_valid, 'index', []):
        if m in MODEL_ORDER:
            data[m]['Valid Proportion'] = grouped_valid.loc[m]
    return data

def load_boardgame_metrics(game):
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
        data[m]['Correct Proportion'] = best_wins / 100.0
    return data

def load_molecule_metrics():
    data = {m: {} for m in MODEL_ORDER}
    model_map = {
        'oa:gpt-3.5-turbo-0125': 'gpt3_5',
        'oa:gpt-4-1106-preview': 'gpt4',
        'oa:gpt-4o-2024-08-06': 'gpt4o',
        'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
    }
    csv_path = os.path.join(BASE_PATH_MOLECULE_APP, "evaluation_summaryoa:gpt-4o-mini-2024-07-18.csv")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Warning: Unable to load CSV file {csv_path}. Error: {e}")
        return data
    df['model'] = df['model'].map(model_map).fillna(df['model'])
    df = df[df['model'].isin(MODEL_ORDER)]
    for m in MODEL_ORDER:
        df_model = df[df['model'] == m]
        if df_model.empty:
            continue
        best_row = df_model.loc[df_model['accuracy'].idxmax()]
        correct_count = best_row['correct_count']
        incorrect_count = best_row['incorrect_count']
        total = correct_count + incorrect_count
        accuracy = best_row['accuracy']
        data[m]['Accuracy'] = accuracy
    return data

# -------------------------
# Main: Build Overall Performance Table
# -------------------------
def main():
    shapes_data = load_shapes_metrics()  # shapes: fraction (0 to 1)
    lcl_data = load_lcl_metrics()         # LCL: fractions
    tictactoe_data = load_boardgame_metrics('tictactoe')  # boardgames: fraction wins
    connectfour_data = load_boardgame_metrics('connectfour')
    battleship_data = load_boardgame_metrics('battleship')
    molecule_data = load_molecule_metrics()  # GtS: fraction accuracy

    # Build a list of rows (one per model)
    rows = []
    for m in MODEL_ORDER:
        # Multiply fractions by 100 to get percentages.
        bs = battleship_data.get(m, {}).get('Correct Proportion', 0) * 100
        ttt = tictactoe_data.get(m, {}).get('Correct Proportion', 0) * 100
        cf = connectfour_data.get(m, {}).get('Correct Proportion', 0) * 100
        sh = shapes_data.get(m, {}).get('Correct Proportion', 0) * 100
        # LCL1 from "Correct Proportion" and LCL2 from "Valid Proportion"
        lcl1 = lcl_data.get(m, {}).get('Correct Proportion', 0) * 100
        lcl2 = lcl_data.get(m, {}).get('Valid Proportion', 0) * 100
        gts = molecule_data.get(m, {}).get('Accuracy', 0) * 100
        overall = np.mean([bs, ttt, cf, sh, lcl1, lcl2, gts])
        rows.append({
            "Model": MODEL_LABELS[m],
            "Battleship": f"{bs:.2f}",
            "Tic-Tac-Toe": f"{ttt:.2f}",
            "Connect-Four": f"{cf:.2f}",
            "Shapes": f"{sh:.2f}",
            "LCL1": f"{lcl1:.2f}",
            "LCL2": f"{lcl2:.2f}",
            "GtS": f"{gts:.2f}",
            "Overall": f"{overall:.2f}"
        })

    df_overall = pd.DataFrame(rows)
    print("\n=== Overall ChildPlay Performance ===")
    print(df_overall.to_string(index=False))

if __name__ == "__main__":
    main()
