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
    'gpt4o': 'GPT-4o'
}
TEMPERATURES = [0, 0.0, 0.5, 1, 1.5]  # include 0.0 explicitly if needed
BASE_PATH_SHAPES = '../experiment_shapes'
SHAPES = ['square', 'triangle', 'cross']

def get_shape_stats(model, temp, shape):
    path = os.path.join(
        BASE_PATH_SHAPES,
        model.replace(":", "_"),
        str(temp).replace(".", "_"),
        shape,
        'results.json'
    )
    if not os.path.exists(path):
        return 0, 0  # wins, total attempts
    try:
        with open(path, 'r') as f:
            d = json.load(f)
            wins = d.get('P1 Wins', d.get('Wins', 0))
            losses = d.get('Losses', d.get('P2 Wins', 0))
            total = wins + losses
            return wins, total
    except:
        return 0, 0

def main():
    rows = []
    for m in MODEL_ORDER:
        for t in TEMPERATURES:
            shape_stats = {}
            total_wins = 0
            total_attempts = 0
            for shape in SHAPES:
                wins, total = get_shape_stats(m, t, shape)
                shape_stats[shape] = (wins, total)
                total_wins += wins
                total_attempts += total
            # For each shape, compute percentages (if total_attempts>0)
            row = {"Model": MODEL_LABELS[m], "Temp.": t}
            for shape in SHAPES:
                wins, total = shape_stats[shape]
                corr = (wins / total * 100) if total > 0 else 0
                incorr = 100 - corr if total > 0 else 0
                row[f"{shape.capitalize()} Correct (%)"] = f"{corr:.2f}"
                row[f"{shape.capitalize()} Incorrect (%)"] = f"{incorr:.2f}"
            overall_corr = (total_wins / total_attempts * 100) if total_attempts > 0 else 0
            overall_incorr = 100 - overall_corr if total_attempts > 0 else 0
            row["Overall Correct (%)"] = f"{overall_corr:.2f}"
            row["Overall Incorrect (%)"] = f"{overall_incorr:.2f}"
            rows.append(row)
    df_shapes = pd.DataFrame(rows)
    # Optionally sort the table
    df_shapes = df_shapes.sort_values(by=["Model", "Temp."])
    print("\n=== Shapes Breakdown Comparison ===")
    print(df_shapes.to_string(index=False))

if __name__ == "__main__":
    main()
