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
TEMPERATURES = [0, 0.5, 1, 1.5]
BASE_PATH_BOARDGAMES = '../experiment_board_games'

def get_boardgame_results(game, model, temp):
    folder = f"experiment_{game}_{model}_oneshot_temp_{temp}"
    path = os.path.join(BASE_PATH_BOARDGAMES, folder, f"results_{game}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            d = json.load(f)
            wins = d.get('P1 Wins', 0)
            # If ties are present, assume losses = 100 - wins - ties.
            ties = d.get('Ties', 0)
            losses = 100 - wins - ties
            return wins, losses
    except:
        return None

def main():
    rows = []
    games = ['battleship', 'tictactoe', 'connectfour']
    # For each model and temperature, we will print one row per game.
    for m in MODEL_ORDER:
        for t in TEMPERATURES:
            row = {"Model": MODEL_LABELS[m], "Temp.": t}
            for game in games:
                res = get_boardgame_results(game, m, t)
                if res is None:
                    win_rate = np.nan
                    lose_rate = np.nan
                else:
                    wins, losses = res
                    win_rate = wins  # already out of 100
                    lose_rate = losses
                row[f"{game.capitalize()} Win Rate (%)"] = f"{win_rate:.2f}" if not np.isnan(win_rate) else "N/A"
                row[f"{game.capitalize()} Lose Rate (%)"] = f"{lose_rate:.2f}" if not np.isnan(lose_rate) else "N/A"
            rows.append(row)
    df_bg = pd.DataFrame(rows)
    df_bg = df_bg.sort_values(by=["Model", "Temp."])
    print("\n=== Board Games Win/Lose Rates ===")
    print(df_bg.to_string(index=False))

if __name__ == "__main__":
    main()
