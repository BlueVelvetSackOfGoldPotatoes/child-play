import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------- Configuration ----------------------
MODEL_ORDER = ['gpt3_5', 'gpt4', 'gpt4o_mini', 'gpt4o']
MODEL_LABELS = {
    'gpt3_5': 'GPT-3.5',
    'gpt4': 'GPT-4',
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt4o': 'GPT-4o',
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
custom_palette = ["#B53C68", "#3772FF", "#FDCA40", "#DF2935", "#EE7A3B", "#8B4E9A", "#358935"]

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
    """
    shape_folders = ['square', 'triangle', 'cross']
    data_out = {m: {} for m in MODEL_ORDER}
    
    for m in MODEL_ORDER:
        best_temp = shapes_data[m].get('Best Temp', None)
        if best_temp is None:
            for shape in shape_folders:
                data_out[m][shape] = 0.0
            continue
        
        total_wins_all_shapes = 0
        total_attempts_all_shapes = 0
        shape_wins = {}
        
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
        
        for shape in shape_folders:
            if total_attempts_all_shapes > 0:
                data_out[m][shape] = shape_wins[shape] / total_attempts_all_shapes
            else:
                data_out[m][shape] = 0.0
    
    return data_out

# ---------------------------------------------------------------------
# 2) LCL
# ---------------------------------------------------------------------
def load_lcl_metrics():
    """
    Loads the LCL metrics: 'Correct Proportion' and 'Valid Proportion'
    from the CSV files, selecting, for each model, the row with the highest value.
    """
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
        print(f"Warning: Can't load df_validity CSV. Error: {e}")
        df_valid = pd.DataFrame(columns=['Model', 'Correct'])
    try:
        df_construct = pd.read_csv(os.path.join(BASE_PATH_LCL, '../lcl_experiments/df_construct.csv'))
    except Exception as e:
        print(f"Warning: Can't load df_construct CSV. Error: {e}")
        df_construct = pd.DataFrame(columns=['Model', 'Valid'])
    
    if 'Model' in df_valid.columns:
        df_valid['Model'] = df_valid['Model'].map(model_map_lcl).fillna(df_valid['Model'])
    if 'Model' in df_construct.columns:
        df_construct['Model'] = df_construct['Model'].map(model_map_lcl).fillna(df_construct['Model'])
    
    best_valid = {}
    if not df_valid.empty:
        for m in MODEL_ORDER:
            df_model = df_valid[df_valid['Model'] == m]
            if not df_model.empty:
                best_row = df_model.loc[df_model['Correct'].idxmax()]
                best_valid[m] = best_row['Correct']
    best_construct = {}
    if not df_construct.empty:
        for m in MODEL_ORDER:
            df_model = df_construct[df_construct['Model'] == m]
            if not df_model.empty:
                best_row = df_model.loc[df_model['Valid'].idxmax()]
                best_construct[m] = best_row['Valid']
    
    for m in MODEL_ORDER:
        if m in best_valid:
            data[m]['Correct Proportion'] = best_valid[m]
        if m in best_construct:
            data[m]['Valid Proportion'] = best_construct[m]
    
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
                    wins = d.get('P1 Wins', 0)
                    if wins > best_wins:
                        best_wins = wins
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
        data[m]['Correct Proportion'] = best_wins / 100.0
        data[m]['Tie Proportion'] = ties / 100.0
        data[m]['Wrong Moves Proportion'] = wrong_moves / 100.0
    return data

# ---------------------------------------------------------------------
# 4) Molecule App
# ---------------------------------------------------------------------
def load_molecule_metrics():
    """
    For each model, reads the CSV file evaluation_summaryoa:gpt-4o-mini-2024-07-18.csv and selects 
    the best row based on highest accuracy. Returns:
      - 'Accuracy'
      - 'Avg Chem. Similarity'
      - 'Avg String Distance'
      - 'Incorrect SMILES Fraction' (calculated as incorrect_smiles_count / (correct_count + incorrect_count))
    """
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
        avg_chem_similarity = best_row['avg_chemical_similarity']
        avg_string_distance = best_row['avg_string_distance']
        incorrect_smiles_fraction = best_row['incorrect_smiles_count'] / total if total > 0 else 0.0
        data[m]['Accuracy'] = accuracy
        data[m]['Avg Chem. Similarity'] = avg_chem_similarity
        data[m]['Avg String Distance'] = avg_string_distance
        data[m]['Incorrect SMILES Fraction'] = incorrect_smiles_fraction
    return data

# ---------------------------------------------------------------------
# Main Aggregation & Plotting
# ---------------------------------------------------------------------
def main():
    # -------------------------
    # Load all the data using best temperature per model
    # -------------------------
    shapes_data = load_shapes_metrics()
    shapes_breakdown = load_shapes_breakdown_by_best_temp(shapes_data)
    lcl_data = load_lcl_metrics()
    tictactoe_data = load_boardgame_metrics('tictactoe')
    connectfour_data = load_boardgame_metrics('connectfour')
    battleship_data = load_boardgame_metrics('battleship')
    molecule_data = load_molecule_metrics()

    # Override LCL data with provided best-performance values (as fractions)
    # Provided overall performance table:
    # GPT-3.5: LCL1 = 50.00% (0.50), LCL2 = 1.00% (0.01)
    # GPT-4:   LCL1 = 51.00% (0.51), LCL2 = 2.00% (0.02)
    # GPT-4o-mini: LCL1 = 57.00% (0.57), LCL2 = 6.00% (0.06)
    # GPT-4o:  LCL1 = 75.00% (0.75), LCL2 = 36.00% (0.36)
    lcl_data_override = {
        'gpt3_5': {'Correct Proportion': 0.50, 'Valid Proportion': 0.01},
        'gpt4':   {'Correct Proportion': 0.51, 'Valid Proportion': 0.02},
        'gpt4o_mini': {'Correct Proportion': 0.57, 'Valid Proportion': 0.06},
        'gpt4o':  {'Correct Proportion': 0.75, 'Valid Proportion': 0.36}
    }
    lcl_data = lcl_data_override

    # Define override overall performance values (best temp per model) as given in the table:
    overall_perf = {
        'gpt3_5': {'Battleship': 10.00, 'TicTacToe': 53.00, 'ConnectFour': 76.00, 'Shapes': 37.33, 'LCL1': 50.00, 'LCL2': 1.00, 'GtS': 1.30},
        'gpt4':   {'Battleship': 4.00,  'TicTacToe': 77.00, 'ConnectFour': 80.00, 'Shapes': 92.00, 'LCL1': 51.00, 'LCL2': 2.00, 'GtS': 6.40},
        'gpt4o_mini': {'Battleship': 0.00, 'TicTacToe': 61.00, 'ConnectFour': 79.00, 'Shapes': 52.00, 'LCL1': 57.00, 'LCL2': 6.00, 'GtS': 3.40},
        'gpt4o':  {'Battleship': 0.00,  'TicTacToe': 92.00, 'ConnectFour': 80.00, 'Shapes': 97.33, 'LCL1': 75.00, 'LCL2': 36.00, 'GtS': 5.70}
    }

    # Gather data into a dictionary for easy iteration
    all_data = {
        'Shapes': shapes_data,
        'LCL': lcl_data,
        'Tic-Tac-Toe': tictactoe_data,
        'Connect-Four': connectfour_data,
        'Battleship': battleship_data,
        'GtS': molecule_data
    }

    # Define legend locations for each plot type
    legend_locs = {
        "Shapes": "upper left",
        "LCL": "upper left",
        "Tic-Tac-Toe": "upper left",
        "Connect-Four": "center right",
        "Battleship": "center right",
        "GtS": "upper left"
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

    # -----------------------------------------------------
    # 1) Plot each "game" separately
    # -----------------------------------------------------
    for game_name, game_dict in all_data.items():
        if game_name == "Shapes":
            plt.figure(figsize=(10, 6))
            plt.title("Correct Proportion per Model for the Shapes Game\n(Stacked by Shape Contribution at Best Temp)",
                      fontsize=18, fontweight='bold')
            x = np.arange(len(MODEL_ORDER))
            square_vals = np.array([shapes_breakdown[m].get('square', 0) for m in MODEL_ORDER])
            triangle_vals = np.array([shapes_breakdown[m].get('triangle', 0) for m in MODEL_ORDER])
            cross_vals = np.array([shapes_breakdown[m].get('cross', 0) for m in MODEL_ORDER])
            plt.stackplot(
                x,
                square_vals,
                triangle_vals,
                cross_vals,
                labels=["Square", "Triangle", "Cross"],
                colors=[custom_palette[i] for i in range(3)]
            )
            plt.xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER])
            plt.xlabel("Model", fontsize=14, fontweight='bold')
            plt.ylabel("Correct Proportion (stacked)", fontsize=14, fontweight='bold')
            plt.legend(loc="upper left")
            plt.tight_layout()
            plt.show()
        else:
            plt.figure(figsize=(10, 6))
            plt.title(f"Best Temperature per Model for {game_name}", fontsize=18, fontweight='bold')
            x_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
            all_y = []
            color_index = 0
            if MODEL_ORDER and MODEL_ORDER[0] in game_dict:
                metric_keys = sorted(game_dict[MODEL_ORDER[0]].keys())
            else:
                metric_keys = []
            for metric_name in metric_keys:
                if metric_name == "Avg String Distance":
                    continue
                y_vals = []
                for m in MODEL_ORDER:
                    val = game_dict[m].get(metric_name, np.nan)
                    y_vals.append(val)
                all_y.extend([v for v in y_vals if not np.isnan(v)])
                plt.plot(
                    x_labels,
                    y_vals,
                    marker='o',
                    label=metric_name,
                    color=custom_palette[color_index % len(custom_palette)]
                )
                color_index += 1
            if all_y:
                global_min = min(all_y)
                global_max = max(all_y)
                margin = (global_max - global_min) * 0.1 if (global_max - global_min) != 0 else 0.1
                plt.ylim(global_min - margin, global_max + margin)
            else:
                plt.ylim(0, 1)
            plt.xlabel("Model", fontsize=14, fontweight='bold')
            plt.ylabel("Proportion / Fraction", fontsize=14, fontweight='bold')
            legend_loc = legend_locs.get(game_name, "upper left")
            plt.legend(title="Metric", loc=legend_loc)
            plt.tight_layout()
            plt.show()

    # -----------------------------------------------------
    # 2) Compute combined score for each model using best temperature values per domain
    # -----------------------------------------------------
    combined_scores = []
    for m in MODEL_ORDER:
        metrics = []
        sh_val = all_data['Shapes'][m].get('Correct Proportion', None)
        if sh_val is not None:
            metrics.append(sh_val)
        ttt_val = all_data['Tic-Tac-Toe'][m].get('Correct Proportion', None)
        if ttt_val is not None:
            metrics.append(ttt_val)
        cf_val = all_data['Connect-Four'][m].get('Correct Proportion', None)
        if cf_val is not None:
            metrics.append(cf_val)
        bs_val = all_data['Battleship'][m].get('Correct Proportion', None)
        if bs_val is not None:
            metrics.append(bs_val)
        lcl_val = all_data['LCL'][m].get('Correct Proportion', None)
        if lcl_val is not None:
            metrics.append(lcl_val)
        mol_val = all_data['GtS'][m].get('Accuracy', None)
        if mol_val is not None:
            metrics.append(mol_val)
        if metrics:
            mean_val = np.mean(metrics)
            std_val = np.std(metrics)
            combined_err = std_val / np.sqrt(len(metrics))
        else:
            mean_val = 0.0
            combined_err = 0.0
        combined_scores.append({
            'model': m,
            'combined_score': mean_val,
            'combined_err': combined_err
        })

    # -----------------------------------------------------
    # 3) Plot final stacked area chart of combined score using provided overall performance values
    #    (We use the override values from the overall performance table.)
    # -----------------------------------------------------
    overall_perf = {
        'gpt3_5': {'Battleship': 10.00, 'TicTacToe': 53.00, 'ConnectFour': 76.00, 'Shapes': 37.33, 'LCL1': 50.00, 'LCL2': 1.00, 'GtS': 1.30},
        'gpt4':   {'Battleship': 4.00,  'TicTacToe': 77.00, 'ConnectFour': 80.00, 'Shapes': 92.00, 'LCL1': 51.00, 'LCL2': 2.00, 'GtS': 6.40},
        'gpt4o_mini': {'Battleship': 0.00, 'TicTacToe': 61.00, 'ConnectFour': 79.00, 'Shapes': 52.00, 'LCL1': 57.00, 'LCL2': 6.00, 'GtS': 3.40},
        'gpt4o':  {'Battleship': 0.00,  'TicTacToe': 92.00, 'ConnectFour': 80.00, 'Shapes': 97.33, 'LCL1': 75.00, 'LCL2': 36.00, 'GtS': 5.70}
    }
    # Use best-temp performance values (do not average over temperatures)
    domain_keys = ['Battleship', 'TicTacToe', 'ConnectFour', 'Shapes', 'LCL1', 'LCL2', 'GtS']
    stacked_data = {m: [] for m in MODEL_ORDER}
    for m in MODEL_ORDER:
        for key in domain_keys:
            stacked_data[m].append(overall_perf[m][key] / 7.0)
    x = np.arange(len(MODEL_ORDER))
    battleship_arr = np.array([stacked_data[m][0] for m in MODEL_ORDER])
    tictactoe_arr  = np.array([stacked_data[m][1] for m in MODEL_ORDER])
    connectfour_arr= np.array([stacked_data[m][2] for m in MODEL_ORDER])
    shapes_arr     = np.array([stacked_data[m][3] for m in MODEL_ORDER])
    lcl1_arr       = np.array([stacked_data[m][4] for m in MODEL_ORDER])
    lcl2_arr       = np.array([stacked_data[m][5] for m in MODEL_ORDER])
    gts_arr        = np.array([stacked_data[m][6] for m in MODEL_ORDER])
    
    plt.figure(figsize=(10, 6))
    plt.title("Final Combined Score with Stacked Contributions", fontsize=18, fontweight='bold')
    plt.stackplot(x,
                  battleship_arr,
                  tictactoe_arr,
                  connectfour_arr,
                  shapes_arr,
                  lcl1_arr,
                  lcl2_arr,
                  gts_arr,
                  labels=["Battleship", "Tic-Tac-Toe", "Connect-Four", "Shapes", "LCL1", "LCL2", "GtS"],
                  colors=custom_palette[:7])
    plt.xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER])
    plt.xlabel("Model", fontsize=14, fontweight='bold')
    plt.ylabel("Score Contribution (%)", fontsize=14, fontweight='bold')
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # -----------------------------------------------------
    # (Commented out overall LCL performance table plot using provided numbers)
    """
    % \begin{table}[H]
    % \centering
    % \resizebox{1.0\textwidth}{!}{
    % \begin{tabular}{lcccccccc}
    % \toprule
    % \textbf{Model} & \textbf{Battleship} & \textbf{Tic-Tac-Toe} & \textbf{Connect-Four} & \textbf{Shapes} & \textbf{LCL1} & \textbf{LCL2} & \textbf{GtS} & \textbf{Overall} \\ 
    % \midrule
    % \textbf{GPT-3.5}      & \textbf{10.00} & 53.00 & 76.00 & 37.33 & 50.00 & 1.00 & 1.30 & 32.66 \\
    % \textbf{GPT-4}        & 4.00 & 77.00 & \textbf{80.00} & 92.00 & 51.00 & 2.00 & \textbf{6.40} & 44.63 \\
    % \textbf{GPT-4o-mini}  & 0.00 & 61.00 & 79.00 & 52.00 & 57.00 & 6.00 & 3.40 & 36.91 \\ 
    % \textbf{GPT-4o}       & 0.00 & \textbf{92.00} & \textbf{80.00} & \textbf{97.33} & \textbf{75.00} & \textbf{36.00} & 5.70 & \textbf{55.15} \\
    % \bottomrule
    % \end{tabular}
    % }
    % \caption{Summary of the best LLM performances on each of the benchmark tasks and overall ChildPlay performance. LCL1 corresponds to validity testing, LCL2 to construct generation.}
    % \label{tab:overall_performance}
    % \end{table}
    """

if __name__ == "__main__":
    main()
