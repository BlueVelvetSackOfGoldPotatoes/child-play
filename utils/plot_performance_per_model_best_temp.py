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
custom_palette = ["#DD1C1A", "#F0C808", "#086788", "#06AED5", "#FFF1D0"]

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
def load_lcl_metrics():
    """
    Loads the LCL metrics: 'Correct Proportion' and 'Valid Proportion'
    from the CSV files, if present.
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
        print(f"Warning: Can't load df_validity_4o_experiments CSV. Error: {e}")
        df_valid = pd.DataFrame(columns=['Model', 'Correct'])
    try:
        df_construct = pd.read_csv(os.path.join(BASE_PATH_LCL, '../lcl_experiments/df_construct.csv'))
    except Exception as e:
        print(f"Warning: Can't load df_construct_4o_experiments CSV. Error: {e}")
        df_construct = pd.DataFrame(columns=['Model', 'Valid'])
    
    # Map the models to simplified names
    if 'Model' in df_valid.columns:
        df_valid['Model'] = df_valid['Model'].map(model_map_lcl).fillna(df_valid['Model'])
    if 'Model' in df_construct.columns:
        df_construct['Model'] = df_construct['Model'].map(model_map_lcl).fillna(df_construct['Model'])
    
    # Compute the mean proportions
    grouped_correct = df_valid.groupby('Model')['Correct'].mean() if not df_valid.empty else {}
    grouped_valid = df_construct.groupby('Model')['Valid'].mean() if not df_construct.empty else {}
    
    # Store the results in the data dict
    for m in MODEL_ORDER:
        data[m] = {}
    # 'Correct Proportion'
    for m in getattr(grouped_correct, 'index', []):
        if m in MODEL_ORDER:
            data[m]['Correct Proportion'] = grouped_correct.loc[m]
    # 'Valid Proportion'
    for m in getattr(grouped_valid, 'index', []):
        if m in MODEL_ORDER:
            data[m]['Valid Proportion'] = grouped_valid.loc[m]
    
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

# ---------------------------------------------------------------------
# Main Aggregation & Plotting
# ---------------------------------------------------------------------
def main():
    # -------------------------
    # Load all the data
    # -------------------------
    shapes_data = load_shapes_metrics()                # picks ONE best temp; stores overall 'Correct Proportion'
    shapes_breakdown = load_shapes_breakdown_by_best_temp(shapes_data)
    
    lcl_data = load_lcl_metrics()
    tictactoe_data = load_boardgame_metrics('tictactoe')
    connectfour_data = load_boardgame_metrics('connectfour')
    battleship_data = load_boardgame_metrics('battleship')
    molecule_data = load_molecule_metrics()  # GtS game

    # We'll gather them to iterate easily
    all_data = {
        'Shapes': shapes_data,
        'LCL': lcl_data,
        'Tic-Tac-Toe': tictactoe_data,
        'Connect-Four': connectfour_data,
        'Battleship': battleship_data,
        'GtS': molecule_data
    }

    sns.set(style="whitegrid")
    plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

    # -----------------------------------------------------
    # 1) Plot each "game" separately
    # -----------------------------------------------------
    for game_name, game_dict in all_data.items():
        if game_name == "Shapes":
            # Single stacked plot using the shape breakdown at the single best temperature
            plt.figure(figsize=(10, 6))
            plt.title("Correct Proportion per Model for the Shapes Game\n(Stacked by Shape Contribution at Best Temp)",
                      fontsize=18, fontweight='bold')
            x = np.arange(len(MODEL_ORDER))
            
            square_vals   = np.array([shapes_breakdown[m].get('square',   0) for m in MODEL_ORDER])
            triangle_vals = np.array([shapes_breakdown[m].get('triangle', 0) for m in MODEL_ORDER])
            cross_vals    = np.array([shapes_breakdown[m].get('cross',    0) for m in MODEL_ORDER])
            
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
            plt.legend(loc='upper right')
            plt.tight_layout()
            plt.show()
        else:
            # For other games, we do a line plot for each metric
            plt.figure(figsize=(10, 6))
            plt.title(f"Best Temperature per Model for {game_name}", fontsize=18, fontweight='bold')
            x_labels = [MODEL_LABELS[m] for m in MODEL_ORDER]
            all_y = []
            color_index = 0
            
            # We'll iterate the metrics found in the first model's dictionary
            if MODEL_ORDER and MODEL_ORDER[0] in game_dict:
                metric_keys = sorted(game_dict[MODEL_ORDER[0]].keys())
            else:
                metric_keys = []
            
            for metric_name in metric_keys:
                # E.g. skip "Avg String Distance" if you want to keep y-lims simpler
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
            plt.legend(title="Metric", loc='upper right')
            plt.tight_layout()
            plt.show()

    # -----------------------------------------------------
    # 2) Compute combined score for each model
    #    (average of 6 key metrics) + standard error
    # -----------------------------------------------------
    combined_scores = []
    for m in MODEL_ORDER:
        metrics = []
        # 1) Shapes -> 'Correct Proportion'
        sh_val = all_data['Shapes'][m].get('Correct Proportion', None)
        if sh_val is not None:
            metrics.append(sh_val)
        # 2) Tic-Tac-Toe -> 'Correct Proportion'
        ttt_val = all_data['Tic-Tac-Toe'][m].get('Correct Proportion', None)
        if ttt_val is not None:
            metrics.append(ttt_val)
        # 3) Connect-Four -> 'Correct Proportion'
        cf_val = all_data['Connect-Four'][m].get('Correct Proportion', None)
        if cf_val is not None:
            metrics.append(cf_val)
        # 4) Battleship -> 'Correct Proportion'
        bs_val = all_data['Battleship'][m].get('Correct Proportion', None)
        if bs_val is not None:
            metrics.append(bs_val)
        # 5) LCL -> 'Correct Proportion'
        lcl_val = all_data['LCL'][m].get('Correct Proportion', None)
        if lcl_val is not None:
            metrics.append(lcl_val)
        # 6) Molecule -> 'Accuracy'
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
    # 3) Plot final stacked area chart of combined score
    #    (6 domain contributions)
    # -----------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.title("Final Combined Score with Stacked Contributions", fontsize=18, fontweight='bold')
    
    x = np.arange(len(MODEL_ORDER))
    # Each domain's contribution = (that domain's metric value) / 6
    shapes_contrib = np.array([all_data['Shapes'][m].get('Correct Proportion', 0) for m in MODEL_ORDER]) / 6.0
    ttt_contrib    = np.array([all_data['Tic-Tac-Toe'][m].get('Correct Proportion', 0) for m in MODEL_ORDER]) / 6.0
    cf_contrib     = np.array([all_data['Connect-Four'][m].get('Correct Proportion', 0) for m in MODEL_ORDER]) / 6.0
    bs_contrib     = np.array([all_data['Battleship'][m].get('Correct Proportion', 0) for m in MODEL_ORDER]) / 6.0
    lcl_contrib    = np.array([all_data['LCL'][m].get('Correct Proportion', 0) for m in MODEL_ORDER]) / 6.0
    gts_contrib    = np.array([all_data['GtS'][m].get('Accuracy', 0)          for m in MODEL_ORDER]) / 6.0
    
    stack_colors = ["#DD1C1A", "#F0C808", "#086788", "#06AED5", "#5B9279", "#FF7F0E"]
    plt.stackplot(x,
                  shapes_contrib,
                  ttt_contrib,
                  cf_contrib,
                  bs_contrib,
                  lcl_contrib,
                  gts_contrib,
                  labels=["Shapes", "Tic-Tac-Toe", "Connect-Four", "Battleship", "LCL", "GtS"],
                  colors=stack_colors)
    
    # Optional error-bar line:
    # combined_vals = [cs['combined_score'] for cs in combined_scores]
    # combined_errs = [cs['combined_err'] for cs in combined_scores]
    # plt.errorbar(x, combined_vals, yerr=combined_errs, fmt='-o', capsize=5, color='black', label='Combined Score')
    
    plt.xticks(x, [MODEL_LABELS[m] for m in MODEL_ORDER])
    plt.xlabel("Model", fontsize=14, fontweight='bold')
    plt.ylabel("Average Score Contribution", fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
