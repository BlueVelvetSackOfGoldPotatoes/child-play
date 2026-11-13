import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import argparse  # Added for command-line argument parsing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def is_winning_move(board, player):
    # Check all rows and columns
    for i in range(3):
        if np.all(board[i, :] == player):
            return True
        if np.all(board[:, i] == player):
            return True
    # Check both diagonals
    if (board[0, 0] == player and board[1, 1] == player and board[2, 2] == player) or \
       (board[0, 2] == player and board[1, 1] == player and board[2, 0] == player):
        return True
    return False

def can_win_next(board, player):
    rows, cols = board.shape
    for row in range(rows):
        for col in range(cols):
            if board[row][col] == 0:  # Check unoccupied cells
                board[row][col] = player  # Hypothetically place player's mark
                if is_winning_move(board, player):
                    board[row][col] = 0  # Reset after check
                    return True
                board[row][col] = 0  # Reset after check
    return False

def count_moves(game_logs):
    """Count immediate tactical errors with explicit episode boundaries."""
    from utils.board_move_metrics import count_moves as count_episode_moves
    return count_episode_moves(game_logs, 3, 3, False, is_winning_move, can_win_next)


def plot_results(models, results, conditions, plot_type='bar'):
    data_to_plot = []
    stats_to_plot = []

    for model in models:
        for condition in conditions:
            result = results.get(model, {}).get(condition, {})
            data_for_model_condition = {'Average Moves': [], 'Missed Wins': [], 'Missed Blocks': []}
            for player_type in (0, 1):
                average_moves = result.get('average_moves', {}).get(player_type, 0)
                missed_wins = result.get('missed_wins', {}).get(player_type, 0)
                missed_blocks = result.get('missed_blocks', {}).get(player_type, 0)

                player_label = 'Model' if player_type == 0 else 'Random'
                data_to_plot.append({
                    'Model': model,
                    'Temperature': condition,
                    'Player': f'{model} {player_label}',
                    'Average Moves': average_moves,
                    'Missed Wins': missed_wins,
                    'Missed Blocks': missed_blocks
                })

                data_for_model_condition['Average Moves'].append(average_moves)
                data_for_model_condition['Missed Wins'].append(missed_wins)
                data_for_model_condition['Missed Blocks'].append(missed_blocks)

            model_std = {stat: np.std(data_for_model_condition[stat]) for stat in data_for_model_condition}
            stats_to_plot.append({
                'Model': model,
                'Temperature': condition,
                'Std': model_std
            })

    df = pd.DataFrame(data_to_plot)
    std_df = pd.DataFrame(stats_to_plot)

    if plot_type == 'bar':
        plt.figure(figsize=(24, 6))
        bar_width = 0.10  # Reduced width to accommodate multiple bars side by side

        for i, measure in enumerate(['Average Moves', 'Missed Wins', 'Missed Blocks']):
            ax = plt.subplot(1, 3, i+1)
            unique_temperatures = df['Temperature'].unique().astype(str)
            temp_positions = np.arange(len(unique_temperatures))

            offset = -bar_width * len(models) * 0.5  # Initialize offset

            for idx, model in enumerate(models):
                for player_type in ['Model', 'Random']:
                    condition_data = df[(df['Model'] == model) & (df['Player'].str.contains(player_type))]
                    values = condition_data[measure].values
                    
                    if measure == 'Average Moves':
                        errors = condition_data.apply(lambda x: std_df[(std_df['Model'] == x['Model']) & (std_df['Temperature'] == x['Temperature'])]['Std'].iloc[0][measure], axis=1)
                        corrected_errors = np.where(values - errors < 0, values, errors)  # Avoid negative values
                        if player_type == 'Random':
                            ax.bar(temp_positions + offset, values, width=bar_width, label=f'Random Player (vs {model})', yerr=corrected_errors, capsize=5)
                        else:
                            ax.bar(temp_positions + offset, values, width=bar_width, label=f'{model} {player_type}', yerr=corrected_errors, capsize=5)
                    else:
                        if player_type == 'Random':
                            ax.bar(temp_positions + offset, values, width=bar_width, label=f'Random Player (vs {model})')
                        else:
                            ax.bar(temp_positions + offset, values, width=bar_width, label=f'{model} {player_type}')

                    offset += bar_width

            ax.set_xticks(temp_positions)
            ax.set_xticklabels([temp.split('_')[1] for temp in unique_temperatures], fontsize=12, fontweight='bold')
            ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')

            ax.set_title(measure, fontsize=14, fontweight='bold')
            ax.set_ylabel('')  # Remove y-axis label

            if i == 2:  # Only add legend to the last plot
                ax.legend(title='Player and Model', title_fontsize='12', fontsize='10', loc='upper right')

        plt.subplots_adjust(wspace=0.3)  # Increase space between plots
        plt.tight_layout()
        plt.show()

    elif plot_type == 'scatter':
        fig, axes = plt.subplots(3, 1, figsize=(24, 18))  # Create a shared figure with subplots
        handles, labels = None, None  # Initialize variables for legend handles and labels

        for i, (ax, measure) in enumerate(zip(axes, ['Average Moves', 'Missed Wins', 'Missed Blocks'])):
            # Create the lineplot without a legend
            sns.lineplot(
                data=df,
                x='Temperature',
                y=measure,
                hue='Player',
                style='Model',
                markers=True,
                dashes=False,
                markersize=10,
                linewidth=2,
                palette='deep',
                ax=ax,
                legend=(i == 0)  # Only include the legend for the first subplot
            )

            # Set titles and labels
            ax.set_title(measure, fontsize=14, fontweight='bold')
            ax.set_ylabel(measure, fontsize=12, fontweight='bold')

            # Remove x-axis labels for all but the last subplot
            if i < 2:  # If not the last subplot
                ax.set_xlabel('')
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Temperature', fontsize=12, fontweight='bold')

            # Collect handles and labels from the first subplot
            if i == 0:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend_.remove()  # Remove the legend from the first subplot after capturing handles

        # Add the legend outside the top-most plot
        if handles and labels:  # Ensure handles and labels are not None
            fig.legend(
                handles,
                labels,
                title='Player and Model',
                bbox_to_anchor=(0.96, 0.89)
            )
        plt.subplots_adjust(right=0.85, hspace=0.3)  # Adjust right to make space for legend
        # plt.tight_layout()
        # Adjust layout to ensure all plots and the legend are fully visible
        plt.show()

def average_stats(stats_list):
    averaged_stats = {}
    keys = stats_list[0].keys()
    for key in keys:
        if isinstance(stats_list[0][key], dict):
            averaged_stats[key] = {k: np.mean([d[key][k] for d in stats_list]) for k in stats_list[0][key].keys()}
        else:
            averaged_stats[key] = np.mean([d[key] for d in stats_list])
    return averaged_stats

def process_experiments(base_path, models, conditions):
    all_results = {}
    for model in models:
        model_results = {}
        print(f"Processing results for model: {model}")
        for condition in conditions:
            file_name = f'experiment_tictactoe_{model}_oneshot_{condition}/game_logs_tictactoe.json'
            path = os.path.join(base_path, file_name)
            with open(path, 'r') as file:
                game_logs = json.load(file)
                game_stats = count_moves(game_logs)
                print("-"*50)
                print(path)
                print(game_stats)

                model_results[condition] = game_stats
        all_results[model] = model_results
        
    print("=" * 80)
    print("Final Results Summary:")
    for model, conditions_results in all_results.items():
        for condition, stats in conditions_results.items():
            print(f"Model {model}, Condition {condition}:")
            print("  Model Player Stats:")
            print(f"    Average Moves: {stats['average_moves'][0]}")
            print(f"    Missed Wins: {stats['missed_wins'][0]}")
            print(f"    Missed Blocks: {stats['missed_blocks'][0]}")
            print("  Random Player Stats:")
            print(f"    Average Moves: {stats['average_moves'][1]}")
            print(f"    Missed Wins: {stats['missed_wins'][1]}")
            print(f"    Missed Blocks: {stats['missed_blocks'][1]}")
            print(f"  Total Games Played: {stats.get('total_games', 'Data not available')}")
            print("-" * 50)
    print("=" * 80)

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Process Tic-Tac-Toe game logs and generate plots.')
    parser.add_argument('--scatter', action='store_true', help='Generate scatter (line) plots instead of bar plots.')
    args = parser.parse_args()

    base_path = '../experiment_board_games'
    models = ['gpt3_5', 'gpt4', 'gpt4o', 'gpt4o_mini']
    conditions = ['temp_0', 'temp_0.5', 'temp_1', 'temp_1.5']

    results = process_experiments(base_path, models, conditions)
    plot_type = 'scatter' if args.scatter else 'bar'
    plot_results(models, results, conditions, plot_type=plot_type)

if __name__ == '__main__':
    main()
