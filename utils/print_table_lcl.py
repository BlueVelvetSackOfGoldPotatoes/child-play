#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

# ---------------------- Configuration ----------------------
MODEL_ORDER = ['gpt3_5', 'gpt4', 'gpt4o_mini', 'gpt4o']
MODEL_LABELS = {
    'gpt3_5': 'GPT-3.5',
    'gpt4': 'GPT-4',
    'gpt4o_mini': 'GPT-4o-mini',
    'gpt4o': 'GPT-4o'
}
BASE_PATH_LCL = '../lcl_experiments'
model_map_lcl = {
    'oa:gpt-3.5-turbo-0125': 'gpt3_5',
    'oa:gpt-3.5-turbo-1106': 'gpt3_5',
    'oa:gpt-4-1106-preview': 'gpt4',
    'oa:gpt-4o-2024-08-06': 'gpt4o',
    'oa:gpt-4o-mini-2024-07-18': 'gpt4o_mini'
}

# -------------------------
# Load CSVs and compute per temperature stats
# -------------------------
def compute_lcl_stats():
    try:
        df_valid = pd.read_csv(os.path.join(BASE_PATH_LCL, '../lcl_experiments/df_validity.csv'))
    except Exception as e:
        print(f"Error loading validity CSV: {e}")
        return None, None
    try:
        df_construct = pd.read_csv(os.path.join(BASE_PATH_LCL, '../lcl_experiments/df_construct.csv'))
    except Exception as e:
        print(f"Error loading construct CSV: {e}")
        return None, None

    # Map model names
    df_valid['Model'] = df_valid['Model'].map(model_map_lcl).fillna(df_valid['Model'])
    df_construct['Model'] = df_construct['Model'].map(model_map_lcl).fillna(df_construct['Model'])

    # Assume there is a "temperature" column in both CSVs.
    return df_valid, df_construct

def main():
    df_valid, df_construct = compute_lcl_stats()
    if df_valid is None or df_construct is None:
        return

    # We'll build rows: one for each (model, temperature)
    rows = []
    for m in MODEL_ORDER:
        for t in sorted(df_valid['Temperature'].unique()):
            # Filter by model and temperature
            df_v = df_valid[(df_valid['Model'] == m) & (df_valid['Temperature'] == t)]
            df_c = df_construct[(df_construct['Model'] == m) & (df_construct['Temperature'] == t)]
            if df_v.empty or df_c.empty:
                continue
            # Compute mean and standard error for Validity
            valid_vals = df_v['Correct'].values
            mean_valid = np.mean(valid_vals) * 100
            se_valid = (np.std(valid_vals, ddof=1) / np.sqrt(len(valid_vals))) * 100
            # For Construct Generation (assume column name "Valid")
            construct_vals = df_c['Valid'].values
            mean_construct = np.mean(construct_vals) * 100
            se_construct = (np.std(construct_vals, ddof=1) / np.sqrt(len(construct_vals))) * 100
            rows.append({
                "Model": MODEL_LABELS[m],
                "Temp.": t,
                "Validity (%)": f"{mean_valid:.2f}",
                "SE Validity (%)": f"{se_valid:.2f}",
                "Construct (%)": f"{mean_construct:.2f}",
                "SE Construct (%)": f"{se_construct:.2f}"
            })
    df_table = pd.DataFrame(rows)
    # Sort by model then temperature
    df_table = df_table.sort_values(by=["Model", "Temp."])
    print("\n=== LCL Proportions Table ===")
    print(df_table.to_string(index=False))

if __name__ == "__main__":
    main()
