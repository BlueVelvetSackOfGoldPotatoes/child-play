import pandas as pd

# Path to your CSV file
csv_file = "../molecule_app/evaluation_summaryoa:gpt-4o-mini-2024-07-18.csv"

# Read the CSV
df = pd.read_csv(csv_file)

# Map original model names to the names used in the table
model_map = {
    'oa:gpt-4-1106-preview': 'GPT-4',
    'oa:gpt-3.5-turbo-0125': 'GPT-3.5',
    'oa:gpt-4o-2024-08-06': 'GPT-4o',
    'oa:gpt-4o-mini-2024-07-18': 'GPT-4o-mini'
}
df['Model'] = df['model'].map(model_map)

# Compute the "Incorrect" column (non-invalid errors) as:
# Incorrect = incorrect_count - incorrect_smiles_count
df['Incorrect'] = df['incorrect_count'] - df['incorrect_smiles_count']

# Compute Accuracy (%) = Correct / (Correct + Incorrect) * 100
df['Accuracy (%)'] = (df['correct_count'] / (df['correct_count'] + df['Incorrect'])) * 100

# Round Similarity and Accuracy to match table formatting
df['avg_chemical_similarity'] = df['avg_chemical_similarity'].round(3)
df['Accuracy (%)'] = df['Accuracy (%)'].round(1)

# Rename columns to match the table headers
df = df.rename(columns={
    'temperature': 'Temp.',
    'correct_count': 'Correct',
    'incorrect_smiles_count': 'Invalid',
    'avg_chemical_similarity': 'Similarity'
})

# Select and order columns as in your table: Model, Temp., Correct, Incorrect, Invalid, Similarity, Accuracy (%)
df = df[['Model', 'Temp.', 'Correct', 'Incorrect', 'Invalid', 'Similarity', 'Accuracy (%)']]

# Sort for clarity by Model and Temperature
df = df.sort_values(by=['Model', 'Temp.'])

# Print the resulting table
print(df.to_string(index=False))
