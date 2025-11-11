import pandas as pd

input_csv = '.\hyperparam_sweep_results\combined_hyperparam_results.csv'
output_csv = '.\hyperparam_sweep_results\cleaned_hyperparam_results.csv'

df = pd.read_csv(input_csv)


# determine unique runs
cols = ['chunk_size', 'learning_rate', 'batch_size', 'buffer_size', 'total_timesteps']

df_unique = df.drop_duplicates(subset=cols, keep='first')
df_unique.to_csv(output_csv, index=False)

