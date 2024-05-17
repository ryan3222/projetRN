import pandas as pd

input_file = 'final_data.csv'
output_file = 'final_data_shuffled.csv'

data = pd.read_csv(input_file, header=None)
shuffled_data = data.sample(frac=1).reset_index(drop=True)
shuffled_data.to_csv(output_file, index=False, header=False)