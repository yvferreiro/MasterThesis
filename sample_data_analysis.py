import pandas as pd
df = pd.read_json ('sample_dataset.json', orient='split')
print(df)
