import pandas as pd

df = pd.read_csv('data/iris.csv')
df = df[df['target'] != 2]  # binary classification
df.to_csv('data/iris_processed.csv', index=False)
	
