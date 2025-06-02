import pandas as pd
from sklearn.datasets import load_iris

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df['target'] = load_iris().target
df.to_csv('data/iris.csv', index=False)
