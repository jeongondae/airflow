import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv('data/iris_processed.csv')
X = df.drop(columns='target')
y = df['target']

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, 'models/model.pkl')
