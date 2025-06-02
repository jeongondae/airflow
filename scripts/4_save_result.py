import joblib
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv('data/iris_processed.csv')
X = df.drop(columns='target')
y = df['target']

model = joblib.load('models/model.pkl')
acc = accuracy_score(y, model.predict(X))

with open('models/metrics.txt', 'w') as f:
    f.write(f"Accuracy: {acc}")
