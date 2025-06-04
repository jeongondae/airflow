import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# 데이터 로딩
df = pd.read_csv('data/iris_preprocessed.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 모델 학습
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 모델 저장
joblib.dump(model, 'models/model.pkl')

# MLflow 로깅
mlflow.set_experiment("Iris_Classifier")
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_metric("train_score", model.score(X_train, y_train))
