import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def main():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train.values.ravel())
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")

if __name__ == "__main__":
    main()

# src/evaluate.py
import pandas as pd
from sklearn.metrics import accuracy_score
import joblib
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("IrisClassifier")

def main():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    model = joblib.load("models/model.pkl")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")
        print(f"Logged accuracy: {acc}")

if __name__ == "__main__":
    main()
