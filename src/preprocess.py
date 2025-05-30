import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)

def main():
    df = pd.read_csv("data/raw/iris.csv")
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["train"]["test_size"], random_state=params["train"]["random_state"]
    )
    os.makedirs("data/processed", exist_ok=True)
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

if __name__ == "__main__":
    main()
