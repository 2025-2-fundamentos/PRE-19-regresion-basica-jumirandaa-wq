import pickle

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def main():
    # Load dataset
    df = pd.read_csv("files/input/auto_mpg.csv")
    df = df.dropna()
    df["Origin"] = df["Origin"].map({1: "USA", 2: "Europe", 3: "Japan"})
    df = pd.get_dummies(df, columns=["Origin"], prefix="", prefix_sep="")

    # Features and labels
    y = df.pop("MPG")
    X = df

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train an MLP regressor
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), random_state=0, max_iter=1000)
    mlp.fit(X_scaled, y)

    # Save artifacts expected by the tests
    with open("mlp.pickle", "wb") as f:
        pickle.dump(mlp, f)

    with open("features_scaler.pickle", "wb") as f:
        pickle.dump(scaler, f)

    print("Saved mlp.pickle and features_scaler.pickle")


if __name__ == "__main__":
    main()
