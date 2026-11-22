
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "breast_cancer_wisconsin.csv"


def load_and_preprocess_data():
    df = pd.read_csv(DATA_PATH)

    print("Missing values:", df.isnull().sum().sum())
    print("Duplicate rows:", df.duplicated().sum())

    df = df.drop_duplicates()

    X = df.drop(columns=["target", "target_name"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    load_and_preprocess_data()
