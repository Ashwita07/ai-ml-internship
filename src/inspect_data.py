from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "breast_cancer_wisconsin.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nColumn data types:")
print(df.dtypes)
print("\nFirst five rows:")
print(df.head())
print("\nTarget counts:")
print(df["target_name"].value_counts())
