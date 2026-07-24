from pathlib import Path
import pandas as pd

# ==============================
# Project Paths
# ==============================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"

# ==============================
# Load Data
# ==============================
train = pd.read_csv(RAW_DATA / "train.csv")
stores = pd.read_csv(RAW_DATA / "stores.csv")
transactions = pd.read_csv(RAW_DATA / "transactions.csv")
oil = pd.read_csv(RAW_DATA / "oil.csv")
holidays = pd.read_csv(RAW_DATA / "holidays_events.csv")

# ==============================
# Function to summarize a dataset
# ==============================
def summarize(df, name):
    print("=" * 80)
    print(f"{name}")
    print("=" * 80)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\n")

# ==============================
# Print Summary
# ==============================
summarize(train, "TRAIN")
summarize(stores, "STORES")
summarize(transactions, "TRANSACTIONS")
summarize(oil, "OIL")
summarize(holidays, "HOLIDAYS")