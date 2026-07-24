from pathlib import Path
import pandas as pd

# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# ============================================
# Load Master Dataset
# ============================================

master_df = pd.read_csv(
    PROCESSED_DATA / "master_dataset.csv"
)

print(master_df.shape)

print(master_df.head())

print("=" * 80)
print("Missing Values")
print("=" * 80)

print(master_df.isnull().sum())

# ============================================
# Investigate Missing Transactions
# ============================================

missing_transactions = master_df[
    master_df["transactions"].isnull()
]

print("=" * 80)
print("Missing Transactions")
print("=" * 80)

print("Rows with missing transactions:", len(missing_transactions))

print("\nUnique Dates with Missing Transactions:")
print(missing_transactions["date"].nunique())

print("\nFirst 10 Missing Dates:")
print(sorted(missing_transactions["date"].unique())[:10])

# ============================================
# Investigate Missing Oil Prices
# ============================================

missing_oil = master_df[
    master_df["dcoilwtico"].isnull()
]

print("=" * 80)
print("Missing Oil Prices")
print("=" * 80)

print("Rows with missing oil prices:", len(missing_oil))

print("\nUnique Dates:")
print(missing_oil["date"].nunique())

print("\nFirst 10 Missing Dates:")
print(sorted(missing_oil["date"].unique())[:10])
print(master_df["transactions"].describe())
print(master_df["dcoilwtico"].describe())

# ============================================
# Fill Missing Transactions
# ============================================

master_df["transactions"] = master_df["transactions"].fillna(0)

print("\nMissing Transactions After Filling:")
print(master_df["transactions"].isnull().sum())

# ============================================
# Fill Missing Oil Prices
# ============================================

master_df = master_df.sort_values("date")

master_df["dcoilwtico"] = master_df["dcoilwtico"].ffill()
master_df["dcoilwtico"] = master_df["dcoilwtico"].bfill()

print("\nMissing Oil Prices After Filling:")
print(master_df["dcoilwtico"].isnull().sum())

print("=" * 80)
print("Missing Values After Cleaning")
print("=" * 80)

print(master_df.isnull().sum())

master_df.to_csv(
    PROCESSED_DATA / "cleaned_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")