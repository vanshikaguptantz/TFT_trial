# ==============================================Block 1==========================================
from pathlib import Path
import pandas as pd

# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DATA.mkdir(parents=True, exist_ok=True)

# ============================================
# Load datasets
# ============================================

train = pd.read_csv(RAW_DATA / "train.csv")
stores = pd.read_csv(RAW_DATA / "stores.csv")
transactions = pd.read_csv(RAW_DATA / "transactions.csv")
oil = pd.read_csv(RAW_DATA / "oil.csv")
holidays = pd.read_csv(RAW_DATA / "holidays_events.csv")

# ============================================
# Convert date columns to datetime
# ============================================

train["date"] = pd.to_datetime(train["date"])
transactions["date"] = pd.to_datetime(transactions["date"])
oil["date"] = pd.to_datetime(oil["date"])
holidays["date"] = pd.to_datetime(holidays["date"])

print("Datasets loaded successfully.")

# ============================================ Block 2 ============================================
# ============================================
# Merge Store Information
# ============================================

master_df = train.merge(
    stores,
    on="store_nbr",
    how="left"
)

print("\nAfter merging Stores:")
print(master_df.shape)
print(master_df.head())

# =========================================== Block 3 ============================================

# ============================================
# Merge Transactions
# ============================================

master_df = master_df.merge(
    transactions,
    on=["date", "store_nbr"],
    how="left"
)

print("\nAfter merging Transactions:")
print(master_df.shape)

print("\nMissing transaction values:")
print(master_df["transactions"].isnull().sum())

print("\nFirst 10 rows:")
print(master_df.head(10))

#==============================BLOCK4===========================================
# ============================================
# Merge Oil Prices
# ============================================

master_df = master_df.merge(
    oil,
    on="date",
    how="left"
)

print("\nAfter merging Oil:")
print(master_df.shape)

print("\nMissing Oil Prices:")
print(master_df["dcoilwtico"].isnull().sum())

print("\nFirst 5 Rows:")
print(master_df.head())

# ==============================BLOCK5===========================================
# ============================================
# Explore Holidays
# ============================================

print("=" * 80)
print("Holiday Types")
print("=" * 80)

print(holidays["locale"].value_counts())

print("\nHoliday Categories")
print("=" * 80)

print(holidays["type"].value_counts())

print("\nSample Holidays")
print("=" * 80)

print(holidays.head(20))

#==============================BLOCK6===========================================
# ============================================
# Save Merged Dataset
# ============================================

master_df.to_csv(
    PROCESSED_DATA / "master_dataset.csv",
    index=False
)

print("\nMaster dataset saved successfully.")