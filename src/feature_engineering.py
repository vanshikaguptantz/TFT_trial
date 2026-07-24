from pathlib import Path
import pandas as pd

# ============================================
# Paths
# ============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# ============================================
# Load Dataset
# ============================================

master_df = pd.read_csv(
    PROCESSED_DATA / "cleaned_dataset.csv"
)

master_df["date"] = pd.to_datetime(master_df["date"])

print("Dataset Loaded")
print(master_df.shape)

# ============================================
# Calendar Features
# ============================================

master_df["year"] = master_df["date"].dt.year

master_df["month"] = master_df["date"].dt.month

master_df["day"] = master_df["date"].dt.day

master_df["day_of_week"] = master_df["date"].dt.dayofweek

master_df["week_of_year"] = master_df["date"].dt.isocalendar().week.astype(int)

master_df["quarter"] = master_df["date"].dt.quarter

# Identify if it's a weekend or not as day of week 0 --> Monday and 6 --> Sunday
master_df["is_weekend"] = (
    master_df["day_of_week"] >= 5
).astype(int)

# Identify month start and month end
master_df["is_month_start"] = master_df["date"].dt.is_month_start.astype(int)

master_df["is_month_end"] = master_df["date"].dt.is_month_end.astype(int)

# Create time ids so that instead of date we get a continous series of numbers to represent time
# For now we are assuming the data to be monthly data, but later we can change this to weekly or daily data. For now, we will create a time index based on months.
master_df["time_idx"] = (
    (master_df["date"].dt.year - master_df["date"].dt.year.min()) * 12
    + master_df["date"].dt.month
)

# Verify the new features in the master dataframe before saving it to a new CSV file
print("=" * 80)
print(master_df.tail())

print("\nNew Columns")

print(master_df.columns)
print("=" * 80)

# Now save the master dataframe with the new features to a new CSV file
master_df.to_csv(
    PROCESSED_DATA / "featured_dataset.csv",
    index=False
)

print("Feature engineering completed.")