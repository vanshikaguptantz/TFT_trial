#==============================Block1=========================
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA = PROJECT_ROOT / "data" / "raw"

train = pd.read_csv(RAW_DATA / "train.csv")

# Convert date to datetime
train["date"] = pd.to_datetime(train["date"])

print("=" * 80)
print("DATE INFORMATION")
print("=" * 80)

print("Start Date :", train["date"].min())
print("End Date   :", train["date"].max())

print("\nTotal Days :", train["date"].nunique())

print("\n")

#==============================Block2=========================

print("=" * 80)
print("STORE INFORMATION")
print("=" * 80)

print("Number of Stores:")
print(train["store_nbr"].nunique())

print("\nStore IDs:")
print(sorted(train["store_nbr"].unique()))

#==============================Block3=========================

print("=" * 80)
print("PRODUCT FAMILIES")
print("=" * 80)

print("Number of Families:")
print(train["family"].nunique())

print("\nFamilies:")

for family in sorted(train["family"].unique()):
    print(family)

#==============================Block4=========================

print("=" * 80)
print("TIME SERIES")
print("=" * 80)

series = train.groupby(["store_nbr", "family"]).ngroups

print("Total Time Series:")
print(series)

#==============================Block5=========================

print("=" * 80)
print("DUPLICATES")
print("=" * 80)

duplicates = train.duplicated().sum()

print("Duplicate Rows:", duplicates)

#==============================Block6=========================

print("=" * 80)
print("SALES")
print("=" * 80)

print("Missing Sales:")
print(train["sales"].isnull().sum())

print("\nMinimum Sales:")
print(train["sales"].min())

print("\nMaximum Sales:")
print(train["sales"].max())

print("\nAverage Sales:")
print(train["sales"].mean())

