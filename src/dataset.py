# from pathlib import Path

# import pandas as pd
# from pytorch_forecasting import TimeSeriesDataSet

# # ============================================================
# # Paths
# # ============================================================

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"

# # ============================================================
# # Load Dataset
# # ============================================================

# master_df = pd.read_csv(
#     PROCESSED_DATA / "featured_dataset.csv"
# )

# master_df["date"] = pd.to_datetime(master_df["date"])

# print("=" * 80)
# print("Dataset Loaded")
# print("=" * 80)
# print(master_df.shape)

# # ============================================================
# # Sort Dataset
# # ============================================================

# master_df = master_df.sort_values(
#     ["store_nbr", "family", "time_idx"]
# ).reset_index(drop=True)

# # ============================================================
# # Convert Categorical Columns
# # ============================================================

# categorical_columns = [
#     "store_nbr",
#     "family",
#     "city",
#     "state",
#     "store_type",
#     "cluster",
# ]

# for col in categorical_columns:
#     master_df[col] = master_df[col].asstore_type(str)

# # ============================================================
# # Verify Data store_types
# # ============================================================

# print("\nCategorical Columns")

# print(
#     master_df[
#         categorical_columns
#     ].dstore_types
# )

# # ============================================================
# # Check Number of Time Series
# # ============================================================

# print("\n" + "=" * 80)
# print("Total Time Series")
# print("=" * 80)

# print(
#     master_df.groupby(
#         ["store_nbr", "family"]
#     ).ngroups
# )

# # ============================================================
# # Train / Validation Split
# # ============================================================

# training_cutoff = master_df["time_idx"].max() - 30

# print("\nTraining Cutoff :", training_cutoff)

# # ============================================================
# # Create Training Dataset
# # ============================================================

# training = TimeSeriesDataSet(
#     master_df[
#         master_df.time_idx <= training_cutoff
#     ],

#     time_idx="time_idx",

#     target="sales",

#     group_ids=[
#         "store_nbr",
#         "family",
#     ],

#     max_encoder_length=90,

#     max_prediction_length=30,

#     static_categoricals=[
#         "store_nbr",
#         "family",
#         "city",
#         "state",
#         "store_type",
#         "cluster",
#     ],

#     time_varying_known_reals=[
#         "time_idx",
#         "day",
#         "month",
#         "year",
#         "day_of_week",
#         "week_of_year",
#         "quarter",
#         "is_weekend",
#         "is_month_start",
#         "is_month_end",
#         "onpromotion",
#     ],

#     time_varying_unknown_reals=[
#         "sales",
#         "transactions",
#         "dcoilwtico",
#     ],

#     add_relative_time_idx=True,

#     add_target_scales=True,

#     add_encoder_length=True,
#     allow_missing_timesteps=True,
# )

# # ============================================================
# # Create Validation Dataset
# # ============================================================

# validation = TimeSeriesDataSet.from_dataset(
#     training,
#     master_df,
#     predict=True,
#     stop_randomization=True,
# )

# # ============================================================
# # Create DataLoaders
# # ============================================================

# batch_size = 64

# train_dataloader = training.to_dataloader(
#     train=True,
#     batch_size=batch_size,
#     num_workers=0,
# )

# val_dataloader = validation.to_dataloader(
#     train=False,
#     batch_size=batch_size,
#     num_workers=0,
# )

# # ============================================================
# # Summary
# # ============================================================

# print("\n" + "=" * 80)
# print("Dataset Creation Successful")
# print("=" * 80)

# print(f"Training Samples   : {len(training)}")
# print(f"Validation Samples : {len(validation)}")

# print("\nDataLoaders Created Successfully")
# print(f"Train Batches      : {len(train_dataloader)}")
# print(f"Validation Batches : {len(val_dataloader)}")

# print("\nReady for TFT Training!")



#================================== Updated Code with improved reusability and modularity ==================================

from pathlib import Path

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA = PROJECT_ROOT / "data" / "processed"


# ============================================================
# Dataset Loader Function
# ============================================================

def get_datasets(batch_size=64):

    # ========================================================
    # Load Dataset
    # ========================================================

    master_df = pd.read_csv(
        PROCESSED_DATA / "featured_dataset.csv"
    )

    # Rename reserved column name
    master_df.rename(
        columns={
            "type": "store_type"
        },
        inplace=True,
    )

    master_df["date"] = pd.to_datetime(master_df["date"])

    print("=" * 80)
    print("Dataset Loaded")
    print("=" * 80)
    print(master_df.shape)

    # ========================================================
    # Sort Dataset
    # ========================================================

    master_df = (
        master_df
        .sort_values(
            ["store_nbr", "family", "time_idx"]
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # Convert Categorical Columns
    # ========================================================

    categorical_columns = [
        "store_nbr",
        "family",
        "city",
        "state",
        "store_type",
        "cluster",
    ]

    for col in categorical_columns:
        master_df[col] = master_df[col].astype(str)

    # ========================================================
    # Verify Data Types
    # ========================================================

    print("\nCategorical Columns")

    print(
        master_df[
            categorical_columns
        ].dtypes
    )

    # ========================================================
    # Total Time Series
    # ========================================================

    print("\n" + "=" * 80)
    print("Total Time Series")
    print("=" * 80)

    print(
        master_df.groupby(
            ["store_nbr", "family"]
        ).ngroups
    )

    # ========================================================
    # Train Validation Split
    # ========================================================

    training_cutoff = (
        master_df["time_idx"].max() - 30
    )

    print("\nTraining Cutoff :", training_cutoff)

    # ========================================================
    # Training Dataset
    # ========================================================

    training = TimeSeriesDataSet(

        master_df[
            master_df.time_idx <= training_cutoff
        ],

        time_idx="time_idx",

        target="sales",

        group_ids=[
            "store_nbr",
            "family",
        ],

        max_encoder_length=90,

        max_prediction_length=30,

        static_categoricals=[
            "store_nbr",
            "family",
            "city",
            "state",
            "store_type",
            "cluster",
        ],

        time_varying_known_reals=[
            "time_idx",
            "day",
            "month",
            "year",
            "day_of_week",
            "week_of_year",
            "quarter",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "onpromotion",
        ],

        time_varying_unknown_reals=[
            "sales",
            "transactions",
            "dcoilwtico",
        ],

        add_relative_time_idx=True,

        add_target_scales=True,

        add_encoder_length=True,

        allow_missing_timesteps=True,
    )

    # ========================================================
    # Validation Dataset
    # ========================================================

    validation = TimeSeriesDataSet.from_dataset(

        training,

        master_df,

        predict=True,

        stop_randomization=True,
    )

    # ========================================================
    # DataLoaders
    # ========================================================

    train_dataloader = training.to_dataloader(
        train=True,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 80)
    print("Dataset Creation Successful")
    print("=" * 80)

    print(f"Training Samples   : {len(training)}")
    print(f"Validation Samples : {len(validation)}")

    print("\nDataLoaders Created Successfully")

    print(f"Train Batches      : {len(train_dataloader)}")
    print(f"Validation Batches : {len(val_dataloader)}")

    print("\nReady for TFT Training!")

    return (
        training,
        validation,
        train_dataloader,
        val_dataloader,
    )


# ============================================================
# Test Dataset File
# ============================================================

if __name__ == "__main__":

    get_datasets(batch_size=256)