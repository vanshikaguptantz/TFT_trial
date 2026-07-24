# from pathlib import Path

# import lightning.pytorch as pl
# from pytorch_forecasting import TemporalFusionTransformer
# from pytorch_forecasting.metrics import QuantileLoss

# # Import objects created in dataset.py
# from dataset import training


# # ============================================================
# # Random Seed
# # ============================================================

# pl.seed_everything(42)


# # ============================================================
# # Build Temporal Fusion Transformer
# # ============================================================

# tft = TemporalFusionTransformer.from_dataset(
#     training,

#     # -----------------------------
#     # Learning Parameters
#     # -----------------------------
#     learning_rate=0.001,

#     # -----------------------------
#     # Network Architecture
#     # -----------------------------
#     hidden_size=32,

#     lstm_layers=2,

#     attention_head_size=4,

#     hidden_continuous_size=16,

#     dropout=0.1,

#     # -----------------------------
#     # Output
#     # -----------------------------
#     output_size=7,

#     loss=QuantileLoss(),

#     # -----------------------------
#     # Logging
#     # -----------------------------
#     log_interval=10,

#     reduce_on_plateau_patience=4,
# )


# # ============================================================
# # Print Model Summary
# # ============================================================

# print("=" * 80)
# print("Temporal Fusion Transformer Created Successfully")
# print("=" * 80)

# print(tft)

# print("\nNumber of Parameters:")
# print(f"{tft.size()/1e3:.2f}K")



#============================================================= updated code for model.py with reusable get_model function=========================================


import lightning.pytorch as pl

from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss


# ============================================================
# Random Seed
# ============================================================

pl.seed_everything(42)


# ============================================================
# Build TFT Model
# ============================================================

def get_model(training,
    learning_rate=1e-3,
    hidden_size=32,
    lstm_layers=2,
    attention_head_size=4,
    hidden_continuous_size=16,
    dropout=0.1,):

    model = TemporalFusionTransformer.from_dataset(

        training,

        # ----------------------------------------------------
        # Learning Parameters
        # ----------------------------------------------------

        learning_rate=learning_rate,

        # ----------------------------------------------------
        # Network Architecture
        # ----------------------------------------------------

        hidden_size=hidden_size,

        lstm_layers=lstm_layers,

        attention_head_size=attention_head_size,

        hidden_continuous_size=hidden_continuous_size,

        dropout=dropout,

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_size=7,

        loss=QuantileLoss(),

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        log_interval=10,

        reduce_on_plateau_patience=4,
    )

    print("\n" + "=" * 80)
    print("Temporal Fusion Transformer Created Successfully")
    print("=" * 80)

    print(model)

    print("\nNumber of Parameters")
    print(f"{model.size()/1e3:.2f}K")

    return model


# ============================================================
# Test Model
# ============================================================

if __name__ == "__main__":

    from dataset import get_datasets

    training, _, _, _ = get_datasets()

    model = get_model(training)