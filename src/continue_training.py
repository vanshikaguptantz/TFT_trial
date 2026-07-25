import os

os.environ["MPLBACKEND"] = "Agg"
from pathlib import Path

import lightning.pytorch as pl
import torch

from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)

from lightning.pytorch.loggers import (
    CSVLogger,
    TensorBoardLogger,
)

from dataset import get_datasets
from model import get_model

# ======================================
# Supress sklearn warning
# ======================================
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but StandardScaler was fitted with feature names"
)


# ============================================================
# Random Seed
# ============================================================

pl.seed_everything(42)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "outputs"

MODEL_DIR = OUTPUT_DIR / "models"

LOG_DIR = OUTPUT_DIR / "logs"

TB_LOG_DIR = LOG_DIR / "tensorboard"

CSV_LOG_DIR = LOG_DIR / "csv"

CHECKPOINT_PATH = MODEL_DIR / "best_model.ckpt"

PLOTS_DIR = OUTPUT_DIR / "plots"

METRICS_DIR = OUTPUT_DIR / "metrics"

PREDICTIONS_DIR = OUTPUT_DIR / "predictions"

# ============================================================
# Create Output Directories
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)

TB_LOG_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

METRICS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load Dataset
# ============================================================

training, validation, train_loader, val_loader = get_datasets()


# ============================================================
# Build Model
# ============================================================

print("=" * 80)
print("Building Model")
print("=" * 80)

model = get_model(training)


# ============================================================
# Logger
# ============================================================

tensorboard_logger = TensorBoardLogger(
    save_dir=TB_LOG_DIR,
    name="tft",
)

csv_logger = CSVLogger(
    save_dir=LOG_DIR,
    name="csv_logs",
)


# ============================================================
# Callbacks
# ============================================================

checkpoint_callback = ModelCheckpoint(
    dirpath=MODEL_DIR,
    filename="best_model",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    save_last=True,
    verbose=True,
)

lr_monitor = LearningRateMonitor(
    logging_interval="epoch"
)


# ============================================================
# Trainer
# ============================================================

trainer = pl.Trainer(

    max_epochs=20,

    accelerator="gpu" if torch.cuda.is_available() else "cpu",

    devices=1,

    logger=[tensorboard_logger, csv_logger],

    callbacks=[
        checkpoint_callback,
        lr_monitor,
    ],

    enable_progress_bar=True,

    log_every_n_steps=10,
)


# ============================================================
# Train Model
# ============================================================

print("\n" + "=" * 80)
print("Starting Training")
print("=" * 80)

trainer.fit(
    model,
    train_loader,
    val_loader,
    ckpt_path=CHECKPOINT_PATH,
)


# ============================================================
# Training Summary
# ============================================================

print("\n" + "=" * 80)
print("Training Complete")
print("=" * 80)

print(f"Best Model Path      : {checkpoint_callback.best_model_path}")

print(f"Best Validation Loss : {checkpoint_callback.best_model_score}")

print("=" * 80)