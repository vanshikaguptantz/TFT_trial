import sys
import torch
import lightning
import pytorch_forecasting
import pandas as pd
import numpy as np

print("=" * 50)
print("Python Version:", sys.version)
print("Torch Version:", torch.__version__)
print("Lightning Version:", lightning.__version__)
print("PyTorch Forecasting:", pytorch_forecasting.__version__)
print("Pandas Version:", pd.__version__)
print("NumPy Version:", np.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("=" * 50)