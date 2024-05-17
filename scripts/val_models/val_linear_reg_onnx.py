"""
Run Linear Regression model in ONNX format
"""

import polars as pd
import torch
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError

from models.run_onnx import run_model_onnx

mae = MeanAbsoluteError()
r2 = R2Score()

total = pd.read_csv(
    "/root/projects/ml-selection/data/processed_data/under_str.csv",
)
Y = pd.read_csv(
    "/root/projects/ml-selection/data/processed_data/under_seeb.csv",
)

train_size = int(0.9 * len(total))
test_size = len(total) - train_size
atom = total["atom"][train_size:].values.tolist()
distance = total["distance"][train_size:].values.tolist()
seebeck = Y["Seebeck coefficient"][train_size:].values.tolist()

preds = run_model_onnx(
    "/root/projects/ml-selection/models/onnx/linear_regression_model.onnx",
    atom,
    distance,
)

r2.update(torch.tensor(preds), torch.tensor(seebeck))
r2_res = r2.compute()
mae.update(torch.tensor(preds), torch.tensor(seebeck))
mae_res = mae.compute()

print(f"R2: {r2_res}, MAE: {mae_res}")
