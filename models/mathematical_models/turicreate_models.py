"""
Any of the below regression models can be used for predicting Seebeck Coefficient values of binary compounds.
"""

import pandas as pd
import torch
import turicreate as tc
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError
from turicreate import SFrame

mae = MeanAbsoluteError()
metric = R2Score()

# Crystal in vectors format
total = pd.read_csv(
    "/root/projects/ml-selection/data/processed_data/under_str.csv",
)
seebeck = pd.read_csv(
    "/root/projects/ml-selection/data/processed_data/under_seeb.csv",
)
total = pd.concat([seebeck["Seebeck coefficient"], total], axis=1)
features = ["atom", "distance"]

train_size = int(0.9 * len(total))
test_size = len(total) - train_size

train = total.iloc[:train_size]
test = total.iloc[train_size:]

# LINEAR REGRESSION MODEL
train_r, test_r = SFrame(train), SFrame(test)
model_linear = tc.linear_regression.create(
    train_r, target="Seebeck coefficient", features=features, validation_set=test_r
)
coefficients_linear = model_linear.coefficients
predictions_linear = model_linear.predict(test_r)

metric.update(
    torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
)
r2_res_r = metric.compute()
mae.update(
    torch.tensor(predictions_linear), torch.tensor(test_r["Seebeck coefficient"])
)
mae_result_r = mae.compute()

# DECISION TREE MODEL
train_d, test_d = SFrame(train), SFrame(test)
model_decision = tc.decision_tree_regression.create(
    train_d, target="Seebeck coefficient", features=features, validation_set=test_r
)
predictions_decision = model_decision.predict(test_d)

featureimp_decision = model_decision.get_feature_importance()
metric.update(
    torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
)
r2_res_d = metric.compute()
mae.update(
    torch.tensor(predictions_decision), torch.tensor(test_d["Seebeck coefficient"])
)
mae_result_d = mae.compute()

# BOOSTED TREES MODEL
train_b, test_b = SFrame(train), SFrame(test)
model_boosted = tc.boosted_trees_regression.create(
    train_b, target="Seebeck coefficient", features=features, validation_set=test_r
)
predictions_boosted = model_boosted.predict(test_b)
results_boosted = model_boosted.evaluate(test_b)
featureboosted = model_boosted.get_feature_importance()
metric.update(
    torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
)
r2_res_b = metric.compute()
mae.update(
    torch.tensor(predictions_boosted), torch.tensor(test_b["Seebeck coefficient"])
)
mae_result_b = mae.compute()

# RANDOM FOREST MODEL
train_r, test_r = SFrame(train), SFrame(test)
model_random = tc.random_forest_regression.create(
    train_r, target="Seebeck coefficient", features=features, validation_set=test_r
)
predictions_random = model_random.predict(test_r)
results_random = model_random.evaluate(test_r)
featureimp_random = model_random.get_feature_importance()
metric.update(
    torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
)
r2_res_rf = metric.compute()
mae.update(
    torch.tensor(predictions_random), torch.tensor(test_r["Seebeck coefficient"])
)
mae_result_rf = mae.compute()

print(
    f"LINEAR REGRESSION MODEL:\nR2: {r2_res_r}, MAE: {mae_result_r}, min pred: {predictions_linear.min()}, max pred: {predictions_linear.max()}\n\n"
)
print(
    f"DECISION TREE MODEL\nR2: {r2_res_d}, MAE: {mae_result_d}, min pred: {predictions_decision.min()}, max pred: {predictions_decision.max()}\n\n"
)
print(
    f"BOOSTED TREES MODEL\nR2: {r2_res_b}, MAE: {mae_result_b}, min pred: {predictions_boosted.min()}, max pred: {predictions_boosted.max()}\n\n"
)
print(
    f"RANDOM FOREST MODEL\nR2: {r2_res_rf}, MAE: {mae_result_rf}, min pred: {predictions_random.min()}, max pred: {predictions_random.max()}\n\n"
)
