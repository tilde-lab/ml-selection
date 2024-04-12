"""
Linear regression model predicts the value of Seebeck coefficient.
"""
import numpy as np
import pandas as pd
import torch
from sklearn import linear_model
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError

mae = MeanAbsoluteError()
r2 = R2Score()

# Crystal in vectors format
total = pd.read_csv(
    "/root/projects/ml-selection/data/processed_data/under_str.csv",
)
seebeck = pd.read_csv(
    "/root/projects/ml-selection/data/processed_data/under_seeb.csv",
)
total_transformed = []
seebeck_transformed = [i[0] for i in seebeck.values.tolist()]

# Data preparing
for i, row in enumerate(total.values.tolist()):
    atoms = eval(row[0])
    distance = eval(row[1])
    total_transformed.append([l for l in atoms])
    [total_transformed[i].append(k) for k in distance]

train_size = int(0.9 * len(total))
test_size = len(total) - train_size

train_y = np.array(seebeck_transformed[:train_size])
test_y = np.array(seebeck_transformed[train_size:])

train_x = np.array(total_transformed[:train_size])
test_x = np.array(total_transformed[train_size:])

# Create linear regression
regr = linear_model.LinearRegression()

regr.fit(train_x, train_y)

# Make predictions using the testing set
pred = regr.predict(test_x)

r2.update(torch.tensor(pred), torch.tensor(test_y))
r2_res = r2.compute()
mae.update(torch.tensor(pred), torch.tensor(test_y))
mae = mae.compute()

print(f"MAE: {mae}, R2: {r2_res}, min pred: {pred.min()}, max pred {pred.max()}")
