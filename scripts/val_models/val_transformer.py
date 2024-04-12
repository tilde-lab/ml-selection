"""
Test Transformer model
"""
import pandas as pd
import torch
import torch.utils.data as data
from models.neural_network_models.transformer.transformer_reg import TransformerModel


total = pd.read_csv(
    r"/root/projects/ml-selection/data/processed_data/under_str.csv",
)
seebeck = pd.read_csv(
    r"/root/projects/ml-selection/data/processed_data/under_seeb.csv",
)
dataset = pd.concat([seebeck["Seebeck coefficient"], total], axis=1).values.tolist()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

model = TransformerModel(n_feature=2, heads=2)

model.load_state_dict(
    torch.load(
        r"/root/projects/ml-selection/models/neural_network_models/transformer/weights/01.pth"
    )
)

model.val(model, test_data)
