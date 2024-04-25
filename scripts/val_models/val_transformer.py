"""
Test Transformer model
"""
import pandas as pd
import torch
import torch.utils.data as data

from models.neural_network_models.transformer.transformer_reg import \
    TransformerModel

poly = pd.read_csv(
    f"/root/projects/ml-selection/data/processed_data/large_poly_descriptor.csv",
)
seebeck = pd.read_json(
    "/root/projects/ml-selection/data/raw_data/median_seebeck.json", orient='split',
)
dataset = pd.merge(seebeck, poly, on="phase_id", how="inner").drop(columns=['phase_id', 'Formula']).values.tolist()

train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
test_data = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))

model = TransformerModel(n_feature=2, heads=2)

model.load_state_dict(
    torch.load(
        r"/root/projects/ml-selection/models/neural_network_models/transformer/weights/20_01.pth"
    )
)

model.val(model, test_data)
