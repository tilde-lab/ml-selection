import torch
from sklearn.metrics import explained_variance_score
from torcheval.metrics import R2Score
from torchmetrics import MeanAbsoluteError

from ml_selection.metrics.statistic_metrics import theils_u


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor) -> tuple:
    """Compute R2, MAE, EVS, Theil's U metrics"""
    mae, r2 = MeanAbsoluteError(), R2Score()

    mae.update(y_pred, y_true)
    r2.update(y_pred, y_true)

    try:
        mae_result = mae.compute()
    except:
        mae_result = None

    try:
        r2_res = r2.compute()
    except:
        r2_res = None

    try:
        evs = explained_variance_score(y_pred, y_true)
    except:
        evs = None

    try:
        theils_u_res = theils_u(y_pred, y_true)
    except:
        theils_u_res = None

    return (r2_res, mae_result, evs, theils_u_res)
