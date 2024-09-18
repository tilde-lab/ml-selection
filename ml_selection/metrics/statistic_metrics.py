import numpy as np
import torch


def theils_u(y_true: torch.tensor, y_pred: torch.tensor):
    """
    Theil's U Statistic.
    If Theil's U < 1, model is better than naiv pred,
    If Theil's U == 1, models acc like naiv pred,
    If Theil's U > 1, model is worse than naiv pred
    """
    if type(y_true) == torch.Tensor:
        y_true, y_pred = np.array(y_true), np.array(y_pred)

    numerator = np.sqrt(np.mean((y_pred - y_true) ** 2))

    # naive prediction: in moment i -> y=i-1
    y_naive = np.roll(y_true, 1)
    # copy first element
    y_naive[0] = y_true[0]

    denominator = np.sqrt(np.mean((y_true - y_naive) ** 2))

    theils_u_stat = numerator / denominator
    return theils_u_stat
