import numpy as np
import torch


def theils_u(y_true: torch.tensor, y_pred: torch.tensor):
    """Theil's U Statistic"""
    numerator = np.sqrt(np.mean((y_pred.cpu().numpy() - y_true.cpu().numpy()) ** 2))

    # naive prediction: in moment i -> y=i-1
    y_naive = np.roll(y_true.cpu().numpy(), 1)
    # copy first element
    y_naive[0] = y_true.cpu().numpy()[0]

    denominator = np.sqrt(np.mean((y_true.cpu().numpy() - y_naive) ** 2))

    theils_u_stat = numerator / denominator
    return theils_u_stat


if __name__ == "__main__":
    # test
    y_true = np.array([3, -0.5, 2, 7, 4.2])
    y_pred = np.array([2.5, 0.0, 2, 8, 4.5])
    u_stat = theils_u(y_true, y_pred)
    print(f"Theil's U Statistic: {u_stat}")