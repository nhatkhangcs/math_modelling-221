import torch
import torch.nn as nn

class MSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, Y_true, Y_pred):
        slice = range(0, Y_pred.shape[0] - 1)
        Y_true = Y_true[1:, :]
        Y_pred = Y_pred[slice, :]

        return self.criterion(Y_true, Y_pred)