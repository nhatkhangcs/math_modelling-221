import torch
import torch.nn as nn

class InitMSE(nn.Module):
    def __init__(self, R0=-2, J0=3, init_weight=1):
        super().__init__()
        self.init_values = torch.tensor([[R0, J0]], dtype=torch.float32)
        self.init_weight = init_weight
        self.crit = nn.MSELoss()

    def forward(self, Y_true, Y_pred, init_pred):
        return self.init_weight * self.crit(self.init_values, init_pred) + self.crit(Y_true, Y_pred)

