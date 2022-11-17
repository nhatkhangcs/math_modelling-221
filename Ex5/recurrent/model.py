import torch
import torch.nn as nn
import os

from utils import explicit_euler

class Model(nn.Module):
    def __init__(self, hidden_size, sequence_len=999, save_path='model/', save_name='model.pt'):
        super().__init__()

        self.sequence_len = sequence_len
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(2, hidden_size, 2, batch_first=True)      # 2 layers of LSTM

        in_features = sequence_len * hidden_size
        self.fc = nn.Sequential(
            nn.Linear(in_features, 10),
            nn.Softplus(),
            nn.Linear(10, 10),
            nn.Softplus(),
            nn.Linear(10, 4)
        )

        self.save_path = save_path
        self.save_name = save_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)



    def forward(self, X, data):
        # h0 = torch.zeros(2, x.size(0), self.hidden_size)

        out, _ = self.lstm(X)

        flat = out.flatten()
        unsqueeze_out = flat.unsqueeze(0)

        abcd = self.fc(unsqueeze_out)
        y_pred = explicit_euler(data, abcd[0])

        return y_pred

    def save(self):
        torch.save(self.state_dict(), self.save_path + self.save_name)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print('Model loaded from \"' + load_path + '\"')

