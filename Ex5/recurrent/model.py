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
            nn.Dropout(0.5),
            nn.Linear(in_features, 256),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Dropout(0.5),
            nn.Linear(128, 4)
        )

        self.save_path = save_path
        self.save_name = save_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)



    def forward(self, X, data):
        # h0 = torch.zeros(2, x.size(0), self.hidden_size)

        out, _ = self.lstm(X)

        out = out.flatten().unsqueeze(0)

        abcd = self.fc(out)
        self.abcd = abcd[0]
        y_pred = explicit_euler(data, self.abcd)

        return y_pred

    def get_abcd(self, del_RJ):
        out, _ = self.lstm(del_RJ)

        out = out.flatten().unsqueeze(0)

        abcd = self.fc(out)

        return abcd

    def save(self):
        torch.save(self.state_dict(), self.save_path + self.save_name)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print('Model loaded from \"' + load_path + '\"')

