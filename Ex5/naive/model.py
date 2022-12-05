import torch
import torch.nn as nn
import os

class NetFit(nn.Module):
    def __init__(self, save_path='./', save_name='net_fit69.pt'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 6),
            nn.Softplus(),
            nn.Linear(6, 6),
            nn.Softplus(),
            nn.Linear(6, 2),
        )
        # self.net = nn.Sequential(
        #     nn.Linear(1, 6),
        #     nn.Sigmoid(),
        #     nn.Linear(6, 6),
        #     nn.Sigmoid(),
        #     nn.Linear(6, 2),
        # )
        # self.net = nn.Sequential(
        #     nn.Linear(1, 6),
        #     nn.Tanh(),
        #     nn.Linear(6, 6),
        #     nn.Tanh(),
        #     nn.Linear(6, 2),
        # )
        # self.net = nn.Sequential(
        #     nn.Linear(1, 6),
        #     nn.LeakyReLU(),
        #     nn.Linear(6, 6),
        #     nn.LeakyReLU(),
        #     nn.Linear(6, 2),
        # )
        self.save_path = save_path
        self.save_name = save_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def forward(self, x):
        return self.net(x)

    def save(self):
        torch.save(self.state_dict(), self.save_path + self.save_name)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
        print('Model loaded from \"' + load_path + '\"')

    def grad_off(self):
        for p in self.parameters(): 
            p.requires_grad = False

class Net4(nn.Module):
    def __init__(self, save_path='./', save_name='net4_69.pt'):
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

        self.save_path = save_path
        self.save_name = save_name
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def forward(self, x):
        return self.linear(x)

    def save(self):
        torch.save(self.state_dict(), self.save_path + self.save_name)

    def load(self, load_path):
        self.load_state_dict(torch.load(load_path))
