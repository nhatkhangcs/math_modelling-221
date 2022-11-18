from ast import arg
import torch
import torch.nn as nn
from yaml import load

from model import *
from config import *
from data import *
from utils import *
from loss import *

def main():
    RJ, delta_RJ = load_data()

    # Train hyper params
    learning_rate = args.lr
    num_epochs = args.num_epochs

    # model setup
    model = Model(100, save_name=args.model_name)
    loss = MSE()

    # DEBUG
    # loss_test = nn.MSELoss()
    # y_true = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        Y_pred = model(delta_RJ, RJ)

        l = loss(RJ, Y_pred)
        l.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if epoch % args.print_every== 0:
            print(f'Epoch: {epoch}, train loss: {l:.7f}')
        
        if l < min_loss:
            min_loss = l
            model.save()

    model.eval()
    abcd = model.get_abcd(delta_RJ)
    print(abcd)

def view_model():
    _, delta_RJ = load_data()

    model = Model(5, save_name=args.model_name)
    model.load(args.model_path + args.model_name)

    model.eval()
    abcd = model.get_abcd(delta_RJ)
    print(abcd)

if __name__ == '__main__':
    main()
    # view_model()