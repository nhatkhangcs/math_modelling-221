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
    model = Model(5, save_name=args.model_name)
    loss = MSE()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epochs + 1):
        model.train()
        Y_pred = model(delta_RJ, RJ)

        l = loss(RJ, Y_pred)
        l.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 1== 0:
            print(f'Epoch: {epoch}, train loss: {l:.7f}')



if __name__ == '__main__':
    main()