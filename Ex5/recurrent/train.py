from xml.etree.ElementPath import prepare_descendant
import torch
import torch.nn as nn

from model import *
from config import *
from data import *
from utils import *
from loss import *

def train1():
    RJ, delta_RJ = load_data()
    # RJ = RJ[899:, :]
    # delta_RJ = delta_RJ[899:, :]

    # Train hyper params
    learning_rate = args.lr
    num_epochs = args.num_epochs

    # model setup
    model = Model3(args.hidden_size, sequence_len=args.sequence_len, save_name=args.model_name)
    loss = MSE()

    # DEBUG
    # loss_test = nn.MSELoss()
    # y_true = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    min_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model.train()
        model.reset_hidden_state()
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


    model.load(args.model_path + args.model_name)
    model.eval()
    model.reset_hidden_state()
    abcd = model.get_abcd(delta_RJ)
    print(abcd)

def train2():
    dataset = RJData()

    data_loader = DataLoader(dataset=dataset, batch_size=6, num_workers=2, drop_last=True)

    # Train hyper params
    learning_rate = args.lr
    num_epochs = args.num_epochs

    # model setup
    model = Model3(args.hidden_size, sequence_len=args.sequence_len, save_name=args.model_name)
    # loss = nn.MSELoss()
    loss = MSE()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train = RJ[:7, :]
    # true = RJ[[7], :]
    # print(true)

    min_loss = float('inf')
    model.load(args.model_path + args.model_name)

    RJ = next(iter(data_loader))
    print(RJ)
    predict_RJ = explicit_euler(RJ, [ 2.1601,  3.7439, 5.6771, -2.5837])
    print(predict_RJ)
    return

    for epoch in range(1, num_epochs + 1):
        total_epoch_loss = 0.
        for RJ in iter(data_loader):
            # train = RJ[:7, :]
            # # delta = delta_RJ(train)
            # true = RJ[[7], :]
            train = RJ
            # print(true)
            model.reset_hidden_state()
            Y_pred = model(train, train)

            l = loss(train, Y_pred)
            total_epoch_loss += l.item()
            l.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
        mean_epoch_loss = total_epoch_loss/len(iter(data_loader))
        if epoch % args.print_every == 0:
            print(f'Epoch: {epoch}, train loss: {mean_epoch_loss:.7f}')
            if mean_epoch_loss < min_loss:
                min_loss = mean_epoch_loss
                model.save()


    model.load(args.model_path + args.model_name)
    model.eval()
    for RJ in iter(data_loader):
        # train = RJ[:7, :]
        train = RJ
        delta = delta_RJ(train)
        model.reset_hidden_state()
        abcd = model.get_abcd(delta)
        print(abcd)


def view_model():
    _, delta_RJ = load_data()
    delta_RJ = delta_RJ[899:, :]

    model = Model2(args.hidden_size, sequence_len=args.sequence_len, save_name=args.model_name)
    model.load(args.model_path + args.model_name)

    model.eval()
    abcd = model.get_abcd(delta_RJ)
    print(abcd)

if __name__ == '__main__':
    train2()
    # view_model()