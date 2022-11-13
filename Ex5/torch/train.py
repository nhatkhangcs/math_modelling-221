import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model import NetFit, Net4
from loss import InitMSELoss
from config import args

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it

def plot_data_with_hypo(time_step, x, Y):
    data = pd.read_csv(args.data_path)
    R_love = data['R']
    J_love = data['J']


    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.plot(x[:, 0], Y[:, 0], 'c-', markersize=2, label='R Predict')
    plt.plot(x[:, 0], Y[:, 1], 'm-', markersize=2, label='J Predict')
    
    plt.title('Fitting plot :)')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.show()

def derivative(model, t):
    diff_x = torch.tensor(1e-4, dtype=torch.float32)
    diff_y = model(t + diff_x) - model(t - diff_x)
    return torch.div(diff_y, 2*diff_x)

def train_fit():
    data = pd.read_csv(args.data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    # data = data.drop(['index'],axis=1)
    Y_true = data.to_numpy().reshape((-1, 2))
    Y_true = torch.from_numpy(Y_true).float()
    x = torch.linspace(0., 1., 1000).unsqueeze(0).transpose(0, 1).float()

    random_idx = torch.randperm(1000)
    Y_true_perm = Y_true[random_idx, :]
    x_perm = x[random_idx, :]

    x_train, x_test = x_perm.split([800, 200], dim=0)
    Y_train, Y_test = Y_true_perm.split([800, 200], dim=0)
    
    # x_train, x_test = (x_perm, x_perm)
    # Y_train, Y_test = (Y_true_perm, Y_true_perm)

    # Train hyper params
    learning_rate = 0.01
    num_epochs = 30000

    model_fit = NetFit(save_path=args.fit_path, save_name=args.save_fit_name)
    loss = InitMSELoss(init_weight=8)
    optim = torch.optim.SGD(model_fit.parameters(), lr=learning_rate)

    min_loss = float('inf')
    init_x = torch.tensor([[0]], dtype=torch.float32)
    for epoch in range(1, num_epochs + 1):
        model_fit.train()
        Y_pred = model_fit(x_train)
        init_pred = model_fit(init_x)
        l = loss(Y_train, Y_pred, init_pred)
        l.backward()
        optim.step()
        optim.zero_grad()
        model_fit.eval()
        with torch.no_grad():
            Y_pred = model_fit(x_test)
            init_pred = model_fit(init_x)
            l_test = loss(Y_test, Y_pred, init_pred)
            if (l_test < min_loss):
                # print(f'Old val loss: {min_loss:.5f} -> New val loss: {l_test:.5f}. Model saved')
                min_loss = l_test
                model_fit.save()

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, train loss: {l:.7f}, test loss: {l_test:.7f}')

    z = torch.linspace(0., 1., 10000).unsqueeze(0).transpose(0, 1).float()
    with torch.no_grad():
        Y =  model_fit(z)
        print(model_fit(init_x))
    plot_data_with_hypo(x, z, Y)



def train_4():
    model_fit = NetFit()
    model_fit.load(args.fit_path + args.save_fit_name)
    model_fit.grad_off()
    model_fit.eval()

    t = torch.linspace(0., 1., 1000).unsqueeze(0).transpose(0, 1).float()
    # z = torch.linspace(0., 1., 10000).unsqueeze(0).transpose(0, 1).float()
    # o = model_fit(z)
    # plot_data_with_hypo(t, z, o)


    X = model_fit(t)
    Y_true = derivative(model_fit, t)
    random_idx = torch.randperm(1000)
    Y_true_perm = Y_true[random_idx, :]
    X_perm = X[random_idx, :]

    # X_train, X_test = X_perm.split([950, 50], dim=0)
    # Y_train, Y_test = Y_true_perm.split([950, 50], dim=0)

    X_train, X_test = (X_perm, X_perm)
    Y_train, Y_test = (Y_true_perm, Y_true_perm)

    # Train hyper params
    learning_rate = 0.001
    num_epochs = 20000
    
    model4 = Net4(save_path=args.abcd_path, save_name=args.save_abcd_name)
    loss = nn.MSELoss()
    optim = torch.optim.SGD(model4.parameters(), lr=learning_rate)

    min_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        model4.train()
        Y_pred = model4(X_train)
        l = loss(Y_train, Y_pred)
        l.backward()
        optim.step()
        optim.zero_grad()
        model4.eval()
        with torch.no_grad():
            Y_pred = model4(X_test)
            l_test = loss(Y_test, Y_pred)
            if (l_test < min_loss):
                # print(f'Old val loss: {min_loss:.5f} -> New val loss: {l_test:.5f}. Model saved')
                min_loss = l_test
                model4.save()

        if epoch % 500 == 0:
            print(f'Epoch: {epoch}, train loss: {l:.7f}, test loss: {l_test:.7f}')

def view_model():
    model_dict = torch.load(args.abcd_path + args.save_abcd_name)
    print(model_dict)

def plot_model_fit():
    model_fit = NetFit(save_path='fit/')
    model_fit.load(args.fit_path + args.save_fit_name)
    model_fit.grad_off()
    model_fit.eval()

    t = torch.linspace(0., 1., 1000).unsqueeze(0).transpose(0, 1).float()
    z = torch.linspace(0., 1., 10000).unsqueeze(0).transpose(0, 1).float()
    o = model_fit(z)
    plot_data_with_hypo(t, z, o)

def plot_sol():
    data = pd.read_csv(args.data_path)
    R_love = data['R']
    J_love = data['J']

    time_step = np.linspace(0., 1., 1000)
    c1 = -0.27609
    m1_1 = 1.33751
    m1_2 = 1.
    v1 = np.exp((1/2) * (-0.2323 + np.sqrt(103.06037)) * time_step)
    
    c2 = 3.27609
    m2_1 = -0.49776
    m2_2 = 1.
    v2 = np.exp((1/2) * (-0.2323 - np.sqrt(103.06037)) * time_step)

    R_sol = c1 * m1_1 * v1 + c2 * m2_1 * v2
    J_sol = c1 * m1_2 * v1 + c2 * m2_2 * v2

    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.plot(time_step, R_sol, 'c-', markersize=2, label='R sol')
    plt.plot(time_step, J_sol, 'm-', markersize=2, label='J sol')
    
    plt.title('Solution plot :)')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.show()

def plot_sol_test():
    data = pd.read_csv(args.data_path)
    R_love = data['R']
    J_love = data['J']

    time_step = np.linspace(0., 1., 1000)
    c1 = 1.90639
    m1_1 = 1.32752
    m1_2 = 1.
    v1 = np.exp((1/2) * (0.1326 + np.sqrt(94.4945268)) * time_step)
    
    c2 = 1.09360
    m2_1 = -0.48535
    m2_2 = 1.
    v2 = np.exp((1/2) * (0.1326  - np.sqrt(0.1326 )) * time_step)

    R_sol = c1 * m1_1 * v1 + c2 * m2_1 * v2
    J_sol = c1 * m1_2 * v1 + c2 * m2_2 * v2

    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.plot(time_step, R_sol, 'c-', markersize=2, label='R sol')
    plt.plot(time_step, J_sol, 'm-', markersize=2, label='J sol')
    
    plt.title('Solution plot :)')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    # train_fit()
    train_4()
    view_model()
    # plot_model_fit()
    # plot_sol()
    # plot_sol_test()