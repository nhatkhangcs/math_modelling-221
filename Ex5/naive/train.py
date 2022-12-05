import torch
import torch.nn as nn

from model import *
from loss import *
from config import *
from data_utils import *
from visualize import *
from utils import *

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it

def train_fit():
    x, Y_true = load_data()

    # shuffle data
    random_idx = torch.randperm(args.num_samples)
    Y_true_perm = Y_true[random_idx, :]
    x_perm = x[random_idx, :]

    # split data
    num_train_samples = int(args.fit_train_portion * args.num_samples)
    num_test_samples = args.num_samples - num_train_samples
    x_train, x_test = x_perm.split([num_train_samples, num_test_samples], dim=0)
    Y_train, Y_test = Y_true_perm.split([num_train_samples, num_test_samples], dim=0)

    # Train hyper params
    learning_rate = args.fit_lr
    num_epochs = args.fit_num_epochs

    # model setup
    model_fit = NetFit(save_path=args.fit_path, save_name=args.save_fit_name)
    loss = InitMSE(init_weight=7)
    optim = torch.optim.Adam(model_fit.parameters(), lr=learning_rate)

    min_loss = float('inf')
    init_x = torch.tensor([[0.]], dtype=torch.float32)
    epoch_lst = []
    train_loss_lst = []
    test_loss_lst = []
    for epoch in range(1, num_epochs + 1):
        # train
        model_fit.train()
        Y_pred = model_fit(x_train)
        init_pred = model_fit(init_x)

        # backpropagation
        l = loss(Y_train, Y_pred, init_pred)
        l.backward()
        optim.step()
        optim.zero_grad()

        # evalutation
        model_fit.eval()
        with torch.no_grad():
            Y_pred = model_fit(x_test)
            init_pred = model_fit(init_x)
            l_test = loss(Y_test, Y_pred, init_pred)
            if (l_test < min_loss):
                # print(f'Old val loss: {min_loss:.5f} -> New val loss: {l_test:.5f}. Model saved')
                min_loss = l_test
                model_fit.save()
        # epoch_lst.append(epoch)
        # train_loss_lst.append(l.detach().numpy())
        # test_loss_lst.append(l_test.detach().numpy())

        if epoch % 1000 == 0:
            print(f'Epoch: {epoch}, train loss: {l:.7f}, test loss: {l_test:.7f}')
            epoch_lst.append(epoch)
            train_loss_lst.append(l.detach().numpy())
            test_loss_lst.append(l_test.detach().numpy())

    z = torch.linspace(0., 0.999, 10000).unsqueeze(0).transpose(0, 1).float()
    with torch.no_grad():
        Y =  model_fit(z)
        init_pred = model_fit(init_x)
        print(f'R0 = {init_pred[0][0]: .4f}, J0 = {init_pred[0][1]: .4f}')
    plot_data_with_hypo(x, z, Y)
    save_learning_curve(epoch_lst, train_loss_lst, test_loss_lst, 'fit_l_curve.png')



def train_4():
    # load fit model
    model_fit = NetFit()
    model_fit.load(args.fit_path + args.load_fit_name)
    model_fit.grad_off()
    model_fit.eval()

    t = torch.linspace(0., 0.999, 1000).unsqueeze(0).transpose(0, 1).float()

    # get data
    X = model_fit(t)
    Y_true = derivative(model_fit, t)
    
    # shuffle data
    random_idx = torch.randperm(args.num_samples)
    Y_true_perm = Y_true[random_idx, :]
    X_perm = X[random_idx, :]

    # split data
    num_train_samples = int(args.abcd_train_portion * args.num_samples)
    num_test_samples = args.num_samples - num_train_samples
    X_train, X_test = X_perm.split([num_train_samples, num_test_samples], dim=0)
    Y_train, Y_test = Y_true_perm.split([num_train_samples, num_test_samples], dim=0)

    # Train hyper params
    learning_rate = args.abcd_lr
    num_epochs = args.abcd_num_epochs
    
    # model setup
    model4 = Net4(save_path=args.abcd_path, save_name=args.save_abcd_name)
    loss = nn.MSELoss()
    optim = torch.optim.Adam(model4.parameters(), lr=learning_rate)

    min_loss = float('inf')
    a_list = []
    b_list = []
    c_list = []
    d_list = []
    epoch_marks = []
    epoch_lst = []
    train_loss_lst = []
    test_loss_lst = []
    for epoch in range(1, num_epochs + 1):
        # train
        model4.train()
        Y_pred = model4(X_train)

        # backpropagation
        l = loss(Y_train, Y_pred)
        l.backward()
        optim.step()
        optim.zero_grad()

        # evaluation
        model4.eval()
        with torch.no_grad():
            Y_pred = model4(X_test)
            l_test = loss(Y_test, Y_pred)
            if (l_test < min_loss):
                # print(f'Old val loss: {min_loss:.5f} -> New val loss: {l_test:.5f}. Model saved')
                min_loss = l_test
                model4.save()
        with torch.no_grad():
            param_list = list(model4.parameters())
            a = param_list[0][0][0].detach().numpy()
            b = param_list[0][0][1].detach().numpy()
            c = param_list[0][1][0].detach().numpy()
            d = param_list[0][1][1].detach().numpy()
            a_list.append(np.array(a))
            b_list.append(np.array(b))
            c_list.append(np.array(c))
            d_list.append(np.array(d))
            

            epoch_marks.append(epoch)

        # epoch_lst.append(epoch)
        # train_loss_lst.append(l.detach().numpy())
        # test_loss_lst.append(l_test.detach().numpy())

        if epoch % 500 == 0:
            print(f'Epoch: {epoch}, train loss: {l:.7f}, test loss: {l_test:.7f}')
            epoch_lst.append(epoch)
            train_loss_lst.append(l.detach().numpy())
            test_loss_lst.append(l_test.detach().numpy())
            

    abcd_evolution(epoch_marks, a_list, b_list, c_list, d_list)
    save_learning_curve(epoch_lst, train_loss_lst, test_loss_lst, 'abcd_l_curve.png')
    view_model4()

def solution_evaluation():
    time_step, Y_true = load_data()

    c1 = -0.28121
    m1_1 = 1.33105
    m1_2 = 1.
    v1 = torch.exp((1/2) * (-0.4236 + torch.sqrt(torch.tensor(107.5216172))) * time_step)
    
    c2 = 3.28121
    m2_1 = -0.49545
    m2_2 = 1.
    v2 = torch.exp((1/2) * (-0.4236  - torch.sqrt(torch.tensor(107.5216172))) * time_step)

    R_sol = c1 * m1_1 * v1 + c2 * m2_1 * v2
    J_sol = c1 * m1_2 * v1 + c2 * m2_2 * v2
    
    solution = torch.cat([R_sol, J_sol], dim=1)
    
    loss_crit = nn.MSELoss()
    loss = loss_crit(solution, Y_true)
    print(f'Loss of solution is: {loss.item(): .5f}')

if __name__ == '__main__':
    train_fit()
    train_4()
    # view_model4()
    # plot_model_fit()
    # plot_sol()
    # plot_sol_test()
    # solution_evaluation()