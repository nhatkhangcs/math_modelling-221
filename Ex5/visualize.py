import imp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

from config import args
from model import *

torch.Tensor.ndim = property(lambda self: len(self.shape))  # Fix it

def plot_data():
    data = pd.read_csv('data/exact.csv')
    R_love = data['R']
    J_love = data['J']

    time_step = np.linspace(0., 0.999, 1000)

    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.title('Dynamic of love between Romeo and Juliet')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.savefig('figures/plot_data.png')

def plot_data_with_hypo(time_step, x, Y):
    data = pd.read_csv(args.data_path)
    R_love = data['R']
    J_love = data['J']


    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.plot(x[:, 0], Y[:, 0], 'c-', markersize=2, label='R Predict')
    plt.plot(x[:, 0], Y[:, 1], 'm-', markersize=2, label='J Predict')
    
    plt.title('Fitting plot')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.show()

def plot_model_fit():
    model_fit = NetFit(save_path='fit/')
    model_fit.load(args.fit_path + args.save_fit_name)
    model_fit.grad_off()
    model_fit.eval()

    t = torch.linspace(0., 0.999, 1000).unsqueeze(0).transpose(0, 1).float()
    z = torch.linspace(0., 0.999, 10000).unsqueeze(0).transpose(0, 1).float()
    o = model_fit(z)
    plot_data_with_hypo(t, z, o)

def plot_sol():
    data = pd.read_csv(args.data_path)
    R_love = data['R']
    J_love = data['J']

    time_step = np.linspace(0., 0.999, 1000)
    c1 = -0.28121
    m1_1 = 1.33105
    m1_2 = 1.
    v1 = np.exp((1/2) * (-0.4236 + np.sqrt(107.5216172)) * time_step)
    
    c2 = 3.28121
    m2_1 = -0.49545
    m2_2 = 1.
    v2 = np.exp((1/2) * (-0.4236  - np.sqrt(107.5216172)) * time_step)

    R_sol = c1 * m1_1 * v1 + c2 * m2_1 * v2
    J_sol = c1 * m1_2 * v1 + c2 * m2_2 * v2

    plt.plot(time_step, R_love, 'r.', markersize=2, label='Romeo')
    plt.plot(time_step, J_love, 'g.', markersize=2, label='Juliet')
    plt.plot(time_step, R_sol, 'c-', markersize=2, label='R sol')
    plt.plot(time_step, J_sol, 'm-', markersize=2, label='J sol')
    
    plt.title('Solution plot')
    plt.legend(loc="upper right")
    plt.xlabel('Time (s)')
    plt.ylabel('Love')
    plt.grid()
    plt.show()

def plot_sol_test():
    data = pd.read_csv(args.data_path)
    R_love = data['R']
    J_love = data['J']

    time_step = np.linspace(0., 0.999, 1000)
    c1 = -0.28121
    m1_1 = 1.33105
    m1_2 = 1.
    v1 = np.exp((1/2) * (-0.4236 + np.sqrt(107.52078)) * time_step)
    
    c2 = 3.28121
    m2_1 = -0.49545
    m2_2 = 1.
    v2 = np.exp((1/2) * (-0.4236  - np.sqrt(107.52078)) * time_step)

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

def abcd_evolution(epoch, a, b, c, d):
    plt.plot(epoch, a, 'r-', markersize=2, label='a')
    plt.plot(epoch, b, 'g-', markersize=2, label='b')
    plt.plot(epoch, c, 'b-', markersize=2, label='c')
    plt.plot(epoch, d, 'm-', markersize=2, label='d')
    plt.title('Values of a, b, c, d over epochs')
    plt.legend(loc="upper right")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid()
    plt.savefig(args.figure_path + args.abcd_figure_name)

def main():
    plot_data()

if __name__ == '__main__':
    main()
