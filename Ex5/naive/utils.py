import torch

from config import *

def derivative(model, t):
    diff_x = torch.tensor(1e-4, dtype=torch.float32)
    diff_y = model(t + diff_x) - model(t - diff_x)

    return torch.div(diff_y, 2*diff_x)

def view_model4():
    model_dict = torch.load(args.abcd_path + args.view_abcd_name)
    a = model_dict['linear.weight'][0][0]
    b = model_dict['linear.weight'][0][1]
    c = model_dict['linear.weight'][1][0]
    d = model_dict['linear.weight'][1][1]
    print(f'a = {a: .5f}, b = {b: .5f}, c = {c: .5f}, d = {d: .5f}')