import torch
import pandas as pd
from config import args

def load_data():
    data = pd.read_csv(args.data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    # data = data.drop(['index'],axis=1)
    Y_true = data.to_numpy().reshape((-1, 2))
    Y_true = torch.from_numpy(Y_true).float()
    x = torch.linspace(0., 0.999, 1000).unsqueeze(0).transpose(0, 1).float()

    return x, Y_true