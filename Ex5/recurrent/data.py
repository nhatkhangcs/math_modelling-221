import pandas as pd
import torch
from config import args

def normalize_data(RJ):     # x - min / max - min -> to [0, 1] range
    max_RJ, _ = RJ.max(dim=0)
    min_RJ, _ = RJ.min(dim=0)

    RJ = (RJ - min_RJ) / (max_RJ - min_RJ)

    return RJ

def load_data():
    data = pd.read_csv(args.data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    # data = data.drop(['index'],axis=1)
    RJ = data.to_numpy().reshape((-1, 2))
    RJ = torch.from_numpy(RJ).float()
    # normalize_data(RJ)

    helper = lambda RJ, RJ_prev: RJ - RJ_prev
    delta_list = [helper(RJ[i], RJ[i - 1]) for i in range(1, RJ.shape[0])]

    delta_RJ = torch.stack(delta_list)
    # print(delta_RJ)
    # print(delta_RJ.max(dim=0))

    RJ.requires_grad = False
    delta_RJ.requires_grad = False

    return RJ, delta_RJ

