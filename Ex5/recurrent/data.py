import pandas as pd
import torch
from config import args

def load_data():
    data = pd.read_csv(args.data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    # data = data.drop(['index'],axis=1)
    RJ = data.to_numpy().reshape((-1, 2))
    RJ = torch.from_numpy(RJ).float()

    helper = lambda RJ, RJ_prev: RJ - RJ_prev
    delta_list = [helper(RJ[i], RJ[i - 1]) for i in range(1, RJ.shape[0])]

    delta_RJ = torch.stack(delta_list)

    return RJ, delta_RJ

