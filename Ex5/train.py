import numpy as np
import pandas as pd
from model import Net
from loss import Loss
from activation import Activation

def main():
    data = pd.read_csv('data/exact.csv')
    data = data.reset_index(drop=True, inplace=True)
    Y_true = data.to_numpy().transpose()

    x = np.linspace(0., 1., 1000)
    model = Net()
    model.train(x, Y_true, lr=3, num_epochs=10000)

if __name__ == '__main__':
    main()