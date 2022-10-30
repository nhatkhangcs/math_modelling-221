import numpy as np
import pandas as pd
from model import Net
from loss import Loss
from activation import Activation
from visualize import plot_data_with_hypo

def main():
    data = pd.read_csv('data/exact.csv', index_col=0)
    data.reset_index(drop=True, inplace=True)
    # data = data.drop(['index'],axis=1)
    print(data)
    Y_true = data.to_numpy().reshape((-1, 2)).transpose()
    print(Y_true)
    print(data.size)

    x = np.linspace(0., 1., 1000)
    model = Net()
    model.train(x, Y_true, lr=0.01, num_epochs=20000)
    pred = model(x)
    plot_data_with_hypo(pred)

if __name__ == '__main__':
    main()