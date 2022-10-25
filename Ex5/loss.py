import numpy as np
import sys

class Loss():
    def __init__(self, option):
        if option == 'bce':
            self.lossfnc = self.binary_cross_entropy
        elif option == 'hmsq':
            self.lossfnc = self.half_mean_square_loss
        else:
            sys.exit("Invalid loss function option, program terminated!")
        

    def binary_cross_entropy(self, y_pred, y_true):
        return np.mean(-(np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1.0 - y_pred))))

    def half_mean_square_loss(self, y_pred, y_true):
        return np.multiply(np.mean(np.square(np.subtract(y_pred, y_true))), 1./2)
    
    def __call__(self, y_pred, y_true):
        return self.lossfnc(y_pred, y_true)

def main():
    print('Testing half mean square loss:')
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1.2, 1.7, 3.5])
    loss = Loss('hmsq')
    print('Expected: 0.06333')
    print(f'Got: {loss(y_pred, y_true): .5f}\n')

    print('Testing binary cross entropy loss:')
    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.6, 0.5, 0.3])
    loss = Loss('bce')
    print('Expected: 0.80265')
    print(f'Got: {loss(y_pred, y_true): .5f}\n')

if __name__ == '__main__':
    main()