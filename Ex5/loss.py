import numpy as np
# import sys

class Loss():
    def __init__(self, option):
        if option == 'bce':
            self.lossfnc = self.binary_cross_entropy
            # self.back = self.b_binary_cross_entropy
        elif option == 'hmsq':
            self.lossfnc = self.half_mean_square_loss
            self.back = self.b_half_mean_square_loss
        else:
            raise UserWarning('Invalid loss function option, program terminated!')
        

    def binary_cross_entropy(self, y_pred, y_true):
        return np.mean(-(np.multiply(y_true, np.log(y_pred)) + np.multiply(1 - y_true, np.log(1.0 - y_pred))))

    def half_mean_square_loss(self, y_pred, y_true):
        return np.multiply(np.mean(np.square(np.subtract(y_pred, y_true))), 1./2)

    def b_half_mean_square_loss(self, X, Y_true, W, G, cache):
        dz3 = cache['pred'] - Y_true
        dw3 = np.matmul(dz3, np.transpose(cache['a2']))
        db3 = np.sum(dz3, axis=1, keepdims=True)

        da2 = np.matmul(W['w3'].transpose(), dz3)
        dz2 = np.multiply(da2, d_Sigmoid(cache['z2']))
        dw2 = np.matmul(dz2, np.transpose(cache['a1']))
        db2 = np.sum(dz2, axis=1, keepdims=True)
        
        da1 = np.matmul(W['w2'].transpose(), dz2)
        dz1 = np.multiply(da1, d_Sigmoid(cache['z1']))
        dw1 = np.matmul(dz1, np.transpose(X))
        db1 = np.sum(dz1, axis=1, keepdims=True)

        return {
                    'dw1': dw1,
                    'dw2': dw2,
                    'dw3': dw3,
                    'db1': db1,
                    'db2': db2,
                    'db3': db3,
                }
    
    def backward(self, X, Y_true, W, cache_forward):
        return self.back(X, Y_true, W, cache_forward)

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

    loss = Loss('what')

if __name__ == '__main__':
    main()