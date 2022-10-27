import numpy as np
# import sys

class Activation():
    def __init__(self, option):
        if option == 'sigmoid':
            self.activation = self.sigmoid
            self.gradient = self.d_sigmoid
        elif option == 'LReLU':
            self.activation = self.LReLU
            self.gradient = self.d_LReLU
        else:
            raise UserWarning('Invalid activation function option, program terminated!')

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def d_sigmoid(self, z):
        return np.exp(-z)/np.square(1 + np.exp(-z))

    def LReLU(self, z):
        return np.maximum(z, 0.01*z)
    
    def d_LReLU(self, z):
        z[z >= 0] = 1
        z[z < 0] = 0.01
        return z

    def grad(self, z):
        return self.gradient(z)
    
    def __call__(self, z):
        return self.activation(z)

def main():
    pass

if __name__ == '__main__':
    main()