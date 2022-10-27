import numpy as np
from loss import Loss
from activation import Activation

class Net():
    def __init__(self):
        # Network
        self.W = {}
        self.W['w1'] = (np.random.randn(5, 1)) * 0.01
        self.W['w2'] = (np.random.randn(5, 5)) * 0.01
        self.W['w3']= (np.random.randn(2, 5)) * 0.01
        self.B = {}
        self.B['b1'] = np.random.rand(5, 1) - 0.5
        self.B['b2'] = np.random.rand(5, 1) - 0.5
        self.B['b3'] = np.random.rand(2, 1) - 0.5
        self.G = {}
        self.G['g1'] = Activation('LReLU')
        self.G['g2'] = Activation('LReLU')
        # self.w1 = (np.random.randn(5, 30)) * 0.01
        # self.w2 = (np.random.randn(5, 5)) * 0.01
        # self.w3 = (np.random.randn(2, 5)) * 0.01
        # self.b1 = np.random.rand(5, 1) - 0.5
        # self.b2 = np.random.rand(5, 1) - 0.5
        # self.b3 = np.random.rand(2, 1) - 0.5

        # Activation and loss
        # self.sigmoid = Activation('sigmoid')
        self.lrelu = Activation('LReLU')
        self.loss = Loss('hmsq')

    def __call__(self, X, cache_res=False):
        return self.forward(X, cache_res)

    def forward(self, X, cache_res=False):
        # self.z1 = np.matmul(self.w1, X) + self.b1
        # self.a1 = self.lrelu(self.z1)
        # self.z2 = np.matmul(self.w2, self.a1) + self.b2
        # self.a2 = self.lrelu(self.z2)
        # self.z3 = np.matmul(self.w3, self.a2) + self.b3
        # y_pred = self.z3

        z1 = np.matmul(self.W['w1'], X) + self.B['b1']
        a1 = self.lrelu(z1)
        z2 = np.matmul(self.W['w2'], a1) + self.B['b2']
        a2 = self.lrelu(z2)
        z3 = np.matmul(self.W['w3'], a2) + self.B['b3']
        y_pred = z3

        if cache_res == False:
            return  y_pred
        else:
            return  { 
                        'pred': y_pred,
                        'z1': z1,
                        'z2': z2,
                        'z3': z3,
                        'a1': a1,
                        'a2': a2,
                    }
    
    def train(self, X, Y, lr=0.5, num_epochs=50):
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)
        m = X.shape[1]
        for epoch in range(1, num_epochs + 1):
            forw_cache = self.forward(X, cache_res=True)
            back = self.loss.backward(X, Y, self.W, self.G, forw_cache)

            self.W['w1'] -= np.multiply(lr/m, back['dw1'])
            self.W['w2'] -= np.multiply(lr/m, back['dw2'])
            self.W['w3'] -= np.multiply(lr/m, back['dw3'])
            self.W['b1'] -= np.multiply(lr/m, back['db1'])
            self.W['b2'] -= np.multiply(lr/m, back['db2'])
            self.W['b3'] -= np.multiply(lr/m, back['db3'])
    
            if epoch % 500 == 0:
                loss = self.loss(forw_cache['pred'], Y)
                print(f'Epoch {epoch}/{num_epochs}, loss: {loss.sum()}')

            if epoch == num_epochs:
                loss = self.loss(forw_cache['pred'], Y)
                print(f'Epoch {num_epochs}/{num_epochs}, loss: {loss.sum()}')
    
    
        