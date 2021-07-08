import numpy as np

class PreceptronSimple:

    def __init__(self):
        self.W = []

    def train(self, X, D, epoch = 1):
        self.W = np.zeros(X.shape[1] + 1)

        for _ in range(epoch):
            i = 0
            while i < len(X):
                y = self.predict(X[i])
                if y != D[i]:
                    self.W[0]  += D[i] * 1
                    self.W[1:] += D[i] * X[i]
                else:
                    i += 1
        return self
    
    def predict(self, Xval):
        h = np.dot(Xval, self.W[1:]) + self.W[0]
        y = 1 if h >= 0 else -1 # funcion de activacion escalonada (sign)
        return y

