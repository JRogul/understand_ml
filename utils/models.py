import numpy as np

class K_Nearest_Neighbors:
    def __init__(self, K):
        self.X = None
        self.y = None
        self.K = K

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
    def predict(self, X_test, y_test):
        distances = np.sum(np.power(self.X[:] - X_test, 2), axis=1)
        indices = np.argsort(distances)[:self.K]
        #majority vote
        pred = np.argmax(np.bincount(self.y[indices]))
        return pred, y_test