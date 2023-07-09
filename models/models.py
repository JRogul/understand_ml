import numpy as np

class K_Nearest_Neighbors:
    """
    Implements the K-Nearest Neighbors algorithm.
    
    K: int, number of nearest neighbors.
    """
    
    def __init__(self, K):
        self.X = None
        self.y = None
        self.K = K

    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        
    def predict(self, X_test, y_test=[]):

        if isinstance(X_test, list):
            X_test = np.array(X_test)
        elif not isinstance(X_test, np.ndarray):
            raise ValueError("X_test must be a list or a numpy array.")
            

        predictions = []
        for pred in range(len(X_test)):
            distances = np.sum(np.power(self.X[:] - X_test[pred], 2), axis=1)
            indices = np.argsort(distances)[:self.K]
            #majority vote
            predictions.append(np.argmax(np.bincount(self.y[indices])))
        if len(y_test) == 0:
            return predictions
        else:
            return predictions, y_test