import numpy as np

class NormalEquation:
    
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = self.dummy_column(X)
        self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ y
        
        
    def prediction(self, X):
        X = self.dummy_column(X)
        self.predict_value = X @ self.theta
        return self.predict_value
    def dummy_column(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    
    