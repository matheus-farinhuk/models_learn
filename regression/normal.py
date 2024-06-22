import numpy as np

class NormalEquation:
    def __init__(self):
        pass
    def fit(self, X, y):

        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return theta
        
    
    
    