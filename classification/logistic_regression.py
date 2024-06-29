import numpy as np
from .logistic_regression_models import binary_logisc_reg, softmax
from regression.tools import DummyColumn, Scaled
from .tools import sigma


class LogisticRegression:
    def __init__(self, model='binary', epochs = 100, eta = 0.1, scale = False):
        self.scale = scale
        self.epochs = epochs
        self.eta = eta
        if model == 'binary':
            self.model = binary_logisc_reg
        else:
            self.model = softmax
    
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = Scaled(X) if self.scale == True else X
        self.X = DummyColumn(X)
        self.theta = self.model(self.X, y, epochs=self.epochs, eta=self.eta)
        return self.theta
    
    def predict(self, X):
        self.X = Scaled(X) if self.scale == True else X
        self.X = DummyColumn(X)
        self.predicted = sigma(self.X @ self.theta)
        return (0.5 < self.predicted)
        
