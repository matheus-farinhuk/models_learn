import numpy as np
import inspect
from .normal import NormalEquation
from .batchgradient import BatchGradient

class LinearRegression():
    def __init__(self, model='Normal', scale = False):
        self.scale = scale
        if model == 'Normal':
            self.model = NormalEquation()
        elif model == 'BatchGD':
            self.model = BatchGradient()
    def fit(self, X, y, **kwargs):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = self.Scaled(X) if self.scale == True else X
        self.X = self.dummy_column(self.X)

        model_fit_params = inspect.signature(self.model.fit).parameters
        unexpected_params = [k for k in kwargs if k not in model_fit_params]
        if unexpected_params:
            raise ValueError(f"Unexpected parameters: {unexpected_params}")
        
        self.theta = self.model.fit(self.X, y, **kwargs)
    def prediction(self, X):
        X = self.dummy_column(X)
        self.predict_value = X @ self.theta
        return self.predict_value
    def dummy_column(self, X):
        return np.hstack((np.ones((X.shape[0], 1)), X))
    def Scaled(self, X):
        return (X - np.mean(X, axis=0)) / np.std(X, axis=0)