import numpy as np
from .models_linear_reg import NormalEquation, BatchGradient, StochasticGradient
from .tools import DummyColumn, Scaled

class LinearRegression():
    def __init__(self, model: str='Normal', scale: bool = False, 
                 learning_schedule=False, epochs=50, eta=0.1, batch_size = 1):
        self.batch_size = batch_size
        self.scale = scale
        self.learning_schedule = learning_schedule
        self.epochs = epochs
        self.eta = eta
        if model == 'Normal':
            self.model = NormalEquation
        elif model == 'BatchGD':
            self.model = BatchGradient
        elif model == 'StochasticGD':
            self.model = StochasticGradient
        
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        self.X = Scaled(X) if self.scale == True else X
        self.X = DummyColumn(self.X)
        
        self.theta = self.model(self.X, y, learning_schedule=self.learning_schedule, eta=self.eta,
                                epochs=self.epochs, batch_size = self.batch_size)
    def prediction(self, X):
        self.X = Scaled(X) if self.scale == True else X
        X = DummyColumn(self.X)
        self.predict_value = X @ self.theta
        return self.predict_value
