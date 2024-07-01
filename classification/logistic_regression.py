import numpy as np
from .logistic_regression_models import binary_logisc_reg, softmax
from regression.tools import DummyColumn, Scaled
from .tools import sigma


class LogisticRegression:
    def __init__(self, epochs = 100, eta = 0.1, scale = False):
        self.scale = scale
        self.epochs = epochs
        self.eta = eta

    
    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if np.unique(y).shape[0] > 2:
            self.model = softmax
        else:
            self.model = binary_logisc_reg
        X = Scaled(X) if self.scale == True else X
        X = DummyColumn(X)
        self.theta = self.model(X, y, epochs=self.epochs, eta=self.eta)
        return self.theta
    
    def predict(self, X):
        X = Scaled(X) if self.scale == True else X
        X = DummyColumn(X)
        if self.model == binary_logisc_reg:
            self.probability_vector = sigma(X @ self.theta)
            return (0.5 < self.probability_vector)
        elif self.model == softmax:
            self.probability_vector = (np.exp(X @ self.theta.T) / np.exp(X @ self.theta.T).sum(axis=1, keepdims=True))
            return np.argmax(self.probability_vector, axis=1, keepdims=True)
    def predict_proba(self, X):
        X = Scaled(X) if self.scale == True else X
        X = DummyColumn(X)
        if self.model == binary_logisc_reg:
            self.pprobability_vector = sigma(X @ self.theta)
            return self.probability_vector
        elif self.model == softmax:
            self.probability_vector = (np.exp(X @ self.theta.T) / np.exp(X @ self.theta.T).sum(axis=1, keepdims=True))
            return self.probability_vector
