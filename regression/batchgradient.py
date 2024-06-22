import numpy as np

class BatchGradient:
    def __init__(self):
        pass
    def fit(self, X, y, learning_rate = 0.1, epochs = 1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        m = len(X)
        self.save_theta = None
        # self.theta = np.random.uniform(y.min(), y.max(), size=(X.shape[1], 1))
        theta = np.zeros((X.shape[1], 1))
        for _ in range(self.epochs):
            gradient = (2 / m) * X.T @ (X @ theta - y)
            theta -= self.learning_rate * gradient
        return theta
