import numpy as np
from .tools import LearningSchedule



def NormalEquation(X, y, **kwarg):

    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

def BatchGradient(X, y, eta=0.1, epochs=1000, **kwarg):
    m = len(X)
    # self.theta = np.random.uniform(y.min(), y.max(), size=(X.shape[1], 1))
    theta = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
        gradient = (2 / m) * X.T @ (X @ theta - y)
        theta -= eta * gradient
    return theta

def StochasticGradient(X, y, eta=0.1, epochs=50,learning_schedule=False, **kwarg):
    if learning_schedule:
        try:
            t0, t1 = (learning_schedule)
        except(TypeError, ValueError) as e:
            print(f'{e} \nlearning_schedule expect a tuple of (t0, t1)')
    m = len(X)
    theta = np.zeros((X.shape[1], 1))
    for epoch in range(epochs):
        for iteration in range(m):
            idx = np.random.randint(m)
            xi = X[idx : idx + 1]
            yi = y[idx : idx + 1]
            gradient = 2 * xi.T @ (xi @ theta - yi)
            eta = LearningSchedule(epoch * m + iteration, t0, t1) if learning_schedule else eta
            theta -= eta * gradient
    return theta





    