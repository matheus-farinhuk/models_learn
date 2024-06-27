import numpy as np
from .tools import LearningSchedule, mean_squared_error



def NormalEquation(X, y, **kwarg):

    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

def BatchGradient(X, y, eta, epochs, **kwarg):
    m = len(X)
    theta = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
        gradient = (2 / m) * X.T @ (X @ theta - y)
        theta -= eta * gradient
    return theta

def StochasticGradient(X, y, eta, epochs, learning_schedule, tol, early_stop, iteration_no_change,
                        **kwarg):
    if learning_schedule:
        try:
            t0, t1 = (learning_schedule)
        except(TypeError, ValueError) as e:
            print(f'{e} \nlearning_schedule expect a tuple of (t0, t1)')
    m = len(X)
    iteration_no_change_value = 0
    best_loss = float('inf')
    last_loss = float('inf')
    theta = np.zeros((X.shape[1], 1))
    for epoch in range(epochs):
        for iteration in range(m):
            idx = np.random.randint(m)
            xi = X[idx : idx + 1]
            yi = y[idx : idx + 1]
            predict = xi @ theta
            loss = mean_squared_error(predict, y, squared=False)
            gradient = 2 * xi.T @ (predict - yi)
            eta = LearningSchedule(epoch * m + iteration, t0, t1) if learning_schedule else eta
            theta -= eta * gradient
            if loss < best_loss:
                best_loss = loss
            if early_stop and last_loss == loss:
                iteration_no_change_value += 1
                print(iteration_no_change_value)
                if iteration_no_change_value >= iteration_no_change and loss > best_loss - tol:
                    return theta
            else:
                iteration_no_change_value = 0
            last_loss = loss
    return theta





    