import numpy as np
from .tools import LearningSchedule



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

def StochasticGradient(X, y, eta, epochs, learning_schedule,
                        batch_size, **kwarg):
    if learning_schedule:
        try:
            t0, t1 = learning_schedule  
        except (TypeError, ValueError) as e:  # Added space
            print(f'{e} \nlearning_schedule expect a tuple of (t0, t1)')
            return 
    m = X.shape[0]

    theta = np.zeros((X.shape[1], 1))
    
    for epoch in range(epochs):
        indices = np.random.permutation(m)  
        X_shuffled = X[indices]  
        y_shuffled = y[indices]  
        
        for iteration in range(0, m, batch_size):
            xi = X_shuffled[iteration:iteration+batch_size]  
            yi = y_shuffled[iteration:iteration+batch_size]  
            predict = xi @ theta
            gradient = 2 * xi.T @ (predict - yi)
            if learning_schedule:  
                eta = LearningSchedule(epoch * m + iteration, t0, t1)
            theta -= eta * gradient  
    
    return theta





    