import numpy as np
from .tools import sigma

def binary_logisc_reg(X, y, epochs, eta):
    m = len(X)
    theta = np.zeros((X.shape[1], 1))
    for _ in range(epochs):
        loss = sigma(X @ theta) - y
        gradient = (2 / m) * X.T @ (loss)
        theta -= eta * gradient
    return theta
def softmax(X, y, epochs, eta):
    m = len(X)
    vector_score = np.empty((0, X.shape[1]))
    for idx in range(np.unique(y).shape[0]):
        theta = np.zeros((X.shape[1], 1))
        for _ in range(epochs):
            loss = sigma(X @ theta) - (y == idx)
            gradient = (2 / m) * X.T @ (loss)
            theta -= eta * gradient
        vector_score = np.vstack((vector_score, theta.ravel()))
    return vector_score
