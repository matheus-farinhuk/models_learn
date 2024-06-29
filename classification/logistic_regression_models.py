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
def softmax():
    return
