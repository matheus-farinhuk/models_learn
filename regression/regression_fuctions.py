import numpy as np

def mean_squared_error(y_pred, y, squared = True):
    return 1/len(y) * np.sum((y_pred - y)**2) if squared else 1/len(y) * np.sum((y_pred - y))