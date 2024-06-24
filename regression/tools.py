import numpy as np

def DummyColumn(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))
def Scaled(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def LearningSchedule(t, t0, t1):
    return t0 / (t + t1)