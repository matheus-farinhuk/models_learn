import numpy as np
import pandas as pd

def DummyColumn(X):
    return np.c_[np.ones(len(X)), X]
def Scaled(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def LearningSchedule(t, t0, t1):
    return t0 / (t + t1)

def type_shape(*args):
    data = [np.array(arg).reshape(-1, 1) if len(np.array(arg).shape) <= 1 else np.array(arg) for arg in args]
    return data[0] if len(data) == 1 else data