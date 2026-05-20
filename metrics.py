import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def rounded(value, digits=4):
    return round(float(value), digits)
