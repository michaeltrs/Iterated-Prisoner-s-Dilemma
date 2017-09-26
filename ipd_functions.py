import numpy as np

h = 1e-7

def round_choice(x):
    if x>=0.5:
        return 1
    else:
        return 0


def normalize(X):
    return X/np.sum(X)