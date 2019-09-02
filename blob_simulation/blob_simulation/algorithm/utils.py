import numpy as np


def euclidean_distance(p1, p2):
    d2 = np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])
    return np.sqrt(d2)
