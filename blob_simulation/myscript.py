from blob_simulation.tools.tools import Problem
from blob_simulation.algorithm.Qlearning_v2 import Qlearning

import pickle
import numpy as np

with open('examplePickle', 'rb') as f:
    dat = pickle.load(f)

SIZE = 9
N_ACTIONS = 4


L = np.array(dat)[:SIZE, :SIZE]
dat = L.tolist()

p = Problem(dat)


q_table = {}

for i in range(-SIZE+1, SIZE):
    for ii in range(-SIZE+1, SIZE):
        for iii in range(-SIZE+1, SIZE):
            for iiii in range(-SIZE+1, SIZE):
                q_table[((i, ii), (iii, iiii))] = [
                    np.random.uniform(-5, 0) for i in range(N_ACTIONS)]

algo = Qlearning(p, q_table)
algo.training()
