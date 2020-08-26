import numpy as np


class Permutator(object):
    def __init__(self, N):
        self.perm = np.random.permutation(N)

        invdict = {p: i for i, p in enumerate(self.perm)}
        self.invperm = np.array([invdict[n] for n in range(N)])
