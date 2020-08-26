import numpy as np

from lib import permutator


def test_permutator():
    N = 14
    seed = 123

    np.random.seed(seed)
    perm = permutator.Permutator(N)
    print('perm.perm', perm.perm)
    print('perm.invperm', perm.invperm)

    orig = np.random.choice(10000, N, replace=False)
    print('orig', orig)
    p1 = orig[perm.perm]
    print('p1', p1)
    p2 = p1[perm.invperm]
    print('p2', p2)
    assert not (p2 == p1).all()
    assert (p2 == orig).all()
