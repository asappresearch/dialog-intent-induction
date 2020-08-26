import numpy as np
from collections import defaultdict

import samplers


def test_categories_sampler():
    N = 50
    n_batch = 5
    n_cls = 3
    n_ins = 4

    np.random.seed(123)
    labels = np.random.choice(5, N, replace=True)
    print('labels', labels)

    sampler = samplers.CategoriesSampler(labels, n_batch, n_cls, n_ins)
    for b, batch in enumerate(sampler):
        print('batch', batch)
        classes = []
        count_by_class = defaultdict(int)
        for i in batch:
            classes.append(labels[i])
            count_by_class[labels[i]] += 1
        print('classes', classes)
        print('count_by_class', count_by_class)
        assert len(count_by_class) == n_cls
        for v in count_by_class.values():
            assert v == n_ins
    assert b == n_batch - 1
