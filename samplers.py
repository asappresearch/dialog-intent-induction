import numpy as np
np.random.seed(0)

class CategoriesSampler():

    def __init__(self, labels, n_batch, n_cls, n_ins):
        """
        Args:
            labels: size=(dataset_size), label indices of instances in a data set
            n_batch: int, number of batchs for episode training
            n_cls: int, number of sampled classes
            n_ins: int, number of instances considered for a class

        conceptually, this is for prototypical sampling:
        - for each training step, we sample 'n_ways' classes (in the paper), which is 'n_cls' here
        - we draw 'n_shot' examples of each class, ie n_ins here
            - these will be encoded, and averaged, to get the prototypes
        - and we draw 'n_query' query examples, of each class, which is also 'n_ins' here
            - these will be encoded, and then used to generate the prototype loss, relative to the prototypes

        __iter__ returns a generator, which will yield n_batch sets of training data, one set of training
        data per yield command
        """
        if not isinstance(labels, list):
            labels = labels.tolist()

        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_ins = n_ins

        self.classes = list(set(labels))
        labels = np.array(labels)
        self.cls_indices = {}
        for c in self.classes:
            indices = np.argwhere(labels == c).reshape(-1)
            self.cls_indices[c] = indices

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = np.random.permutation(self.classes)[:self.n_cls]
            for c in classes:
                indices = self.cls_indices[c]
                while len(indices) < self.n_ins:
                    indices = np.concatenate((indices, indices))
                batch.append(np.random.permutation(indices)[:self.n_ins])
            batch = np.stack(batch).flatten('F')
            yield batch

