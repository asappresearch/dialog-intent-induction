import csv
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
np.random.seed(0)

PAD = "__PAD__"
UNK = "__UNK__"
START = "__START__"
END = "__END__"

class Dataset(Dataset):

    def __init__(self, fname, view1_col='view1_col', view2_col='view2_col', label_col='cluster_id', tokenized=True, max_sent=10, train_ratio=.9):
        """
        Args:
            fname: str, training data file
            view1_col: str, the column corresponding to view 1 input
            view2_col: str, the column corresponding to view 2 input
            label_col: str, the column corresponding to label
        """
        id_to_token = [PAD, UNK, START, END]
        token_to_id = {PAD: 0, UNK: 1, START: 2, END: 3}
        id_to_label = [UNK]
        label_to_id = {UNK: 0}
        data = []
        labels = []
        v1_utts = []  # needed for displaying cluster samples
        self.trn_idx, self.tst_idx = [], []
        self.trn_idx_no_unk = []
        with open(fname, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                view1_text, view2_text = row[view1_col], row[view2_col]
                label = row[label_col]
                if 'UNK' == label: label = UNK
                if '<cust_' not in view1_text:
                    view2_sents = sent_tokenize(view2_text.lower())
                else:
                    view2_sents = view2_text.split("> <")
                    for i in range(len(view2_sents) - 1):
                         view2_sents[i] = view2_sents[i] + '>'
                         view2_sents[i+1] = '<' + view2_sents[i]
                v1_utts.append(view1_text)
                if not tokenized:
                    v1_tokens = word_tokenize(view1_text.lower())
                    v2_tokens = [word_tokenize(sent.lower()) for sent in view2_sents]
                else:
                    v1_tokens = view1_text.lower().split()
                    v2_tokens = [sent.lower().split() for sent in view2_sents]
                v2_tokens = v2_tokens[:max_sent]

                def tokens_to_idices(tokens):
                    token_idices = []
                    for token in tokens:
                        if token not in token_to_id:
                            token_to_id[token] = len(token_to_id)
                            id_to_token.append(token)
                        token_idices.append(token_to_id[token])
                    return token_idices
                v1_token_idices = tokens_to_idices(v1_tokens)
                v2_token_idices = [tokens_to_idices(tokens) for tokens in v2_tokens]
                v2_token_idices = [idices for idices in v2_token_idices if len(idices) > 0]
                if len(v1_token_idices) == 0 or len(v2_token_idices) == 0:
                    continue
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                    id_to_label.append(label)
                data.append((v1_token_idices, v2_token_idices))
                labels.append(label_to_id[label])
                if label == UNK and np.random.random_sample() < .1:
                    self.tst_idx.append(len(data)-1)
                else:
                    self.trn_idx.append(len(data)-1)
                    if label != UNK:
                        self.trn_idx_no_unk.append(len(data) - 1)

        self.v1_utts = v1_utts
        self.id_to_token = id_to_token
        self.token_to_id = token_to_id
        self.id_to_label = id_to_label
        self.label_to_id = label_to_id
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i], self.labels[i]
