# src: https://github.com/facebookresearch/DrQA/blob/master/drqa/reader/data.py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from c2nl.inputters.vector import vectorize


# ------------------------------------------------------------------------------
# PyTorch dataset class for SQuAD (and SQuAD-like) data.
# ------------------------------------------------------------------------------


class CommentDataset(Dataset):
    def __init__(self, examples, model):
        self.model = model
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return vectorize(self.examples[index], self.model)

    def lengths(self):
        total_tuple = []
        for ex in self.examples:
            tmp_rep_list = []
            if 'rep_0' in ex:
                tmp_rep_list.append(len(ex['rep_0'].tokens))
            if 'rep_1' in ex:
                tmp_rep_list.append(len(ex['rep_1'].tokens))
            if 'rep_2' in ex:
                tmp_rep_list.append(len(ex['rep_2'].tokens))
            if 'summary' in ex:
                tmp_rep_list.append(len(ex['summary'].tokens))
            tmp_rep_tuple = tuple(tmp_rep_list)
            total_tuple.append(tmp_rep_tuple)
        return total_tuple


# ------------------------------------------------------------------------------
# PyTorch sampler returning batched of sorted lengths (by doc and question).
# ------------------------------------------------------------------------------


class SortedBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]
        if self.shuffle:
            np.random.shuffle(batches)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return len(self.lengths)
