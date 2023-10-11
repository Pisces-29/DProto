import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    def __init__(self, name, encoder, N, K, Q, na_rate, root):
        self.root = root
        path = os.path.join(root, name + ".json")
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert 0
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask = self.encoder.tokenize(item['tokens'],
                                                       item['h'][2][0],
                                                       item['t'][2][0])
        return word, pos1, pos2, mask

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
        # The relation of all instances in an episode.
        label = []
        # The relation of all query instances in an episode
        query_label = []
        na_classes = list(filter(lambda x: x not in target_classes,
                                 self.classes))
        support_index = []

        # support set
        for i, _ in enumerate(target_classes):
            label += [i] * self.K

        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K, False)
            for j in indices:
                word, pos1, pos2, mask = self.__getraw__(
                    self.json_data[class_name][j])
                word = torch.tensor(word).long()
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                self.__additem__(support_set, word, pos1, pos2, mask)
                # Make a note of the class and subscript for each support instance.
                support_index.append((class_name, j))

        # query set
        count = 0
        # Generates a probability for each query instance in the query set.
        query_prob = np.random.rand(self.Q)
        query_prob = np.sort(query_prob)
        # Rank NOTA after all non-NOTAs for easier loss calculation.
        query_prob = query_prob[::-1]
        while count < self.Q:
            # If the probability is greater than the nota rate, non-NOTA is sampled. Otherwise sample NOTA.
            if query_prob[count] > self.na_rate:
                cur_class = np.random.choice(target_classes, 1, False)[0]
            else:
                cur_class = np.random.choice(na_classes, 1, False)[0]
            idx = np.random.choice(list(range(len(self.json_data[cur_class]))), 1, False)[0]
            # Resample if the sampled instance occurs in the support set.
            if (cur_class, idx) in support_index:
                continue
            count += 1
            word, pos1, pos2, mask = self.__getraw__(
                self.json_data[cur_class][idx])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            self.__additem__(query_set, word, pos1, pos2, mask)
            if cur_class not in target_classes:
                query_label += [self.N]
                label += [self.N]
            else:
                query_label += [target_classes.index(cur_class)]
                label += [target_classes.index(cur_class)]

        return support_set, query_set, query_label, label

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    batch_query_label = []
    batch_label = []
    support_sets, query_sets, query_labels, labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_query_label += query_labels[i]
        batch_label += labels[i]
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_query_label = torch.tensor(batch_query_label)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_query_label, batch_label


def get_loader_fewrel_fs(name, encoder, N, K, Q, na_rate, batch_size,
                         num_workers=0, collate_fn=collate_fn, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)
