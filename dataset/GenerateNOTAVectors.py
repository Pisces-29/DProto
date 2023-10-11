import torch
import os
import numpy as np
import random
import json


def generate_NOTA_vector(encoder, root, name, num=20):
    path = os.path.join(root, name + ".json")
    if not os.path.exists(path):
        print("[ERROR] Data file does not exist!")
        assert 0
    json_data = json.load(open(path))
    classes = list(json_data.keys())
    target_classes = random.sample(classes, num)
    random.shuffle(target_classes)
    NOTA_vector_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': []}
    for i, class_name in enumerate(target_classes):
        indices = np.random.choice(
            list(range(len(json_data[class_name]))), 10, True)
        for j in indices:
            item = json_data[class_name][j]
            word, pos1, pos2, mask = encoder.tokenize(item['tokens'], item['h'][2][0], item['t'][2][0])
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            NOTA_vector_set['word'].append(word)
            NOTA_vector_set['pos1'].append(pos1)
            NOTA_vector_set['pos2'].append(pos2)
            NOTA_vector_set['mask'].append(mask)

    for item in NOTA_vector_set.keys():
        NOTA_vector_set[item] = torch.stack(NOTA_vector_set[item], 0)

    NOTA_vectors = encoder(NOTA_vector_set)
    NOTA_vectors = NOTA_vectors.view(num, 10, -1)
    NOTA_vectors = torch.mean(NOTA_vectors, dim=1)

    return NOTA_vectors