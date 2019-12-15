
import math
import random
import numpy as np


def load_data(path):
    data = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            r = line.split('\t')
            if len(r) == 3:
                data.append(r)
    return data


def data_generate(data, batch_size=32):
    """数据生成，每个迭代返回pos和neg两个sample集合，还有margin"""
    entities = {}
    for x in data:
        if x[0] not in entities:
            entities[x[0]] = {}
        if x[2] not in entities:
            entities[x[2]] = {}
        if x[1] not in entities[x[0]]:
            entities[x[0]][x[1]] = []
        if x[1] not in entities[x[2]]:
            entities[x[2]][x[1]] = []
        entities[x[0]][x[1]].append(x[2])
        entities[x[2]][x[1]].append(x[0])

    words = list(entities.keys())
    n_batch = math.ceil(len(data) / batch_size)

    def _get_random(x0, r, x1):
        while True:
            neg = random.choice(words)
            if neg != x0 and neg != x1:
                if neg not in entities[x0][r]:
                    if neg not in entities[x1][r]:
                        return neg
    while True:
        for i in range(n_batch):
            batch = data[i * batch_size: (i + 1) * batch_size]
            if len(batch) < batch_size:
                batch = batch + data[: batch_size - len(batch)]
            neg = [
                [x0, r, _get_random(x0, r, x1)]
                for x0, r, x1 in batch
            ]
            y = np.array([[1.]] * len(batch))  # margin
            yield [np.array(batch), np.array(neg)], y
