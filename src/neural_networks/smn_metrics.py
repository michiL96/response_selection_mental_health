import numpy as np


def compute_recall_at_k(scores, labels, pool_size=10, k=1):
    total = 0
    correct = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            total = total+1
            sublist = np.asarray(scores[i:i + pool_size])
            sorted_indexes = np.argsort(sublist, axis=0)[::-1]
            if 0 in sorted_indexes[:k]:
                correct = correct + 1
    print(f'1 in {pool_size} => R@{k} = {round(float(correct)/total, 4)}')
    return float(correct)/total
