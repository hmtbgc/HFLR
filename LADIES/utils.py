import json
import os
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import dgl
import numpy as np
import torch
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt

def calc_f1(y_true, y_pred, multilabel):
    if multilabel:
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = 0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro")

def evaluate(model, g, labels, mask, multilabel, test):
    model.eval()
    idx = torch.where(mask == True)[0]
    if not test:
        logits = model.minibatch_infer(g, idx)
    else:
        logits = model.test_infer(g)
    logits = logits[mask]
    labels = labels[mask]
    f1 = calc_f1(labels.cpu().numpy(), logits.cpu().numpy(), multilabel)
    return f1


def degree_plot(g, train_nid, name, max_d=100):
    d = g.in_degrees()[train_nid]
    values, counts = torch.unique(d, return_counts=True)
    print(max(values))
    print(max(counts))
    plt.figure(figsize=(10, 6))
    plt.xlim(0, max_d)
    plt.ylim(0, max(counts) * 1.1)
    plt.bar(values, counts)
    #plt.xticks(values)
    #plt.yticks(np.arange(0, max(counts) * 1.1, step=max(counts) // 4))
    plt.xlabel('Node Occurrence Frequency')
    plt.ylabel('Number of Nodes')
    plt.title('Histogram of Node Occurrence Frequencies')
    plt.savefig(f'{name}_node_frequecy.jpg', dpi=300)