import json
import os
import scipy.sparse
from sklearn.preprocessing import StandardScaler
import dgl
import numpy as np
import torch
from sklearn.metrics import f1_score


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

