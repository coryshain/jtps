import sys
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, f1_score, roc_auc_score


def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def get_random_permutation(n):
    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv


def evaluate_classifier(true, pred):
    acc = accuracy_score(true, pred)
    p = precision_score(true, pred, average='macro')
    r = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')

    return {
        'acc': acc,
        'p': p,
        'r': r,
        'f1': f1
    }


def pretty_print_seconds(s):
    s = int(s)
    h = s // 3600
    m = s % 3600 // 60
    s = s % 3600 % 60
    return '%02d:%02d:%02d' % (h, m, s)


def load_model(dir_path):
    """
    Convenience method for reconstructing a saved model object. First loads in metadata from ``m.obj``, then uses
    that metadata to construct the computation graph. Then, if saved weights are found, these are loaded into the
    graph.

    :param dir_path: Path to directory containing the DTSR checkpoint files.
    :return: The loaded DNNSeg instance.
    """

    with open(dir_path + '/m.obj', 'rb') as f:
        m = pickle.load(f)
    m.build(outdir=dir_path)
    m.load(outdir=dir_path)
    return m
