import numpy as np
import pandas as pd
from tqdm import tqdm

from phase_3.pagerank.create_similarity_matrix import create_similarity
from phase_3.pagerank.ppr import PPR


class SVMClassifier:
    def __init__(self):
        self.ws = []

    def fit(self, X, y):
        ws = np.zeros(len(X[0]))
        l, epoch = 2, 10000

        for e in range(epoch):
            for i in range(len(X)):
                ws = (ws + l * ((y[i] * X[i]) - (2 * (1 / epoch) * ws))) \
                    if (y[i] * np.dot(X[i], ws) < 1) else (ws + l * (-2 * (1 / epoch) * ws))

        y_pred_train = [np.dot(_x, ws) for _x in X]
        self.ws = ws
        return ws, [1 if _y >= 1 else 0 for _y in y_pred_train]

    def predict(self, X):
        y_pred = [np.dot(_x, self.ws) for _x in X]
        y_pred = [1 if _o >= 1 else 0 for _o in y_pred]
        return y_pred


class Node:
    def __init__(self, y_pred):
        self.y_pred = y_pred
        self.f_i = 0
        self.threshold = 0
        self.l = None
        self.r = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.y_count = 0
        self.dimensionality = 0
        self.root = None

    def fit(self, X, y):
        self.y_count = len(set(y))
        self.dimensionality = X.shape[1]
        self.root = self.extend(X, y)

    def predict(self, X):
        return [self.predict_single(inputs) for inputs in X]

    def get_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.y_count)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.dimensionality):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.y_count
            num_right = num_parent[:]
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] = num_left[c] + 1
                num_right[c] = num_right[c] - 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.y_count))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.y_count))
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def extend(self, X, y, depth=0):
        y_pred = np.argmax([np.sum(y == i) for i in range(self.y_count)])
        node = Node(y_pred=y_pred)
        if depth < self.max_depth:
            idx, thr = self.get_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.f_i = idx
                node.threshold = thr
                node.l = self.extend(X_left, y_left, depth + 1)
                node.r = self.extend(X_right, y_right, depth + 1)
        return node

    def predict_single(self, x):
        node = self.root
        while node.l:
            node = node.l if x[node.f_i] < node.threshold else node.r
        return node.y_pred


def do_svm(_X_train, _X_test, _y_train, _y_test):
    _y_train = _y_train.to_numpy()
    _y_test = _y_test.to_numpy()

    clf = SVMClassifier()
    clf.fit(_X_train, _y_train)

    y_pred_test = clf.predict(_X_test)

    return float(sum([1 if _y == _y_pred else 0 for _y, _y_pred in zip(y_pred_test, _y_test)])) / len(y_pred_test)


def do_dt(_X_train, _X_test, _y_train, _y_test):
    _y_train = _y_train.to_numpy()
    _y_test = _y_test.to_numpy()

    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(_X_train.to_numpy(), _y_train)

    y_pred_test = clf.predict(_X_test.to_numpy())

    return float(sum([1 if _y == _y_pred else 0 for _y, _y_pred in zip(y_pred_test, _y_test)])) / len(y_pred_test)


def do_ppr(_X_train, _X_test, _y_train, _y_test):
    create_similarity(pd.concat([_X_train, _X_test], axis=0))
    ppr = PPR()
    y_preds = []
    for image_id in tqdm(_X_test.index):
        similar = ppr.compute([image_id], _K=10, _k=5)
        y_pred = [_y_train.loc[int(s["other_image_id"])] for s in similar if int(s["other_image_id"]) in _y_train.index]
        y_pred = max(set(y_pred), key=y_pred.count) if len(y_pred) > 0 else 1
        y_actual = _y_test.loc[image_id]
        y_preds.append((y_pred, y_actual))

    return float(sum([1 if a == b else 0 for a, b in y_preds])) / len(y_preds)


def do_classify(classifier_type, X_train, X_test, y_train, y_test):
    return {
        "svm": do_svm,
        "dt": do_dt,
        "ppr": do_ppr
    }[classifier_type](X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    do_classify()
