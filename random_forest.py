import math
from typing import Union, Callable

from sklearn.model_selection import train_test_split
import numpy as np
import pandas
import random
import copy


from catboost import CatBoostClassifier

# Task 0

def gini(x: np.ndarray) -> float:
    total = x.shape[0]
    unique, counts = np.unique(x, return_counts=True)
    return np.sum(counts * (total - counts)) / total ** 2


def entropy(x: np.ndarray) -> float:
    total = x.shape[0]
    unique, counts = np.unique(x, return_counts=True)
    return np.log2(total) - np.sum(counts * np.log2(counts)) / total


def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable, original=None) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    left_size = left_y.shape[0]
    right_size = right_y.shape[0]
    total = left_size + right_size
    original = original or criterion(np.concatenate([left_y, right_y]))
    splitted = left_size / total * criterion(left_y) + right_size / total * criterion(right_y)
    return original - splitted


# Task 1

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """

    def __init__(self, ys):
        from collections import Counter
        ys_size = ys.shape[0]
        c = Counter(ys.tolist())
        self.counted = {k: v / ys_size for k, v in c.items()}
        self.y = c.most_common(1)[0][0]

    def predict_single(self, X: np.ndarray):
        return self.y


class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value.
    """

    def __init__(self, split_dim: int,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.left = left
        self.right = right

    def predict_single(self, X: np.ndarray):
        if not X[self.split_dim]:
            return self.left.predict_single(X)
        else:
            return self.right.predict_single(X)


class DecisionTree:
    def __init__(self, X, y, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto"):
        points_cnt, dim = X.shape
        if max_features == "auto":
            cnt_features = int(math.ceil(math.sqrt(dim)))
        else:
            cnt_features = max_features
        cnt_features = min(cnt_features, dim)
        self.crit = gini if criterion == "gini" else entropy
        self.X = X
        self.y = y
        self.choiced_idx = np.random.choice(np.arange(points_cnt), points_cnt)
        self.in_bag_idx = np.zeros(points_cnt, dtype=bool)
        self.in_bag_idx[self.choiced_idx] = True
        self.out_of_bag_idx = ~self.in_bag_idx
        self.X_oob = X[self.out_of_bag_idx]
        self.y_oob = y[self.out_of_bag_idx]

        def __build_tree_recursive(indexes, depth: int):
            points_cnt = indexes.shape[0]
            if points_cnt <= min_samples_leaf or depth == max_depth:
                return DecisionTreeLeaf(y[indexes])
            else:
                split_dim = None
                best_gain = 0.0
                original_error = self.crit(y[indexes])
                choiced_features = np.random.choice(np.arange(dim), cnt_features, replace=False)
                for cur_dim in choiced_features:
                    true_idx = indexes[np.argwhere(X[indexes, cur_dim] == 1)].flatten()
                    false_idx = indexes[np.argwhere(X[indexes, cur_dim] == 0)].flatten()
                    left_size, right_size = false_idx.shape[0], true_idx.shape[0]
                    if left_size < min_samples_leaf or right_size < min_samples_leaf:
                        continue
                    left = y[false_idx]
                    right = y[true_idx]
                    calculated_gain = gain(left, right, self.crit, original_error)
                    if best_gain < calculated_gain:
                        split_dim = cur_dim
                        best_gain = calculated_gain
                if split_dim is not None:
                    true_idx = indexes[np.argwhere(X[indexes, split_dim] == 1)].flatten()
                    false_idx = indexes[np.argwhere(X[indexes, split_dim] == 0)].flatten()
                    return DecisionTreeNode(
                        split_dim,
                        __build_tree_recursive(false_idx, depth + 1),
                        __build_tree_recursive(true_idx, depth + 1),
                    )
                else:
                    return DecisionTreeLeaf(y[indexes])

        self.root = __build_tree_recursive(self.choiced_idx, 1)
        self.err_oob = self.calc_err_oob(self.X_oob, self.y_oob)

    def calc_err_oob(self, X, y):
        p = self.predict(X)
        return np.count_nonzero(p != y)

    def predict(self, X):
        return np.array([self.root.predict_single(X[i]) for i in range(X.shape[0])])

    def feature_importance(self, j):
        modified_X = self.X_oob.copy()
        np.random.shuffle(modified_X[:, j])
        new_err = self.calc_err_oob(modified_X, self.y_oob)
        return new_err - self.err_oob


# Task 2

class RandomForestClassifier:
    def __init__(self, criterion="gini", max_depth=None, min_samples_leaf=1, max_features="auto", n_estimators=10):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.criterion = criterion
        self.estimators = None
        self.points_cnt, self.dim = None, None

    def fit(self, X, y):
        self.points_cnt, self.dim = X.shape
        self.estimators = [
            DecisionTree(X, y, self.criterion, self.max_depth, self.min_samples_leaf, self.max_features)
            for _ in range(self.n_estimators)
        ]

    def predict(self, X):
        from collections import Counter
        predicts = np.array([estimator.predict(X) for estimator in self.estimators]).transpose()
        res = [Counter(row.tolist()).most_common(1)[0][0] for row in predicts]
        return res

    def feature_importance(self):
        imp_matrix = [
            [self.estimators[i].feature_importance(j) for i in range(self.n_estimators)]
            for j in range(self.dim)
        ]
        return [
            sum(
                imp_matrix[j][i] for i in range(self.n_estimators)
            ) / self.n_estimators
            for j in range(self.dim)
        ]


# Task 3

def feature_importance(rfc):
    if isinstance(rfc, RandomForestClassifier):
        rfc.feature_importance()
    else:
        pass
    return None


# Task 4

rfc_age = RandomForestClassifier(
    criterion="entropy",
    max_depth=None,
    min_samples_leaf=20,
    max_features=4,
    n_estimators=30
)
rfc_gender = RandomForestClassifier(
    criterion="gini",
    max_depth=6,
    min_samples_leaf=20,
    max_features=15,
    n_estimators=50
)

# Task 5
# Здесь нужно загрузить уже обученную модели
# https://catboost.ai/en/docs/concepts/python-reference_catboost_save_model
# https://catboost.ai/en/docs/concepts/python-reference_catboost_load_model
catboost_rfc_age = CatBoostClassifier()
catboost_rfc_age.load_model(__file__[:-7] + "rf_age.cbm")
catboost_rfc_gender = CatBoostClassifier()
catboost_rfc_gender.load_model(__file__[:-7] + "rf_gender.cbm")
