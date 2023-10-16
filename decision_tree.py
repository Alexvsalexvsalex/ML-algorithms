from sklearn.datasets import make_blobs, make_moons
import math
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List


# Task 1

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


# Task 2

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

    def predict_proba_single(self, X: np.ndarray):
        return self.counted


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

    def __init__(self, split_dim: int, split_value: float,
                 left: Union['DecisionTreeNode', DecisionTreeLeaf],
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

    def predict_proba_single(self, X: np.ndarray):
        if X[self.split_dim] < self.split_value:
            return self.left.predict_proba_single(X)
        else:
            return self.right.predict_proba_single(X)


# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """

    def __init__(self, criterion: str = "gini",
                 max_depth: Optional[int] = None,
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.crit = gini if criterion == "gini" else entropy
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        points_cnt, dim = X.shape

        def __build_tree_recursive(indexes, depth: int):
            points_cnt = indexes.shape[0]
            if points_cnt <= self.min_samples_leaf or depth == self.max_depth:
                return DecisionTreeLeaf(y[indexes])
            else:
                split_desc = None
                best_gain = 0.0
                original_error = self.crit(X)
                for cur_dim in range(dim):
                    idx = indexes[np.argsort(X[indexes, cur_dim])]
                    for i in range(1, points_cnt - 1):
                        if X[idx[i]][cur_dim] == X[idx[i - 1]][cur_dim]:
                            continue
                        left_size, right_size = i, points_cnt - i
                        if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                            continue
                        left = y[idx[:i]]
                        right = y[idx[i:]]
                        calculated_gain = gain(left, right, self.crit, original_error)
                        if best_gain < calculated_gain:
                            split_desc = i, cur_dim
                            best_gain = calculated_gain
                if split_desc is not None:
                    split_i, split_dim = split_desc
                    idx = indexes[np.argsort(X[indexes, split_dim])]
                    split_value = X[idx[split_i]][split_dim]
                    return DecisionTreeNode(
                        split_dim,
                        split_value,
                        __build_tree_recursive(idx[:split_i], depth + 1),
                        __build_tree_recursive(idx[split_i:], depth + 1),
                    )
                else:
                    return DecisionTreeLeaf(y[indexes])

        self.root = __build_tree_recursive(np.arange(points_cnt), 1)

    def predict_proba(self, X: np.ndarray) -> List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь
            {метка класса -> вероятность класса}.
        """

        return [self.root.predict_proba_single(X[i]) for i in range(X.shape[0])]

    def predict(self, X: np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.

        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]


# Task 4
task4_dtc = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=5
)
