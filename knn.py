import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List

import csv


# Task 1

def get_all_data(path_to_csv: str) -> List:
    with open(path_to_csv, mode='r', newline='') as f:
        reader = csv.reader(f, delimiter=',')
        return list(reader)[1:]


def split_lists(lists: List[List], index: int) -> Tuple[List[List], List[List]]:
    left = [l[:index] for l in lists]
    right = [l[index:] for l in lists]
    return left, right


def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M),
        0 --- злокачественной (B).


    """
    all_data = get_all_data(path_to_csv)
    random.shuffle(all_data)
    raw_y, raw_X = split_lists(all_data, 1)
    prepared_y = list(map(lambda sample_y: 1 if sample_y[0] == 'M' else 0, raw_y))
    y = np.array(prepared_y, np.int32)
    X = np.array(raw_X, np.float32)
    return X, y


def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """

    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток,
        1 если сообщение содержит спам, 0 если не содержит.

    """
    all_data = get_all_data(path_to_csv)
    random.shuffle(all_data)
    raw_X, raw_y = split_lists(all_data, len(all_data[0]) - 1)
    prepared_y = list(map(lambda sample_y: sample_y[0], raw_y))
    y = np.array(prepared_y, np.int32)
    X = np.array(raw_X, np.float32)
    return X, y


# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    split_index = int(X.shape[0] * ratio)
    return X[:split_index], y[:split_index], X[split_index:], y[split_index:]


# Task 3

def get_classes_cnt(y) -> int:
    return y.max(initial=0) + 1


def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    samples_cnt = len(y_pred)
    classes_cnt = get_classes_cnt(np.concatenate((y_pred, y_true)))
    result = np.zeros(shape=(classes_cnt, classes_cnt,))
    for i in range(samples_cnt):
        pred = y_pred[i]
        true = y_true[i]
        result[pred][true] += 1

    precision = np.zeros(shape=(classes_cnt,))
    recall = np.zeros(shape=(classes_cnt,))
    for i in range(classes_cnt):
        column = result[:, i]
        row = result[i, :]
        precision[i] = row[i] / np.sum(row)
        recall[i] = column[i] / np.sum(column)
    accuracy = np.trace(result) / samples_cnt
    return precision, recall, accuracy


# Task 4

class KDNode:
    def __init__(self, dim, value, left_child, right_child):
        self.dim = dim
        self.value = value
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self):
        return f"KDNode({self.left_child}, {self.right_child})"

    def get_all_leaves(self):
        return self.left_child.get_all_leaves() + self.right_child.get_all_leaves()


class KDLeaf:
    def __init__(self, points: List[List[int]]):
        self.points = points

    def __str__(self):
        return f"KDLeaf({self.points})"

    def get_all_leaves(self):
        return self.points


class CachedDist:
    def __init__(self, main_point):
        self.cached_dists = {}
        self.main_point = main_point

    def query(self, point):
        if point[0] not in self.cached_dists:
            self.cached_dists[point[0]] = np.linalg.norm(self.main_point - point[1])
        return self.cached_dists[point[0]]


class SegmentHolder:
    def __init__(self, dim):
        self.dim = dim
        self.lower = [[None] for _ in range(dim)]
        self.upper = [[None] for _ in range(dim)]

    def add_up_lim(self, node):
        self.upper[node.dim].append(node.value)

    def add_low_lim(self, node):
        self.lower[node.dim].append(node.value)

    def drop_up_lim(self, dim):
        self.upper[dim].pop()

    def drop_low_lim(self, dim):
        self.lower[dim].pop()

    def dist_to_point(self, point):
        def get_diff(p, l, u):
            if (l is None or l <= p) and (u is None or p <= u):
                return p
            elif l is not None and l > p:
                return l
            else:
                return u

        nearest_point = np.array(
            [get_diff(point[dim], self.lower[dim][-1], self.upper[dim][-1]) for dim in range(self.dim)])
        return np.linalg.norm(nearest_point - point)


class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области,
            в которых не меньше leaf_size точек).

        Returns
        -------

        """
        self.dim = len(X[0])

        def __build_tree_recursive(points, cur_dim):
            points_cnt = len(points)
            if points_cnt <= leaf_size:
                return KDLeaf(points)
            else:
                split = points, []
                for shift in range(self.dim):
                    # check all elements the same
                    if all([point[1][cur_dim] == points[0][1][cur_dim] for point in points]):
                        cur_dim = (cur_dim + 1) % self.dim
                        continue
                    sorted_points = sorted(points, key=lambda x: x[1][cur_dim])
                    i = 0
                    j = 0
                    while i + j < points_cnt:
                        if sorted_points[i][1][cur_dim] != sorted_points[points_cnt - 1 - j][1][cur_dim]:
                            if i < j:
                                i += 1
                            else:
                                j += 1
                        else:
                            if i < j:
                                i = points_cnt - j
                            else:
                                j = points_cnt - i
                    split = sorted_points[:i], sorted_points[i:]
                    break
                left_points, right_points = split
                if len(left_points) > 0 and len(right_points) > 0:
                    median = (left_points[-1][1][cur_dim] + right_points[0][1][cur_dim]) / 2
                    next_dim = (cur_dim + 1) % self.dim
                    return KDNode(
                        cur_dim,
                        median,
                        __build_tree_recursive(left_points, next_dim),
                        __build_tree_recursive(right_points, next_dim),
                    )
                else:
                    return KDLeaf(left_points or right_points)

        self.tree = __build_tree_recursive(list(enumerate(X)), 0)

    def __str__(self):
        return f"KDTree({self.tree})"

    def query(self, X: np.array, k: int = 1, return_distance=False) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k):
            индексы k ближайших соседей для всех точек из X.

        """
        result = [list(map(lambda x: x[0], self.single_query(x, k))) for x in X]
        return result

    def single_query(self, x: np.array, k: int) -> List:
        cached_dist = CachedDist(x)
        segment_holder = SegmentHolder(self.dim)

        def get_bests(leaves):
            return sorted(leaves, key=cached_dist.query)[:k]

        def dfs(node, existing_result) -> List:
            if len(existing_result) == k:
                # If all candidate places are taken
                farthest_neighbour = existing_result[-1]
                dist_to_farthest = cached_dist.query(farthest_neighbour)
                if segment_holder.dist_to_point(x) > dist_to_farthest:
                    # We cannot make it better
                    return existing_result
            if isinstance(node, KDLeaf):
                return get_bests(existing_result + node.get_all_leaves())
            elif isinstance(node, KDNode):
                def process_up_child(lim_node, child, current_result):
                    segment_holder.add_up_lim(lim_node)
                    result = dfs(child, current_result)
                    segment_holder.drop_up_lim(lim_node.dim)
                    return result

                def process_low_child(lim_node, child, current_result):
                    segment_holder.add_low_lim(lim_node)
                    result = dfs(child, current_result)
                    segment_holder.drop_low_lim(lim_node.dim)
                    return result

                if x[node.dim] <= node.value:
                    existing_result = process_up_child(node, node.left_child, existing_result)
                    existing_result = process_low_child(node, node.right_child, existing_result)
                    return existing_result
                else:
                    existing_result = process_low_child(node, node.right_child, existing_result)
                    existing_result = process_up_child(node, node.left_child, existing_result)
                    return existing_result

        return dfs(self.tree, [])


# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size

    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """
        self.tree = KDTree(X, self.leaf_size)
        self.y = y
        self.classes_cnt = get_classes_cnt(y)

    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.


        """
        neighbours = self.tree.query(X, self.n_neighbors)
        result = []
        for indicies in neighbours:
            counter = np.zeros(shape=(self.classes_cnt,))
            for indx in indicies:
                counter[self.y[indx]] += 1
            result.append(counter / np.sum(counter))
        return result

    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.

        Returns
        -------
        np.array
            Вектор предсказанных классов.


        """
        return np.argmax(self.predict_proba(X), axis=1)
