import heapq

from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn


# Task 1

def dist(point, point2):
    return np.linalg.norm(point - point2)


def get_nearest_centroid_static(point, centroids):
    centroids_cnt = len(centroids)
    min_dist = dist(point, centroids[0])
    min_dist_cluster_id = 0
    for cluster_id in range(1, centroids_cnt):
        d = dist(point, centroids[cluster_id])
        if min_dist > d:
            min_dist = d
            min_dist_cluster_id = cluster_id
    return min_dist_cluster_id


class KMeans:
    def __init__(self, n_clusters: int, init: str = "random",
                 max_iter: int = 300):
        """

        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.

        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None

    def fill_centroids(self, X: np.array, extend_mode: bool = False):
        if not extend_mode:
            self.centroids = None
        current_centroids = \
            self.centroids if self.centroids is not None else np.zeros((0, X.shape[1]), dtype=np.float)
        necessary_centroids = self.n_clusters - len(current_centroids)
        if necessary_centroids == 0:
            return

        if self.init == "random":
            min_coords = np.min(X, axis=(0,))
            max_coords = np.max(X, axis=(0,))
            new_centroids = np.random.uniform(min_coords, max_coords, (necessary_centroids, len(min_coords),))
            centroids = np.concatenate((current_centroids, new_centroids))
        elif self.init == "sample":
            new_centroids = X[np.random.choice(len(X), necessary_centroids, replace=False)]
            centroids = np.concatenate((current_centroids, new_centroids))
        elif self.init == "k-means++":
            centroids = current_centroids
            for _ in range(necessary_centroids):
                if len(centroids) > 0:
                    probabilities = np.array(
                        [
                            dist(point, centroids[get_nearest_centroid_static(point, centroids)]) ** 2
                            for point in X
                        ]
                    )
                    probabilities /= np.sum(probabilities)
                else:
                    probabilities = None
                new_centroid_id = np.random.choice(len(X), 1, replace=False, p=probabilities)
                centroids = np.append(centroids, X[new_centroid_id], axis=0)
        else:
            centroids = []
        self.centroids = centroids

    def dist_to_cluster(self, point, cluster_id):
        return dist(point, self.centroids[cluster_id])

    def get_nearest_centroid(self, point):
        return get_nearest_centroid_static(point, self.centroids)

    def fit(self, X: np.array, y=None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.

        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать
            параметры X и y, даже если y не используется).

        """
        self.fill_centroids(X)
        for _ in range(self.max_iter):
            centroids_cnt = len(self.centroids)
            grouped_by_cluster = [[] for _ in range(centroids_cnt)]
            for sample_point in X:
                min_dist_cluster_id = self.get_nearest_centroid(sample_point)
                grouped_by_cluster[min_dist_cluster_id].append(sample_point)
            grouped_by_cluster = list(filter(lambda x: len(x) > 0, grouped_by_cluster))
            current_centroids = \
                np.array(list(map(lambda cluster_points: np.average(cluster_points, axis=0), grouped_by_cluster)))
            self.centroids = current_centroids
            self.fill_centroids(X, extend_mode=False)

    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера,
        к которому относится данный элемент.

        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.

        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров
            (по одному индексу для каждого элемента из X).

        """
        return [self.get_nearest_centroid(point) for point in X]


# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5,
                 leaf_size: int = 40, metric: str = "euclidean"):
        """

        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        points_cnt = len(X)
        tree = KDTree(X, self.leaf_size, self.metric)
        neighbours_ = tree.query_radius(X, self.eps)
        edges = [[] for _ in range(points_cnt)]
        for i, neighbours in enumerate(neighbours_):
            if self.min_samples <= len(neighbours):
                # core
                for neighbour in neighbours:
                    edges[i].append(neighbour)
        labels = [-1] * points_cnt
        cur_label = 0
        for i in range(points_cnt):
            if len(edges[i]) > 0 and labels[i] == -1:
                d = deque()
                d.append(i)
                while len(d) > 0:
                    v = d.popleft()
                    if labels[v] >= 0:
                        continue
                    labels[v] = cur_label
                    for edge in edges[v]:
                        d.append(edge)
                cur_label += 1
        return np.array(labels)


# Task 3

import heapq


class DSU:
    def __init__(self, points_cnt):
        self.points_cnt = points_cnt
        self.vertexes = [i for i in range(points_cnt)]
        self.sizes = [1 for _ in range(points_cnt)]
        self.classes = {i for i in range(points_cnt)}

    def get(self, vertex):
        cur_vertex = self.vertexes[vertex]
        if cur_vertex != vertex:
            self.vertexes[vertex] = self.get(cur_vertex)
            return self.vertexes[vertex]
        else:
            return cur_vertex

    def merge(self, vertex1, vertex2):
        comp1 = self.get(vertex1)
        comp2 = self.get(vertex2)
        if self.sizes[comp1] < self.sizes[comp2]:
            comp1, comp2 = comp2, comp1
        self.sizes[comp1] += self.sizes[comp2]
        self.vertexes[comp2] = comp1
        self.classes.remove(comp2)
        return comp1

    def get_all_labels(self):
        return [self.get(i) for i in range(self.points_cnt)]

    def get_normalized_labels(self):
        labels = self.get_all_labels()
        different_labels = list(set(labels))
        mapper = {diff_label: i for i, diff_label in enumerate(different_labels)}
        return list(map(lambda x: mapper[x], labels))


class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """

        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self, X: np.array, y=None) -> np.array:
        """
        Кластеризует элементы из X,
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """
        points_cnt = len(X)
        dsu = DSU(points_cnt)
        labels_cnt = points_cnt
        h = []
        metric_value = {}
        for i in range(points_cnt):
            metric_value[i] = {}
            for j in range(i):
                dst = dist(X[i], X[j])
                metric_value[i][j] = dst
                metric_value[j][i] = dst
                heapq.heappush(h, (dst, i, j))
        while labels_cnt > self.n_clusters:
            _, f_c, s_c = heapq.heappop(h)
            f_c, s_c = dsu.get(f_c), dsu.get(s_c)
            if f_c == s_c:
                continue
            f_s, s_s = dsu.sizes[f_c], dsu.sizes[s_c]
            res_c = dsu.merge(f_c, s_c)
            for t_c in dsu.classes:
                if f_c != t_c and s_c != t_c:
                    if self.linkage == "average":
                        calced_dist = (s_s * metric_value[f_c][t_c] + f_s * metric_value[s_c][t_c]) / (f_s + s_s)
                    elif self.linkage == "single":
                        calced_dist = min(metric_value[f_c][t_c], metric_value[s_c][t_c])
                    else:
                        calced_dist = max(metric_value[f_c][t_c], metric_value[s_c][t_c])
                    metric_value[res_c][t_c] = calced_dist
                    metric_value[t_c][res_c] = calced_dist
                    heapq.heappush(h, (calced_dist, res_c, t_c))
            labels_cnt -= 1
        return np.array(dsu.get_normalized_labels())
