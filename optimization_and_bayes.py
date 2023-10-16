import math
from typing import NoReturn

import numpy as np
import pandas
import random
import copy


# Task 1

def cyclic_distance(points, dist):
    ans = 0.0
    n = len(points)
    for i in range(n - 1):
        ans += dist(points[i], points[i + 1])
    ans += dist(points[n - 1], points[0])
    return ans


def fix_cyclic_distance(points, permutation, current, dist, i, j):
    n = points.shape[0]

    if i != j + 1:
        current -= dist(points[permutation[j]], points[permutation[(j + 1) % n]])
        current -= dist(points[permutation[i - 1]], points[permutation[i]])
    current += dist(points[permutation[i - 1]], points[permutation[j]])
    current += dist(points[permutation[i]], points[permutation[(j + 1) % n]])

    if i != (j - 1 + n) % n:
        current -= dist(points[permutation[j - 1]], points[permutation[j]])
        current -= dist(points[permutation[i]], points[permutation[(i + 1) % n]])
    current += dist(points[permutation[j]], points[permutation[(i + 1) % n]])
    current += dist(points[permutation[j - 1]], points[permutation[i]])

    return current


def l2_distance(p1, p2):
    return np.linalg.norm(p2 - p1)


def l1_distance(p1, p2):
    return np.sum(np.abs(p2 - p1))


# Task 2

class HillClimb:
    def __init__(self, max_iterations, dist):
        self.max_iterations = max_iterations
        self.dist = dist  # Do not change

    def optimize(self, X):
        return self.optimize_explain(X)[-1]

    def optimize_explain(self, X):
        n = X.shape[0]
        permutation = np.arange(n)
        np.random.shuffle(permutation)
        history = []
        origin_dist = cyclic_distance(X[permutation], self.dist)
        for it in range(self.max_iterations):
            best = None
            best_dist = origin_dist
            for i in range(n):
                for j in range(i):
                    current_dist = fix_cyclic_distance(X, permutation, origin_dist, self.dist, i, j)
                    if current_dist < best_dist:
                        best = i, j
                        best_dist = current_dist
            if best is not None:
                i, j = best
                permutation[i], permutation[j] = permutation[j], permutation[i]
                history.append(permutation.copy())
                origin_dist = best_dist
            else:
                break
        return history


# Task 3

class Genetic:
    def __init__(self, iterations, population, survivors, distance):
        self.pop_size = population
        self.surv_size = survivors
        self.dist = distance
        self.iters = iterations

    def optimize(self, X):
        all_permutations = self.optimize_explain(X)
        ans = None
        ans_dist = 1e15
        for many in all_permutations:
            d = cyclic_distance(X[many[0]], self.dist)
            if d < ans_dist:
                ans = many[0]
                ans_dist = d
        return ans

    def optimize_explain(self, X):
        n = X.shape[0]
        population = np.zeros((self.pop_size, n), dtype=np.int32)
        for i in range(self.pop_size):
            population[i] = np.random.permutation(n)  # Делаем случайные популяции
        history = []
        changes = self.pop_size - self.surv_size
        crossover_rand = np.random.choice(self.surv_size, (self.iters, changes, 2), replace=True)
        segment_rand = np.random.choice(n, (self.iters, changes, 2), replace=True)
        for it in range(min(self.iters, 30)):
            new_population = []
            for i in range(changes):
                for_crossover = population[crossover_rand[it][i]]  # Скрещивание
                segment = np.sort(segment_rand[it][i])
                segment_from_first = for_crossover[0][segment[0]:segment[1] + 1]  # Берем случайный участок из первого
                segment_from_second = np.setdiff1d(for_crossover[1], segment_from_first, assume_unique=True)
                result = np.concatenate([segment_from_first, segment_from_second])
                new_population.append(result)  # Добавление особи в популяцию
            new_population = np.array(new_population)
            population = np.concatenate([population, new_population])  # Склеиваем новые и старые
            distances = [cyclic_distance(X[i], self.dist) for i in population]
            sorted_idx = np.argsort(distances)[:self.pop_size]  # Выживают лучшие
            population = population[sorted_idx]
            history.append(population)
        while len(history) < self.iters:
            history.append(history[-1])
        return history


# Task 4

class BoW:
    def __init__(self, X: np.ndarray, voc_limit: int = 1000):
        from collections import Counter
        c = Counter()
        for sentence in X.tolist():
            c.update(sentence.split())
        self.frequent = c.most_common(voc_limit)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Векторизует предложения.

        Parameters
        ----------
        X : np.ndarray
            Массив строк (предложений) размерности (n_sentences, ),
            который необходимо векторизовать.

        Return
        ------
        np.ndarray
            Матрица векторизованных предложений размерности (n_sentences, vocab_size)
        """
        from collections import Counter
        voc_size = len(self.frequent)
        res = []
        for x in X:
            c = Counter(x.split())
            v = [c[self.frequent[i][0]] for i in range(voc_size)]
            res.append(v)
        return np.array(res)


# Task 5

class NaiveBayes:
    def __init__(self, alpha: float):
        """
        Parameters
        ----------
        alpha : float
            Параметр аддитивной регуляризации.
        """
        self.alpha = alpha
        self.count_size = 8
        self.from_y = None
        self.classes = None
        self.probs = None
        self.expectations = None
        self.derivative = None
        self.class_prob = None
        self.log_class_prob = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        sample_cnt, dim = X.shape
        unique_y, counts_y = np.unique(y, return_counts=True)
        self.classes = unique_y
        self.from_y = {unique_y[i]: i for i in range(unique_y.shape[0])}
        y = np.array([self.from_y[y_i] for y_i in y])
        counters_d = np.ones((unique_y.shape[0], dim, self.count_size), dtype=np.float64)
        counters_d = counters_d * self.alpha
        for i in range(sample_cnt):
            for feature in range(dim):
                vvv = min(max(X[i][feature], 0), self.count_size - 1)
                counters_d[y[i]][feature][vvv] += 1
        self.probs = np.zeros((unique_y.shape[0], dim, self.count_size), dtype=np.float64)
        for i in range(unique_y.shape[0]):
            for feature in range(dim):
                sm = np.sum(counters_d[i][feature])
                self.probs[i][feature] = np.log(counters_d[i][feature] / sm)
        self.class_prob = counts_y / sample_cnt
        self.log_class_prob = np.log(self.class_prob)

    def predict(self, X: np.ndarray) -> list:
        """
        Return
        ------
        list
            Предсказанный класс для каждого элемента из набора X.
        """
        return [self.classes[i] for i in np.argmax(self.log_proba(X), axis=1)]

    def log_proba_single(self, X: np.ndarray) -> np.ndarray:
        sm = np.zeros((self.classes.shape[0]))
        for i in range(X.shape[0]):
            vvv = min(max(X[i], 0), self.count_size - 1)
            sm += self.probs[:, i, vvv]
        return sm + self.log_class_prob

    def log_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return
        ------
        np.ndarray
            Для каждого элемента набора X - логарифм вероятности отнести его к каждому классу.
            Матрица размера (X.shape[0], n_classes)
        """
        return np.array([self.log_proba_single(x) for x in X])
