import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения),
        w[0] должен соответстовать константе,
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.

        """

        self.w = None
        self.mp = None
        self.iterations = iterations

    def _extend_x(self, X: np.ndarray):
        return np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))

    def _predict(self, X):
        return np.matmul(X, self.w)

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон.
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.

        """
        X = self._extend_x(X)
        samples_cnt = X.shape[0]
        features_cnt = X.shape[1]
        self.w = np.zeros((features_cnt,), dtype=X.dtype)
        labels = np.unique(y)
        self.mp = {-1: labels[0], 1: labels[1]}
        inv_mp = {labels[0]: -1, labels[1]: 1}
        for i in range(samples_cnt):
            y[i] = inv_mp[y[i]]
        for it in range(self.iterations):
            y_pred = self._predict(X)
            y_pred_o = np.sign(y_pred)
            df = y != y_pred_o
            self.w += np.sum(X[df] * y[df, np.newaxis], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.

        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.

        Return
        ------
        labels : np.ndarray
            Вектор индексов классов
            (по одной метке для каждого элемента из X).

        """
        X = self._extend_x(X)
        y_pred = self._predict(X)
        res = [self.mp[1] if y_pred[i] >= 0 else self.mp[-1] for i in range(y_pred.shape[0])]
        ans = np.array(res)
        return ans


# Task 2

class PerceptronBest:
    def __init__(self, iterations: int = 100):
        self.w = None
        self.mp = None
        self.iterations = iterations

    def _extend_x(self, X: np.ndarray):
        return np.hstack((np.ones((X.shape[0], 1), dtype=X.dtype), X))

    def _predict2(self, X):
        return np.sign(np.matmul(X, self.w))

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        bst_score = None
        bst_it = None
        X = self._extend_x(X)
        samples_cnt = X.shape[0]
        features_cnt = X.shape[1]
        self.w = np.zeros((features_cnt,), dtype=X.dtype)
        labels = np.unique(y)
        self.mp = {-1: labels[0], 1: labels[1]}
        inv_mp = {labels[0]: -1, labels[1]: 1}
        for i in range(samples_cnt):
            y[i] = inv_mp[y[i]]
        save_every_it = self.iterations // 100
        saved_weights = []
        for it in range(self.iterations + 1):
            y_pred = self._predict2(X)
            df = y != y_pred
            sc = np.sum(df)
            if bst_score is None or bst_score > sc:
                bst_score = sc
                bst_it = it
            if it % save_every_it == 0:
                saved_weights.append(np.copy(self.w))
            self.w += np.sum(X[df] * y[df, np.newaxis], axis=0)

        weight_i = bst_it // save_every_it
        need_it = bst_it % save_every_it
        self.w = saved_weights[weight_i]
        for it in range(need_it):
            y_pred = self._predict2(X)
            df = y != y_pred
            self.w += np.sum(X[df] * y[df, np.newaxis], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._extend_x(X)
        y_pred = self._predict2(X)
        res = [self.mp[x] for x in y_pred]
        ans = np.array(res)
        return ans


# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    cnt = images.shape[0]
    ans = np.zeros((cnt, 2))
    for i in range(cnt):
        image = images[i]
        wb = image > 0.45
        a = 0
        b = 0
        c = np.count_nonzero(wb)
        for w in range(images.shape[2]):
            begin = -1
            end = -1
            for h in range(images.shape[1]):
                if wb[h][w] == 1:
                    if begin == -1:
                        begin = h
                    end = h
            if begin != -1:
                for t in range(begin, end):
                    if wb[t][w] == 0:
                        a += 1
        for h in range(1, images.shape[1] - 1):
            for w in range(1, images.shape[2] - 1):
                if wb[h][w] == 1:
                    r = 0
                    if wb[h - 1][w] == 0:
                        r += 1
                    if wb[h + 1][w] == 0:
                        r += 1
                    if wb[h][w - 1] == 0:
                        r += 1
                    if wb[h][w + 1] == 0:
                        r += 1
                    if r <= 1:
                        b += 1
        ans[i][0] = a
        ans[i][1] = b / c
    return ans
