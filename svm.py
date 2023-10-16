import numpy as np
import copy
from cvxopt import spmatrix, matrix, solvers
from sklearn.datasets import make_classification, make_moons, make_blobs
from typing import NoReturn, Callable

solvers.options['show_progress'] = False


# Task 3

class KernelSVM:
    def __init__(self, C: float, kernel: Callable):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.
        kernel : Callable
            Функция ядра.

        """
        self.C = C
        self.kernel = kernel
        self.support = None
        self.alpha = None
        self.b = None
        self.useful_alpha = None
        self.useful_X = None
        self.useful_y = None

    def find_alpha(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        n, m = X.shape
        Q = matrix(np.array([y * y[i] * self.kernel(X, X[i]) for i in range(n)]), (n, n))
        p = matrix(-1.0, (n, 1))
        I = matrix(0.0, (n, n))
        I[::n + 1] = 1.0
        G = matrix([-I, I])
        h = matrix([0.0] * n + [self.C] * n, (2 * n, 1))
        A = matrix([float(y[i]) for i in range(n)], (1, n))
        b = matrix(0.0)
        sol = solvers.qp(Q, p, G, h, A, b)
        self.alpha = np.array(sol['x']).reshape((n,))

    def predict_no_shift(self, X: np.ndarray):
        ans = np.zeros((X.shape[0],), dtype=X.dtype)
        for j in range(self.useful_X.shape[0]):
            ans += self.useful_alpha[j] * self.useful_y[j] * self.kernel(X, self.useful_X[j])
        return ans

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        self.find_alpha(X, y)
        useful_indexes = 1e-6 < self.alpha
        self.useful_alpha = self.alpha[useful_indexes]
        self.useful_X = X[useful_indexes]
        self.useful_y = y[useful_indexes]
        self.support = [i for i, v in enumerate(self.alpha) if 1e-6 < v < self.C - 1e-6]
        self.b = self.predict_no_shift(X[self.support[0]][None, :]) - y[self.support[0]]

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Возвращает значение решающей функции.

        Parameters
        ----------
        X : np.ndarray
            Данные, для которых нужно посчитать значение решающей функции.

        Return
        ------
        np.ndarray
            Значение решающей функции для каждого элемента X
            (т.е. то число, от которого берем знак с целью узнать класс).

        """
        return self.predict_no_shift(X) - self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Классифицирует элементы X.

        Parameters
        ----------
        X : np.ndarray
            Данные, которые нужно классифицировать

        Return
        ------
        np.ndarray
            Метка класса для каждого элемента X.

        """
        return np.sign(self.decision_function(X))


# Task 1

class LinearSVM(KernelSVM):
    def __init__(self, C: float):
        """

        Parameters
        ----------
        C : float
            Soft margin coefficient.

        """
        super().__init__(C, get_simple_kernel())
        self.C = C
        self.alpha = None
        self.w = None
        self.b = None
        self.support = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает SVM, решая задачу оптимизации при помощи cvxopt.solvers.qp

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения SVM.
        y : np.ndarray
            Бинарные метки классов для элементов X
            (можно считать, что равны -1 или 1).

        """
        super().fit(X, y)
        self.w = np.sum(self.useful_X * (self.useful_alpha * self.useful_y)[:, None], axis=0)


# Task 2


def get_simple_kernel():
    def kernel(X, y):
        return X.dot(y)

    return kernel


def get_polynomial_kernel(c=1, power=2):
    "Возвращает полиномиальное ядро с заданной константой и степенью"

    def kernel(X, y):
        return np.power(X.dot(y) + c, power)

    return kernel


def get_gaussian_kernel(sigma=1.):
    "Возвращает ядро Гаусса с заданным коэффицинтом сигма"

    def kernel(X, y):
        return np.exp(-sigma * np.square(np.linalg.norm(X - y, axis=1)))

    return kernel

