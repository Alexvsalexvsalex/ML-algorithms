import numpy as np
import math


# Task 1

def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    u = np.sum(np.square(y_predicted - y_true))
    n = y_true.shape[0]
    return u / n


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    average = np.average(y_true)
    u = np.sum(np.square(y_predicted - y_true))
    v = np.sum(np.square(y_true - average))
    return 1 - u / v


# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here
        self.mins = None
        self.maxs = None

    def _extend_x(self, X: np.ndarray):
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    def _normalize_fit(self, X: np.ndarray):
        self.mins = np.min(X, axis=0)
        self.maxs = np.max(X, axis=0)
        return self._normalize_predict(X)

    def _normalize_predict(self, X: np.ndarray):
        scale = self.maxs - self.mins
        shift = self.mins
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if scale[j] < 10e-8:
                    X[i][j] = 1.0
                else:
                    X[i][j] = ((X[i][j] - shift[j]) / scale[j] - 0.5) * 2
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._extend_x(X)
        X = self._normalize_fit(X)
        X_trans = X.T
        inv = np.linalg.inv(np.matmul(X_trans, X))
        self.weights = np.matmul(np.matmul(inv, X_trans), y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._extend_x(X)
        X = self._normalize_predict(X)
        return np.matmul(X, self.weights)


# Task 3

class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.weights = None  # Save weights here
        self.alpha = alpha
        self.iterations = iterations
        self.l = l

    def _extend_x(self, X: np.ndarray):
        return np.hstack((X, np.ones((X.shape[0], 1), dtype=X.dtype)))

    def _normalize_fit(self, X: np.ndarray):
        self.mins = np.min(X, axis=0)
        self.maxs = np.max(X, axis=0)
        return self._normalize_predict(X)

    def _normalize_predict(self, X: np.ndarray):
        scale = self.maxs - self.mins
        shift = self.mins
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if scale[j] < 10e-8:
                    X[i][j] = 1.0
                else:
                    X[i][j] = ((X[i][j] - shift[j]) / scale[j] - 0.5) * 2
        return X

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = self._extend_x(X)
        # X = self._normalize_fit(X)
        X_trans = X.T
        features_cnt = X.shape[1]
        samples_cnt = X.shape[0]
        self.weights = np.zeros(features_cnt)
        for it in range(self.iterations):
            y_pred = np.matmul(X, self.weights)
            grad = 2 * np.matmul(X_trans, y_pred - y)
            self.weights -= (self.alpha * grad + self.l * np.sign(self.weights)) / samples_cnt

    def predict(self, X: np.ndarray):
        X = self._extend_x(X)
        # X = self._normalize_predict(X)
        return np.matmul(X, self.weights)


# Task 4

def get_feature_importance(linear_regression):
    weights = linear_regression.weights[:-1]
    return np.abs(weights)


def get_most_important_features(linear_regression):
    weights = linear_regression.weights
    features_cnt = len(weights) - 1
    feature_importance = list(zip(range(features_cnt), get_feature_importance(linear_regression)))
    f_sort = sorted(feature_importance, key=lambda t: -t[1])
    indices = list(map(lambda t: t[0], f_sort))
    return indices
