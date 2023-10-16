import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """

    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.uniform(-0.1, 0.1, (in_features + 1, out_features))
        self.x = None
        self.g = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        if len(x.shape) == 1:
            x = np.hstack((x, np.ones(1, dtype=x.dtype)))
            self.x = x
            return np.matmul(x, self.W)
        else:
            x = np.hstack((x, np.ones((x.shape[0], 1), dtype=x.dtype)))
            self.x = x
            return np.matmul(x, self.W)

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        if len(self.x.shape) == 1:
            self.g = np.zeros((self.in_features + 1, self.out_features), dtype=self.x.dtype)
            self.g = np.outer(self.x, d)
            # next_d = np.sum(self.g, axis=1)
            next_d = np.zeros(self.in_features, dtype=d.dtype)
            for i in range(self.in_features):
                next_d[i] = np.sum(d * self.W[i])
            return next_d
        else:
            self.g = np.zeros((self.x.shape[0], self.in_features + 1, self.out_features), dtype=self.x.dtype)
            for k in range(self.x.shape[0]):
                self.g[k] = np.outer(self.x[k], d[k])
            # next_d = np.sum(self.g, axis=2)
            # next_d = np.zeros((self.x.shape[0], self.in_features), dtype=d.dtype)
            next_d = np.matmul(d, self.W[:-1].T)
            # for i in range(self.x.shape[0]):
            #     for j in range(self.in_features):
            #         next_d[i][j] = np.sum(d[i] * self.W[j])
            return next_d

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        if len(self.g.shape) == 3:
            self.W = self.W - alpha * np.sum(self.g, axis=0)
        else:
            self.W = self.W - alpha * self.g


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """

    def __init__(self):
        self.x = None
        self.g = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x
        return np.where(x > 0, x, 0)

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.g = (self.x >= 0) * d
        return self.g


# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и
            описывающий слои нейронной сети.
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        self.unique_classes = np.sort(np.unique(y))
        self.rev_mp = {u_class: i for i, u_class in enumerate(self.unique_classes)}
        cnt_samples = len(X)
        mapped_y = np.array([self.rev_mp[y_i] for y_i in y])
        for _ in range(self.epochs):
            perm = np.random.permutation(cnt_samples)
            batches = np.array_split(perm, self.batch_size)
            for batch in batches:
                normed_res = self._predict(X[batch])
                derivative = normed_res
                for i in range(len(batch)):
                    derivative[i][mapped_y[batch[i]]] -= 1
                for module in reversed(self.modules):
                    derivative = module.backward(derivative)
                for module in self.modules:
                    module.update(self.alpha)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        for module in self.modules:
            X = module.forward(X)
        res = np.exp(X)
        normed_res = (res.T / np.sum(res, axis=1)).T
        return normed_res

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        return self._predict(X)

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


# Task 3

classifier_moons = MLPClassifier(
    [
        Linear(2, 2),

        Linear(2, 2),

        Linear(2, 2)
    ]
)  # Нужно указать гиперпараметры
classifier_blobs = MLPClassifier(
    [
        Linear(2, 3)
    ]
)  # Нужно указать гиперпараметры


# Task 4
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(32 * 32 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.model(self.flatten(x)), dim=1)

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] +"neural_networks.pth"`, где "neural_networks.pth" - имя файла сохраненной модели `
        """
        with open(__file__[:-7] + "neural_networks.pth", 'rb') as f:
            self.model = torch.load(f)

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        with open(__file__[:-7] + "neural_networks.pth", 'wb') as f:
            torch.save(self.model, f)


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    res = model(X)
    return F.cross_entropy(res, y)
