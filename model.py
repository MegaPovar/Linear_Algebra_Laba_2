import numpy as np


class Perceptron:
    def __init__(
        self,
        n_features,
        init_type="small_random",
        loss_type="cross_entropy",
        l2_lambda=0.0,
        random_state=42,
    ):
        self.n_features = n_features
        self.init_type = init_type
        self.loss_type = loss_type
        self.l2_lambda = l2_lambda
        self.rng = np.random.default_rng(random_state)
        self.w = self._init_weights()  # создали веса
        self.b = 0.0  # начальное смещение нулевое
        self.train_losses = []  # сохраняем loss на обучении
        self.val_losses = []  # сохраняем loss на тесте/валидации

    def _init_weights(self):
        if self.init_type == "zeros":
            return np.zeros(self.n_features)
        if self.init_type == "large_random":
            return self.rng.normal(0.0, 10.0, self.n_features)
        return self.rng.normal(0.0, 0.01, self.n_features)  # маленькие случайные веса по умолчанию

    @staticmethod
    def sigmoid(z): # переводим линейный выход в вероятности 1 и 0, ограничивая z
        z = np.clip(z, -500, 500) 
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):  # прямой проход, получаем вероятности класса 1
        return self.sigmoid(X @ self.w + self.b)

    # Доп 2 функции потерь и регуляризации
    def decision_function(self, X):  # линейный выход без сигмоиды для hinge loss
        return X @ self.w + self.b

    def compute_cross_entropy_loss(self, y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return loss + self.l2_lambda * np.sum(self.w**2) / 2.0

    def compute_hinge_loss(self, y_true, scores): # доп функция потерь хинж лос
        y_signed = self._to_signed_labels(y_true)  # переводим 0/1 в -1/+1
        margins = y_signed * scores
        return np.mean(np.maximum(0.0, 1.0 - margins))

    def compute_loss(self, y_true, X):
        if self.loss_type == "hinge":
            return self.compute_hinge_loss(y_true, self.decision_function(X))
        return self.compute_cross_entropy_loss(y_true, self.forward(X))

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size, momentum_beta=0.0):  # главная обучался по мини батчам, доп 4 momentum
        n_samples = X_train.shape[0]
        velocity_w = np.zeros_like(self.w)
        velocity_b = 0.0

        for _ in range(epochs):
            indices = self.rng.permutation(n_samples)  # перемешиваем объекты перед каждой эпохой
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if self.loss_type == "hinge":
                    dw, db = self._hinge_gradients(X_batch, y_batch)
                else:
                    dw, db = self._cross_entropy_gradients(X_batch, y_batch)

                # Доп 4
                velocity_w = momentum_beta * velocity_w + dw  # если beta = 0, получится обычный SGD
                velocity_b = momentum_beta * velocity_b + db

                self.w -= lr * velocity_w
                self.b -= lr * velocity_b

            self.train_losses.append(self.compute_loss(y_train, X_train))  # фиксируем loss после эпохи
            self.val_losses.append(self.compute_loss(y_val, X_val))

        return self

    def _cross_entropy_gradients(self, X_batch, y_batch): # изменение весов и смещения для обычного обучения
        y_pred = self.forward(X_batch)
        error = y_pred - y_batch

        dw = X_batch.T @ error / X_batch.shape[0] + self.l2_lambda * self.w  # градиент + L2 штраф
        db = np.mean(error)

        return dw, db

    def _hinge_gradients(self, X_batch, y_batch):
        y_signed = self._to_signed_labels(y_batch)
        scores = self.decision_function(X_batch)
        active = y_signed * scores < 1.0  # обновляем только объекты внутри margin

        if not np.any(active):
            return np.zeros_like(self.w), 0.0

        dw = -(X_batch[active].T @ y_signed[active]) / X_batch.shape[0]
        db = -np.sum(y_signed[active]) / X_batch.shape[0]

        return dw, db

    @staticmethod
    def _to_signed_labels(y):
        return np.where(y == 1, 1.0, -1.0)

    def predict(self, X):  # переводим выход модели в классы 0 и 1
        if self.loss_type == "hinge":
            return (self.decision_function(X) >= 0.0).astype(int)
        return (self.forward(X) >= 0.5).astype(int)
