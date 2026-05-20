import numpy as np


class Perceptron:
    def __init__(self, n_features, init_type="small_random", random_state=42):
        self.n_features = n_features
        self.init_type = init_type
        self.rng = np.random.default_rng(random_state)
        self.w = self._init_weights()
        self.b = 0.0
        self.train_losses = []
        self.val_losses = []

    def _init_weights(self):
        if self.init_type == "zeros":
            return np.zeros(self.n_features)
        if self.init_type == "large_random":
            return self.rng.normal(0.0, 10.0, self.n_features)
        return self.rng.normal(0.0, 0.01, self.n_features)

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        return self.sigmoid(X @ self.w + self.b)

    @staticmethod
    def compute_loss(y_true, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))

    def fit(self, X_train, y_train, X_val, y_val, epochs, lr, batch_size):
        n_samples = X_train.shape[0]

        for _ in range(epochs):
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                y_pred = self.forward(X_batch)
                error = y_pred - y_batch

                dw = X_batch.T @ error / X_batch.shape[0]
                db = np.mean(error)

                self.w -= lr * dw
                self.b -= lr * db

            self.train_losses.append(self.compute_loss(y_train, self.forward(X_train)))
            self.val_losses.append(self.compute_loss(y_val, self.forward(X_val)))

        return self

    def predict(self, X):
        return (self.forward(X) >= 0.5).astype(int)
