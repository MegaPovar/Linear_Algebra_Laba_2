from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


RESULTS_DIR = Path("results")
RANDOM_STATE = 42


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


def prepare_data():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=RANDOM_STATE,
        n_clusters_per_class=1,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled, y_train, y_test


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def rounded(value, digits=4):
    return round(float(value), digits)


def write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_loss(history_items, title, path):
    plt.figure(figsize=(9, 6))
    for label, train_losses, val_losses in history_items:
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f"{label}: train")
        plt.plot(epochs, val_losses, linestyle="--", label=f"{label}: test")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_decision_boundary(model, X_train, y_train, X_test, y_test, path):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", alpha=0.65, label="train")
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        alpha=0.95,
        marker="x",
        label="test",
    )

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-12:
        ys = -(model.w[0] * xs + model.b) / model.w[1]
        plt.plot(xs, ys, color="black", linewidth=2, label="w^T x + b = 0")
    else:
        x_line = -model.b / model.w[0]
        plt.axvline(x_line, color="black", linewidth=2, label="w^T x + b = 0")

    plt.xlabel("Feature 1 (standardized)")
    plt.ylabel("Feature 2 (standardized)")
    plt.title("Decision boundary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def train_model(X_train, y_train, X_test, y_test, epochs=100, lr=0.1, batch_size=32, init_type="small_random"):
    model = Perceptron(
        n_features=X_train.shape[1],
        init_type=init_type,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, batch_size=batch_size)

    train_acc = accuracy(y_train, model.predict(X_train))
    test_acc = accuracy(y_test, model.predict(X_test))

    return model, train_acc, test_acc


def run_learning_rate_experiment(X_train, y_train, X_test, y_test):
    rows = []
    histories = []

    for lr in [0.001, 0.01, 0.5, 1.0]:
        model, train_acc, test_acc = train_model(X_train, y_train, X_test, y_test, lr=lr)
        rows.append(
            {
                "learning_rate": lr,
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
            }
        )
        histories.append((f"lr={lr}", model.train_losses, model.val_losses))

    write_rows(
        RESULTS_DIR / "learning_rate_results.csv",
        ["learning_rate", "train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        rows,
    )
    plot_loss(histories, "Learning rate experiment", RESULTS_DIR / "learning_rate_experiment.png")

    return rows


def run_batch_size_experiment(X_train, y_train, X_test, y_test):
    rows = []
    histories = []

    for batch_size in [1, 16, 64, 256]:
        model, train_acc, test_acc = train_model(X_train, y_train, X_test, y_test, batch_size=batch_size)
        rows.append(
            {
                "batch_size": batch_size,
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
            }
        )
        histories.append((f"batch={batch_size}", model.train_losses, model.val_losses))

    write_rows(
        RESULTS_DIR / "batch_size_results.csv",
        ["batch_size", "train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        rows,
    )
    plot_loss(histories, "Batch size experiment", RESULTS_DIR / "batch_size_experiment.png")

    return rows


def run_initialization_experiment(X_train, y_train, X_test, y_test):
    rows = []
    histories = []
    labels = {
        "zeros": "zero",
        "small_random": "small random",
        "large_random": "large random",
    }

    for init_type in ["zeros", "small_random", "large_random"]:
        model, train_acc, test_acc = train_model(X_train, y_train, X_test, y_test, init_type=init_type)
        rows.append(
            {
                "initialization": labels[init_type],
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
            }
        )
        histories.append((labels[init_type], model.train_losses, model.val_losses))

    write_rows(
        RESULTS_DIR / "initialization_results.csv",
        ["initialization", "train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        rows,
    )
    plot_loss(histories, "Initialization experiment", RESULTS_DIR / "initialization_experiment.png")

    return rows


def print_table(title, rows):
    print(f"\n{title}")
    for row in rows:
        print(row)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_data()

    base_model, base_train_acc, base_test_acc = train_model(X_train, y_train, X_test, y_test)

    write_rows(
        RESULTS_DIR / "base_metrics.csv",
        ["train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        [
            {
                "train_accuracy": rounded(base_train_acc),
                "test_accuracy": rounded(base_test_acc),
                "final_train_loss": rounded(base_model.train_losses[-1], 6),
                "final_test_loss": rounded(base_model.val_losses[-1], 6),
            }
        ],
    )

    plot_loss(
        [("base", base_model.train_losses, base_model.val_losses)],
        "Base training loss",
        RESULTS_DIR / "base_loss.png",
    )
    plot_decision_boundary(
        base_model,
        X_train,
        y_train,
        X_test,
        y_test,
        RESULTS_DIR / "decision_boundary.png",
    )

    learning_rate_rows = run_learning_rate_experiment(X_train, y_train, X_test, y_test)
    batch_size_rows = run_batch_size_experiment(X_train, y_train, X_test, y_test)
    initialization_rows = run_initialization_experiment(X_train, y_train, X_test, y_test)

    print_table("Base metrics", [
        {
            "train_accuracy": rounded(base_train_acc),
            "test_accuracy": rounded(base_test_acc),
            "final_train_loss": rounded(base_model.train_losses[-1], 6),
            "final_test_loss": rounded(base_model.val_losses[-1], 6),
        }
    ])
    print_table("Learning rate experiment", learning_rate_rows)
    print_table("Batch size experiment", batch_size_rows)
    print_table("Initialization experiment", initialization_rows)


if __name__ == "__main__":
    main()
