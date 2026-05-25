import matplotlib.pyplot as plt
import numpy as np


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


def plot_dataset_decision_boundary(model, X_train, y_train, X_test, y_test, title, path):
    plt.figure(figsize=(8, 6))

    x_min = min(X_train[:, 0].min(), X_test[:, 0].min()) - 0.5
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max()) + 0.5
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min()) - 0.5
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max()) + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = model.forward(grid).reshape(xx.shape)

    plt.contourf(xx, yy, probabilities, levels=20, cmap="coolwarm", alpha=0.25)
    plt.contour(xx, yy, probabilities, levels=[0.5], colors="black", linewidths=2)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", alpha=0.6, label="train")
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        alpha=0.95,
        marker="x",
        label="test",
    )

    plt.xlabel("Feature 1 (standardized)")
    plt.ylabel("Feature 2 (standardized)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
