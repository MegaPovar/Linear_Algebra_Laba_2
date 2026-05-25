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


def plot_metric(rows, x_key, y_keys, title, path, x_label=None, y_label=None):
    plt.figure(figsize=(8, 6))
    x_values = [row[x_key] for row in rows]

    for y_key in y_keys:
        y_values = [row[y_key] for row in rows]
        plt.plot(x_values, y_values, marker="o", label=y_key)

    plt.xlabel(x_label or x_key)
    plt.ylabel(y_label or "value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# Доп 3
def plot_roc_curve(roc_points, auc_value, path):
    sorted_points = sorted(roc_points, key=lambda point: point[0])
    fpr = [point[0] for point in sorted_points]
    tpr = [point[1] for point in sorted_points]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_misclassified_points(model, X_test, y_test, y_pred, path):
    errors = y_test != y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_test[~errors, 0],
        X_test[~errors, 1],
        c=y_test[~errors],
        cmap="coolwarm",
        alpha=0.55,
        label="correct",
    )
    plt.scatter(
        X_test[errors, 0],
        X_test[errors, 1],
        c=y_test[errors],
        cmap="coolwarm",
        edgecolors="black",
        linewidths=1.4,
        marker="X",
        s=95,
        label="wrong",
    )

    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-12:
        ys = -(model.w[0] * xs + model.b) / model.w[1]
        plt.plot(xs, ys, color="black", linewidth=2, label="boundary")
    else:
        x_line = -model.b / model.w[0]
        plt.axvline(x_line, color="black", linewidth=2, label="boundary")

    plt.xlabel("Feature 1 (standardized)")
    plt.ylabel("Feature 2 (standardized)")
    plt.title("Misclassified test points")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# Доп 5
def plot_cv_results(rows, path):
    plt.figure(figsize=(9, 6))
    batch_sizes = sorted({row["batch_size"] for row in rows})

    for batch_size in batch_sizes:
        batch_rows = [row for row in rows if row["batch_size"] == batch_size]
        batch_rows = sorted(batch_rows, key=lambda row: row["learning_rate"])
        learning_rates = [row["learning_rate"] for row in batch_rows]
        mean_scores = [row["mean_accuracy"] for row in batch_rows]
        std_scores = [row["std_accuracy"] for row in batch_rows]
        plt.errorbar(learning_rates, mean_scores, yerr=std_scores, marker="o", capsize=4, label=f"batch={batch_size}")

    plt.xscale("log")
    plt.xlabel("Learning rate")
    plt.ylabel("Mean CV accuracy")
    plt.title("5-fold cross-validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
