import matplotlib.pyplot as plt
import numpy as np


def translate_title(title):
    titles = {
        "Base training loss": "Потери при базовом обучении",
        "Learning rate experiment": "Влияние скорости обучения",
        "Epoch count experiment": "Влияние количества эпох",
        "Batch size experiment": "Влияние размера батча",
        "Initialization experiment": "Влияние инициализации весов",
        "Custom data generator experiment": "Эксперимент с собственными данными",
        "Cross-entropy and hinge loss": "Сравнение cross-entropy и hinge loss",
        "L2 regularization loss": "Потери при L2-регуляризации",
        "Weight norm and L2 regularization": "Норма весов при L2-регуляризации",
        "Weights and L2 regularization": "Веса при L2-регуляризации",
        "Accuracy and L2 regularization": "Точность при L2-регуляризации",
        "SGD and momentum loss": "Сравнение SGD и momentum",
        "Best CV model loss": "Потери лучшей модели после кросс-валидации",
        "5-fold cross-validation": "5-кратная кросс-валидация",
        "ROC curve": "ROC-кривая",
        "Misclassified test points": "Ошибочные точки тестовой выборки",
        "Decision boundary": "Разделяющая граница",
        "Linear Gaussian clouds": "Линейно разделимые облака",
        "XOR data": "Данные XOR",
        "Circle data": "Данные окружности",
    }
    return titles.get(title, title)


def translate_label(label):
    labels = {
        "train": "обучение",
        "test": "тест",
        "random": "случайная модель",
        "correct": "верно",
        "wrong": "ошибка",
        "boundary": "граница",
        "value": "значение",
        "weight_norm": "норма весов",
        "weight_1": "вес 1",
        "weight_2": "вес 2",
        "train_accuracy": "точность на обучении",
        "test_accuracy": "точность на тесте",
        "accuracy": "точность",
        "weight value": "значение веса",
    }
    return labels.get(label, label)


# график изменения функции потерь
def plot_loss(history_items, title, path):
    plt.figure(figsize=(9, 6))
    for label, train_losses, val_losses in history_items:
        epochs = np.arange(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, label=f"{label}: обучение")
        plt.plot(epochs, val_losses, linestyle="--", label=f"{label}: тест")
    plt.xlabel("Эпоха")
    plt.ylabel("Значение функции потерь")
    plt.title(translate_title(title))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# график разделяющей прямой для базовой модели
def plot_decision_boundary(model, X_train, y_train, X_test, y_test, path):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", alpha=0.65, label="обучение")
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        alpha=0.95,
        marker="x",
        label="тест",
    )

    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-12:
        ys = -(model.w[0] * xs + model.b) / model.w[1]
        plt.plot(xs, ys, color="black", linewidth=2, label="граница w^T x + b = 0")
    else:
        x_line = -model.b / model.w[0]
        plt.axvline(x_line, color="black", linewidth=2, label="граница w^T x + b = 0")

    plt.xlabel("Признак 1 (стандартизированный)")
    plt.ylabel("Признак 2 (стандартизированный)")
    plt.title("Разделяющая граница")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# график границы для своих синтетических данных
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
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="coolwarm", alpha=0.6, label="обучение")
    plt.scatter(
        X_test[:, 0],
        X_test[:, 1],
        c=y_test,
        cmap="coolwarm",
        alpha=0.95,
        marker="x",
        label="тест",
    )

    plt.xlabel("Признак 1 (стандартизированный)")
    plt.ylabel("Признак 2 (стандартизированный)")
    plt.title(translate_title(title))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# график обычной метрики по разным параметрам
def plot_metric(rows, x_key, y_keys, title, path, x_label=None, y_label=None):
    plt.figure(figsize=(8, 6))
    x_values = [row[x_key] for row in rows]

    for y_key in y_keys:
        y_values = [row[y_key] for row in rows]
        plt.plot(x_values, y_values, marker="o", label=translate_label(y_key))

    plt.xlabel(x_label or x_key)
    plt.ylabel(translate_label(y_label or "value"))
    plt.title(translate_title(title))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# Доп 3
# график ROC-кривой
def plot_roc_curve(roc_points, auc_value, path):
    sorted_points = sorted(roc_points, key=lambda point: point[0])
    fpr = [point[0] for point in sorted_points]
    tpr = [point[1] for point in sorted_points]

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC-AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="случайная модель")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# график ошибочно классифицированных точек
def plot_misclassified_points(model, X_test, y_test, y_pred, path):
    errors = y_test != y_pred

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_test[~errors, 0],
        X_test[~errors, 1],
        c=y_test[~errors],
        cmap="coolwarm",
        alpha=0.55,
        label="верно",
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
        label="ошибка",
    )

    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    xs = np.linspace(x_min, x_max, 200)

    if abs(model.w[1]) > 1e-12:
        ys = -(model.w[0] * xs + model.b) / model.w[1]
        plt.plot(xs, ys, color="black", linewidth=2, label="граница")
    else:
        x_line = -model.b / model.w[0]
        plt.axvline(x_line, color="black", linewidth=2, label="граница")

    plt.xlabel("Признак 1 (стандартизированный)")
    plt.ylabel("Признак 2 (стандартизированный)")
    plt.title("Ошибочные точки тестовой выборки")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


# Доп 5
# график результатов кросс-валидации
def plot_cv_results(rows, path):
    plt.figure(figsize=(9, 6))
    batch_sizes = sorted({row["batch_size"] for row in rows})

    for batch_size in batch_sizes:
        batch_rows = [row for row in rows if row["batch_size"] == batch_size]
        batch_rows = sorted(batch_rows, key=lambda row: row["learning_rate"])
        learning_rates = [row["learning_rate"] for row in batch_rows]
        mean_scores = [row["mean_accuracy"] for row in batch_rows]
        std_scores = [row["std_accuracy"] for row in batch_rows]
        plt.errorbar(learning_rates, mean_scores, yerr=std_scores, marker="o", capsize=4, label=f"батч={batch_size}")

    plt.xscale("log")
    plt.xlabel("Скорость обучения")
    plt.ylabel("Средняя accuracy на CV")
    plt.title("5-кратная кросс-валидация")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
