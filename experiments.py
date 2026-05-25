import numpy as np

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE, RESULTS_DIR
from data import split_and_standardize
from metrics import accuracy, classification_metrics, roc_auc, roc_curve_points, rounded
from model import Perceptron
from synthetic_data import generate_circle_data, generate_linear_data, generate_xor_data
from utils import write_rows
from visualization import (
    plot_cv_results,
    plot_dataset_decision_boundary,
    plot_loss,
    plot_metric,
    plot_misclassified_points,
    plot_roc_curve,
)


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    init_type="small_random",
    loss_type="cross_entropy",
    l2_lambda=0.0,
    momentum_beta=0.0,
):
    model = Perceptron(
        n_features=X_train.shape[1],
        init_type=init_type,
        loss_type=loss_type,
        l2_lambda=l2_lambda,
        random_state=RANDOM_STATE,
    )
    model.fit(
        X_train,
        y_train,
        X_test,
        y_test,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        momentum_beta=momentum_beta,
    )

    train_acc = accuracy(y_train, model.predict(X_train))
    test_acc = accuracy(y_test, model.predict(X_test))

    return model, train_acc, test_acc


def convergence_epoch(losses, progress=0.95):
    first_loss = losses[0]
    final_loss = losses[-1]
    target_loss = first_loss - progress * (first_loss - final_loss)

    for epoch, loss in enumerate(losses, start=1):
        if loss <= target_loss:
            return epoch

    return len(losses)


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


# Доп 1
def run_custom_data_experiment():
    datasets = [
        ("linear", "Linear Gaussian clouds", generate_linear_data(noise=0.03, random_state=RANDOM_STATE)),
        ("xor", "XOR data", generate_xor_data(noise=0.03, random_state=RANDOM_STATE)),
        ("circle", "Circle data", generate_circle_data(noise=0.03, random_state=RANDOM_STATE)),
    ]
    rows = []
    histories = []

    for dataset_name, title, (X, y) in datasets:
        X_train, X_test, y_train, y_test = split_and_standardize(X, y)
        model, train_acc, test_acc = train_model(X_train, y_train, X_test, y_test)

        rows.append(
            {
                "dataset": dataset_name,
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
            }
        )
        histories.append((dataset_name, model.train_losses, model.val_losses))
        plot_dataset_decision_boundary(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
            title,
            RESULTS_DIR / f"custom_{dataset_name}_boundary.png",
        )

    write_rows(
        RESULTS_DIR / "custom_data_results.csv",
        ["dataset", "train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        rows,
    )
    plot_loss(histories, "Custom data generator experiment", RESULTS_DIR / "custom_data_loss.png")
    write_custom_data_conclusions(rows)

    return rows


def write_custom_data_conclusions(rows):
    row_by_dataset = {row["dataset"]: row for row in rows}
    lines = [
        "Выводы по собственному генератору данных",
        "",
        (
            "1. На линейно разделимых гауссовых облаках перцептрон работает успешно: "
            f"accuracy на тестовой выборке = {row_by_dataset['linear']['test_accuracy']}."
        ),
        (
            "2. На XOR качество низкое, потому что классы нельзя разделить одной прямой: "
            f"accuracy на тестовой выборке = {row_by_dataset['xor']['test_accuracy']}."
        ),
        (
            "3. На данных вида окружности перцептрон также ограничен линейной границей: "
            f"accuracy на тестовой выборке = {row_by_dataset['circle']['test_accuracy']}."
        ),
        "",
        (
            "Итог: однослойный перцептрон с двумя входами и сигмоидой строит линейную "
            "разделяющую границу. Поэтому он подходит для линейно разделимых данных, "
            "но плохо решает задачи, где нужна нелинейная граница."
        ),
    ]

    (RESULTS_DIR / "custom_data_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


# Доп 2
def run_loss_and_regularization_experiment(X_train, y_train, X_test, y_test):
    loss_rows = run_hinge_loss_experiment(X_train, y_train, X_test, y_test)
    l2_rows = run_l2_regularization_experiment(X_train, y_train, X_test, y_test)
    write_loss_and_regularization_conclusions(loss_rows, l2_rows)

    return loss_rows, l2_rows


def run_hinge_loss_experiment(X_train, y_train, X_test, y_test):
    rows = []
    histories = []
    configs = [
        ("cross_entropy", "cross entropy"),
        ("hinge", "hinge loss"),
    ]

    for loss_type, label in configs:
        model, train_acc, test_acc = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            loss_type=loss_type,
        )
        rows.append(
            {
                "loss": label,
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
                "convergence_epoch": convergence_epoch(model.train_losses),
                "weight_norm": rounded((model.w @ model.w) ** 0.5, 6),
            }
        )
        histories.append((label, model.train_losses, model.val_losses))

    write_rows(
        RESULTS_DIR / "loss_function_results.csv",
        [
            "loss",
            "train_accuracy",
            "test_accuracy",
            "final_train_loss",
            "final_test_loss",
            "convergence_epoch",
            "weight_norm",
        ],
        rows,
    )
    plot_loss(histories, "Cross-entropy and hinge loss", RESULTS_DIR / "loss_function_comparison.png")

    return rows


def run_l2_regularization_experiment(X_train, y_train, X_test, y_test):
    rows = []
    histories = []

    for l2_lambda in [0.0, 0.001, 0.01, 0.1, 1.0]:
        model, train_acc, test_acc = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            l2_lambda=l2_lambda,
        )
        rows.append(
            {
                "lambda": l2_lambda,
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
                "weight_1": rounded(model.w[0], 6),
                "weight_2": rounded(model.w[1], 6),
                "bias": rounded(model.b, 6),
                "weight_norm": rounded((model.w @ model.w) ** 0.5, 6),
            }
        )
        histories.append((f"lambda={l2_lambda}", model.train_losses, model.val_losses))

    write_rows(
        RESULTS_DIR / "l2_regularization_results.csv",
        [
            "lambda",
            "train_accuracy",
            "test_accuracy",
            "final_train_loss",
            "final_test_loss",
            "weight_1",
            "weight_2",
            "bias",
            "weight_norm",
        ],
        rows,
    )
    plot_loss(histories, "L2 regularization loss", RESULTS_DIR / "l2_regularization_loss.png")
    plot_metric(
        rows,
        "lambda",
        ["weight_norm"],
        "Weight norm and L2 regularization",
        RESULTS_DIR / "l2_weight_norm.png",
        x_label="lambda",
        y_label="weight norm",
    )
    plot_metric(
        rows,
        "lambda",
        ["weight_1", "weight_2"],
        "Weights and L2 regularization",
        RESULTS_DIR / "l2_weights.png",
        x_label="lambda",
        y_label="weight value",
    )
    plot_metric(
        rows,
        "lambda",
        ["train_accuracy", "test_accuracy"],
        "Accuracy and L2 regularization",
        RESULTS_DIR / "l2_accuracy.png",
        x_label="lambda",
        y_label="accuracy",
    )

    return rows


def write_loss_and_regularization_conclusions(loss_rows, l2_rows):
    ce_row = next(row for row in loss_rows if row["loss"] == "cross entropy")
    hinge_row = next(row for row in loss_rows if row["loss"] == "hinge loss")
    best_l2 = max(l2_rows, key=lambda row: row["test_accuracy"])
    strongest_l2 = max(l2_rows, key=lambda row: row["lambda"])

    lines = [
        "Выводы по hinge loss и L2-регуляризации",
        "",
        (
            "1. Кросс-энтропия на базовом наборе дала test accuracy = "
            f"{ce_row['test_accuracy']}, hinge loss = {hinge_row['test_accuracy']}."
        ),
        (
            "2. Hinge loss оптимизирует линейный отступ, поэтому значение loss имеет "
            "другой масштаб и напрямую не сравнивается с кросс-энтропией. "
            f"Условная эпоха сходимости: {ce_row['convergence_epoch']} для cross-entropy "
            f"и {hinge_row['convergence_epoch']} для hinge loss."
        ),
        (
            "3. L2-регуляризация уменьшает норму весов. При lambda = "
            f"{strongest_l2['lambda']} норма весов стала {strongest_l2['weight_norm']}."
        ),
        (
            "4. Лучшее качество на тестовой выборке в эксперименте получилось при lambda = "
            f"{best_l2['lambda']}: test accuracy = {best_l2['test_accuracy']}."
        ),
        "",
        (
            "Итог: hinge loss можно использовать для линейной классификации, но для "
            "вероятностной интерпретации удобнее кросс-энтропия. L2-регуляризация "
            "ограничивает рост весов и может улучшать обобщение, если коэффициент не слишком большой."
        ),
    ]

    (RESULTS_DIR / "loss_regularization_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


# Доп 3
def run_metrics_error_analysis(X_train, y_train, X_test, y_test):
    model, _, _ = train_model(X_train, y_train, X_test, y_test)
    y_pred = model.predict(X_test)
    scores = model.forward(X_test)
    metrics = classification_metrics(y_test, y_pred)
    roc_points = roc_curve_points(y_test, scores)
    auc_value = roc_auc(roc_points)

    rows = [
        {
            "accuracy": rounded(metrics["accuracy"]),
            "precision": rounded(metrics["precision"]),
            "recall": rounded(metrics["recall"]),
            "f1_score": rounded(metrics["f1_score"]),
            "roc_auc": rounded(auc_value),
            "true_positive": metrics["true_positive"],
            "true_negative": metrics["true_negative"],
            "false_positive": metrics["false_positive"],
            "false_negative": metrics["false_negative"],
            "error_count": int(metrics["false_positive"] + metrics["false_negative"]),
        }
    ]

    write_rows(
        RESULTS_DIR / "metrics_error_analysis.csv",
        [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
            "true_positive",
            "true_negative",
            "false_positive",
            "false_negative",
            "error_count",
        ],
        rows,
    )

    write_misclassified_points(X_test, y_test, y_pred, scores)
    write_roc_points(roc_points)
    plot_roc_curve(roc_points, auc_value, RESULTS_DIR / "roc_curve.png")
    plot_misclassified_points(model, X_test, y_test, y_pred, RESULTS_DIR / "misclassified_points.png")
    write_metrics_error_conclusions(rows[0])

    return rows


def write_misclassified_points(X_test, y_test, y_pred, scores):
    rows = []

    for index, (point, true_label, pred_label, score) in enumerate(zip(X_test, y_test, y_pred, scores)):
        if true_label != pred_label:
            rows.append(
                {
                    "index": index,
                    "feature_1": rounded(point[0], 6),
                    "feature_2": rounded(point[1], 6),
                    "true_label": int(true_label),
                    "predicted_label": int(pred_label),
                    "probability_class_1": rounded(score, 6),
                }
            )

    write_rows(
        RESULTS_DIR / "misclassified_points.csv",
        ["index", "feature_1", "feature_2", "true_label", "predicted_label", "probability_class_1"],
        rows,
    )


def write_roc_points(roc_points):
    rows = []

    for fpr, tpr, threshold in roc_points:
        rows.append(
            {
                "false_positive_rate": rounded(fpr, 6),
                "true_positive_rate": rounded(tpr, 6),
                "threshold": rounded(threshold, 6) if np.isfinite(threshold) else str(threshold),
            }
        )

    write_rows(
        RESULTS_DIR / "roc_points.csv",
        ["false_positive_rate", "true_positive_rate", "threshold"],
        rows,
    )


def write_metrics_error_conclusions(row):
    lines = [
        "Выводы по метрикам и анализу ошибок",
        "",
        (
            "1. На тестовой выборке получены метрики: "
            f"accuracy = {row['accuracy']}, precision = {row['precision']}, "
            f"recall = {row['recall']}, F1-score = {row['f1_score']}, "
            f"ROC-AUC = {row['roc_auc']}."
        ),
        (
            "2. Ошибочных объектов на тестовой выборке: "
            f"{row['error_count']} из {row['true_positive'] + row['true_negative'] + row['error_count']}."
        ),
        (
            "3. Неправильно классифицированные точки находятся рядом с разделяющей прямой "
            "или в области пересечения классов, где линейной модели труднее уверенно выбрать класс."
        ),
        "",
        "Итог: дополнительных метрик достаточно, чтобы оценить не только долю правильных ответов, но и баланс ошибок разных типов.",
    ]

    (RESULTS_DIR / "metrics_error_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


# Доп 4
def run_momentum_experiment(X_train, y_train, X_test, y_test):
    rows = []
    histories = []

    for beta, label in [(0.0, "SGD"), (0.5, "beta=0.5"), (0.9, "beta=0.9"), (0.99, "beta=0.99")]:
        model, train_acc, test_acc = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            momentum_beta=beta,
        )
        rows.append(
            {
                "method": label,
                "beta": beta,
                "train_accuracy": rounded(train_acc),
                "test_accuracy": rounded(test_acc),
                "final_train_loss": rounded(model.train_losses[-1], 6),
                "final_test_loss": rounded(model.val_losses[-1], 6),
                "convergence_epoch": convergence_epoch(model.train_losses),
            }
        )
        histories.append((label, model.train_losses, model.val_losses))

    write_rows(
        RESULTS_DIR / "momentum_results.csv",
        [
            "method",
            "beta",
            "train_accuracy",
            "test_accuracy",
            "final_train_loss",
            "final_test_loss",
            "convergence_epoch",
        ],
        rows,
    )
    plot_loss(histories, "SGD and momentum loss", RESULTS_DIR / "momentum_loss.png")
    write_momentum_conclusions(rows)

    return rows


def write_momentum_conclusions(rows):
    sgd_row = next(row for row in rows if row["method"] == "SGD")
    best_row = min(rows, key=lambda row: row["convergence_epoch"])

    lines = [
        "Выводы по градиентному спуску с momentum",
        "",
        (
            "1. Обычный SGD достиг условной сходимости за "
            f"{sgd_row['convergence_epoch']} эпох, test accuracy = {sgd_row['test_accuracy']}."
        ),
        (
            "2. Самое быстрое обучение в эксперименте получилось при "
            f"{best_row['method']}: {best_row['convergence_epoch']} эпох, "
            f"test accuracy = {best_row['test_accuracy']}."
        ),
        (
            "3. Большой импульс может ускорять движение по направлению устойчивого градиента, "
            "но слишком большое значение иногда дает колебания и менее плавный loss."
        ),
        "",
        "Итог: momentum может ускорить сходимость, но коэффициент beta нужно подбирать экспериментально.",
    ]

    (RESULTS_DIR / "momentum_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


# Доп 5
def run_cross_validation_experiment(X_train, y_train, X_test, y_test):
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    batch_sizes = [16, 32, 64, 128]
    folds = stratified_kfold_indices(y_train, n_splits=5)
    rows = []

    for lr in learning_rates:
        for batch_size in batch_sizes:
            fold_accuracies = []

            for fold_index, val_indices in enumerate(folds, start=1):
                train_indices = np.setdiff1d(np.arange(len(y_train)), val_indices)
                X_fold_train = X_train[train_indices]
                y_fold_train = y_train[train_indices]
                X_fold_val = X_train[val_indices]
                y_fold_val = y_train[val_indices]

                _, _, val_acc = train_model(
                    X_fold_train,
                    y_fold_train,
                    X_fold_val,
                    y_fold_val,
                    lr=lr,
                    batch_size=batch_size,
                )
                fold_accuracies.append(val_acc)

            rows.append(
                {
                    "learning_rate": lr,
                    "batch_size": batch_size,
                    "fold_1": rounded(fold_accuracies[0]),
                    "fold_2": rounded(fold_accuracies[1]),
                    "fold_3": rounded(fold_accuracies[2]),
                    "fold_4": rounded(fold_accuracies[3]),
                    "fold_5": rounded(fold_accuracies[4]),
                    "mean_accuracy": rounded(np.mean(fold_accuracies)),
                    "std_accuracy": rounded(np.std(fold_accuracies)),
                }
            )

    best_row = max(rows, key=lambda row: (row["mean_accuracy"], -row["std_accuracy"]))
    final_model, final_train_acc, final_test_acc = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        lr=best_row["learning_rate"],
        batch_size=best_row["batch_size"],
    )
    final_rows = [
        {
            "learning_rate": best_row["learning_rate"],
            "batch_size": best_row["batch_size"],
            "train_accuracy": rounded(final_train_acc),
            "test_accuracy": rounded(final_test_acc),
            "final_train_loss": rounded(final_model.train_losses[-1], 6),
            "final_test_loss": rounded(final_model.val_losses[-1], 6),
        }
    ]

    write_rows(
        RESULTS_DIR / "cross_validation_results.csv",
        [
            "learning_rate",
            "batch_size",
            "fold_1",
            "fold_2",
            "fold_3",
            "fold_4",
            "fold_5",
            "mean_accuracy",
            "std_accuracy",
        ],
        rows,
    )
    write_rows(
        RESULTS_DIR / "cross_validation_best_model.csv",
        ["learning_rate", "batch_size", "train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        final_rows,
    )
    plot_cv_results(rows, RESULTS_DIR / "cross_validation_results.png")
    plot_loss(
        [("best CV model", final_model.train_losses, final_model.val_losses)],
        "Best CV model loss",
        RESULTS_DIR / "cross_validation_best_loss.png",
    )
    write_cross_validation_conclusions(best_row, final_rows[0])

    return rows, final_rows


def stratified_kfold_indices(y, n_splits=5):
    rng = np.random.default_rng(RANDOM_STATE)
    folds = [[] for _ in range(n_splits)]

    for class_label in np.unique(y):
        class_indices = np.where(y == class_label)[0]
        class_indices = rng.permutation(class_indices)
        class_parts = np.array_split(class_indices, n_splits)

        for fold, part in zip(folds, class_parts):
            fold.extend(part.tolist())

    return [np.array(sorted(fold)) for fold in folds]


def write_cross_validation_conclusions(best_row, final_row):
    lines = [
        "Выводы по кросс-валидации и подбору гиперпараметров",
        "",
        (
            "1. Проведена 5-кратная стратифицированная кросс-валидация по скорости обучения "
            "и размеру батча."
        ),
        (
            "2. Лучшая комбинация: "
            f"eta = {best_row['learning_rate']}, batch_size = {best_row['batch_size']}. "
            f"Средняя accuracy = {best_row['mean_accuracy']}, std = {best_row['std_accuracy']}."
        ),
        (
            "3. Финальная модель обучена на всех обучающих данных с выбранными параметрами. "
            f"Train accuracy = {final_row['train_accuracy']}, test accuracy = {final_row['test_accuracy']}."
        ),
        "",
        "Итог: кросс-валидация позволяет выбрать гиперпараметры не по одному случайному разбиению, а по среднему качеству на нескольких фолдах.",
    ]

    (RESULTS_DIR / "cross_validation_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")
