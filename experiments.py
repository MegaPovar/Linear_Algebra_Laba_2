from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE, RESULTS_DIR
from data import split_and_standardize
from metrics import accuracy, rounded
from model import Perceptron
from synthetic_data import generate_circle_data, generate_linear_data, generate_xor_data
from utils import write_rows
from visualization import plot_dataset_decision_boundary, plot_loss, plot_metric


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
):
    model = Perceptron(
        n_features=X_train.shape[1],
        init_type=init_type,
        loss_type=loss_type,
        l2_lambda=l2_lambda,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, X_test, y_test, epochs=epochs, lr=lr, batch_size=batch_size)

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
