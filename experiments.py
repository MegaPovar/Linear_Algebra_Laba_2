from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE, RESULTS_DIR
from metrics import accuracy, rounded
from model import Perceptron
from utils import write_rows
from visualization import plot_loss


def train_model(
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    init_type="small_random",
):
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
