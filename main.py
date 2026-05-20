from config import RESULTS_DIR
from data import prepare_data
from experiments import (
    run_batch_size_experiment,
    run_initialization_experiment,
    run_learning_rate_experiment,
    train_model,
)
from metrics import rounded
from utils import print_table, write_rows
from visualization import plot_decision_boundary, plot_loss


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = prepare_data()

    base_model, base_train_acc, base_test_acc = train_model(X_train, y_train, X_test, y_test)

    base_rows = [
        {
            "train_accuracy": rounded(base_train_acc),
            "test_accuracy": rounded(base_test_acc),
            "final_train_loss": rounded(base_model.train_losses[-1], 6),
            "final_test_loss": rounded(base_model.val_losses[-1], 6),
        }
    ]

    write_rows(
        RESULTS_DIR / "base_metrics.csv",
        ["train_accuracy", "test_accuracy", "final_train_loss", "final_test_loss"],
        base_rows,
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

    print_table("Base metrics", base_rows)
    print_table("Learning rate experiment", learning_rate_rows)
    print_table("Batch size experiment", batch_size_rows)
    print_table("Initialization experiment", initialization_rows)


if __name__ == "__main__":
    main()
