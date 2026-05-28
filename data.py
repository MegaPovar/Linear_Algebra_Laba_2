from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE


def split_and_standardize(X, y):  # делим выборку и сразу нормализуем признаки
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    mean = X_train.mean(axis=0)  # параметры считаем только по train
    std = X_train.std(axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std  # test нормализуем теми же mean/std

    return X_train_scaled, X_test_scaled, y_train, y_test


def prepare_data():  # базовый датасет из обязательной части
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=RANDOM_STATE,
        n_clusters_per_class=1,
    )

    return split_and_standardize(X, y)
