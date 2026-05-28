import numpy as np


def accuracy(y_true, y_pred):  # доля правильных
    return float(np.mean(y_true == y_pred))


def rounded(value, digits=4):
    return round(float(value), digits)


# Доп 3
def classification_metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # правильно нашли класс 1
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0  # точность среди моделей класса 1
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0 # из всех объектов класса 1, сколько нашли
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0  

    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def roc_curve_points(y_true, scores):  # строим точки ROC для разных порогов
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    positive_count = np.sum(y_true == 1)
    negative_count = np.sum(y_true == 0)
    points = []

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))

        tpr = tp / positive_count if positive_count > 0 else 0.0
        fpr = fp / negative_count if negative_count > 0 else 0.0
        points.append((fpr, tpr, threshold))

    return points


def roc_auc(points):  # площадь под ROC-кривой методом трапеций
    sorted_points = sorted(points, key=lambda point: point[0])
    fpr = np.array([point[0] for point in sorted_points])
    tpr = np.array([point[1] for point in sorted_points])
    return float(np.trapezoid(tpr, fpr))
