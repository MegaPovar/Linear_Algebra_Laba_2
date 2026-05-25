import numpy as np


# Доп 1
def flip_labels(y, noise, rng):
    y_noisy = y.copy()
    mask = rng.random(y_noisy.shape[0]) < noise
    y_noisy[mask] = 1 - y_noisy[mask]
    return y_noisy


def generate_linear_data(
    n_samples=500,
    centers=((-2.0, -2.0), (2.0, 2.0)),
    covariance=((0.6, 0.0), (0.0, 0.6)),
    noise=0.0,
    random_state=42,
):
    rng = np.random.default_rng(random_state)
    first_count = n_samples // 2
    second_count = n_samples - first_count

    X0 = rng.multivariate_normal(centers[0], covariance, first_count)
    X1 = rng.multivariate_normal(centers[1], covariance, second_count)
    X = np.vstack((X0, X1))
    y = np.array([0] * first_count + [1] * second_count)

    indices = rng.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, flip_labels(y, noise, rng)


def generate_xor_data(n_samples=500, spread=0.35, noise=0.0, random_state=42):
    rng = np.random.default_rng(random_state)
    centers = np.array(
        [
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]
    )
    labels = np.array([0, 1, 1, 0])

    counts = np.full(len(centers), n_samples // len(centers))
    counts[: n_samples % len(centers)] += 1
    corner_indices = np.repeat(np.arange(len(centers)), counts)
    corner_indices = rng.permutation(corner_indices)
    X = centers[corner_indices] + rng.normal(0.0, spread, size=(n_samples, 2))
    y = labels[corner_indices]

    return X, flip_labels(y, noise, rng)


def generate_circle_data(n_samples=500, radius=1.0, outer_radius=1.8, noise=0.0, random_state=42):
    rng = np.random.default_rng(random_state)
    inner_count = n_samples // 2
    outer_count = n_samples - inner_count

    inner_angles = rng.uniform(0.0, 2.0 * np.pi, inner_count)
    inner_distances = radius * np.sqrt(rng.uniform(0.0, 1.0, inner_count))
    inner_points = np.column_stack(
        (
            inner_distances * np.cos(inner_angles),
            inner_distances * np.sin(inner_angles),
        )
    )

    outer_angles = rng.uniform(0.0, 2.0 * np.pi, outer_count)
    outer_distances = np.sqrt(rng.uniform(radius**2, outer_radius**2, outer_count))
    outer_points = np.column_stack(
        (
            outer_distances * np.cos(outer_angles),
            outer_distances * np.sin(outer_angles),
        )
    )

    X = np.vstack((inner_points, outer_points))
    y = np.array([0] * inner_count + [1] * outer_count)

    indices = rng.permutation(n_samples)
    X = X[indices]
    y = y[indices]

    return X, flip_labels(y, noise, rng)
