import numpy as np
import math

def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    learning_rate: float,
    epochs: int
) -> tuple[np.ndarray, float, list[float]]:

    m = len(features)
    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []
    alpha = learning_rate

    for _ in range(epochs):
        predictions = []

        # Forward Pass
        for x in features:
            z = np.dot(x, weights) + bias
            p = 1 / (1 + math.exp(-z))
            predictions.append(p)

        # MSE
        mse = sum(
            (p - y)**2
            for p, y in zip(predictions, labels)
        ) / m

        mse_values.append(mse)

        # Backpropagation
        dw = np.zeros_like(weights)
        db = 0.0

        for x, p, y in zip(features, predictions, labels):
            delta = 2 * (p - y) * p * (1 - p)
            dw += delta * x
            db += delta

        dw = dw / m
        db = db / m

        # Gradient Descent
        weights = weights - alpha * dw
        bias = bias - alpha * db

    # Rounding
    weights = np.round(weights, 4)
    bias = round(float(bias), 4)
    mse_values = [round(float(v), 4) for v in mse_values]

    return weights, bias, mse_values