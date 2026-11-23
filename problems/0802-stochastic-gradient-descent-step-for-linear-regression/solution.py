import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    up_weights=weights.copy()
    n_samples = len(X)
    for i in range(n_iter):
        idx=i%n_samples
        x_i=X[idx]
        y_i=y[idx]
        prediction=np.dot(x_i,up_weights)
        l_i=(prediction-y_i)**2
        gradient=2*(prediction-y_i)*x_i
        up_weights=up_weights-learning_rate*gradient

     
    return up_weights
