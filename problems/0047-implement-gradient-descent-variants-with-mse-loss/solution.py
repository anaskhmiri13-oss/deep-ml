import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    up_weights=weights.copy()
    for i in range(n_epochs): 
        if method=='batch': 
            prediction=np.dot(X,up_weights)
            gradient = (2/len(X)) * (X.T @ (prediction-y))
            up_weights=up_weights-learning_rate*gradient
             


        elif method=='stochastic':
            for j in range(len(X)):
                x_i=X[j]
                y_i=y[j]
                prediction=np.dot(x_i,up_weights)
                gradient=2*(prediction-y_i)*x_i
                up_weights=up_weights-learning_rate*gradient


        elif method == 'mini_batch':
            for j in range(0,len(X),batch_size):
                X_batch=X[j:j+batch_size]
                y_batch=y[j:j+batch_size]
                prediction=np.dot(X_batch,up_weights)
                gradient = (2/len(X_batch)) * (X_batch.T @ (prediction-y_batch))
                up_weights=up_weights-learning_rate*gradient
        else:
            raise ValueError("Unknown method")
            
            

    
    return up_weights


