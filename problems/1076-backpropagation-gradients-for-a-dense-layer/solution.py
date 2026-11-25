import numpy as np

def dense_backward(a_prev: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray) -> dict:
    z=[]
    a=[]
    delta=1.0
    for j in range(len(W)):
        somme=b[j]
        for k in range(len(a_prev)):
            somme +=W[j][k]* a_prev[k]
        z.append(somme)
        a.append(1 / (1 + np.exp(-somme)))
    delta=[]
    for j in range(len(W)):
        d=2*(a[j]-y[j])*a[j]*(1-a[j])
        delta.append(d)
    db = delta

    dW = []
    for j in range(len(W)):
        ligne = []
        for k in range(len(a_prev)):
            d=delta[j]*a_prev[k]
            ligne.append(d)
        dW.append(ligne)
    da_prev = []

    for k in range(len(a_prev)):
        grad = 0.0
        for j in range(len(W)):
            grad += W[j][k] * delta[j]
        da_prev.append(grad)
    l={'dW':dW,'db':db,'da_prev':da_prev}
    return l 
    
    

