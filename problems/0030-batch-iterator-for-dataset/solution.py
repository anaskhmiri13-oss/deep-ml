import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    batch=[]
    for i in range(0,len(X),batch_size):
        j=[]
        X_batch=X[i : i + batch_size]
        j.append(X_batch)
        if y is not None : 
            y_batch=y[i : i + batch_size]
            j.append(y_batch)
        batch.append(j)
        

    return batch 
        