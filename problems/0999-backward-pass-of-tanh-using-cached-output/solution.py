import numpy as np

def tanh_backward(h: np.ndarray, dh: np.ndarray) -> np.ndarray:
    dz=[]
    for h_i,dh_i in zip(h,dh):
        z=1.0-h_i**2
        z=z*dh_i
        dz.append(z)
    return dz 
