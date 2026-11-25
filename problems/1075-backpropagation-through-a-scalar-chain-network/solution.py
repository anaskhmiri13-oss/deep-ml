import numpy as np 
def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:

    activations = [a0]
    z_values = []
    for w_i, b_i in zip(w, b):
        z=float(w_i*activations[-1]+b_i)
        a=float(1/(1+np.exp(-z)))
        activations.append(a)
        z_values.append(z)
    dw = [0.0] * len(w)
    db = [0.0] * len(b)
    delta = 0.0
    for i in range(len(w)-1, -1, -1):
        a = activations[i+1]
        sig_prime = a * (1-a)
        if i == len(w)-1:
            dca = 2 * (a - y)
            delta = float(dca*sig_prime)
        else:
            delta = delta * w[i+1] * sig_prime

        dw[i] = delta * activations[i]
        db[i] = delta
    return {
        "dW": [round(v, 6) for v in dw],
        "dB": [round(v, 6) for v in db]
    }