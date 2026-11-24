import numpy as np

def derivative(function, x):
    match function:
        case 'square':
            return 2*x

        case 'sin':
            return np.cos(x)

        case 'cos':
            return -np.sin(x)

        case 'exp':
            return np.exp(x)
        case 'log' : 
            return 1/x

        case _:
            raise ValueError("Unknown function")
def fun(function,x):
    match function:
        case 'square':
            return x**2

        case 'sin':
            return np.sin(x)

        case 'cos':
            return np.cos(x)

        case 'exp':
            return np.exp(x)
        case 'log':
            return np.log(x)

        case _:
            raise ValueError("Unknown function")


def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    current_x = x
    gradient = 1
    for function in reversed(functions):
        f=derivative(function,current_x)
        gradient=gradient*f
        current_x=fun(function,current_x)
    return gradient
        
    
        


        
            
        
