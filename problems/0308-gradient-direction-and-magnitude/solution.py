
import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
    g = 0

    for l in gradient:
        g += l**2

    g = np.sqrt(g)

    if g == 0:
        direction = [0 for x in gradient]
    else:
        direction = [x/g for x in gradient]

    descent_direction = [-y for y in direction]

    da = {
        "magnitude": g,
        "direction": direction,
        "descent_direction": descent_direction
    }

    return da