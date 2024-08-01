from HSL cimport hsl_to_rgb

import numpy as np

def colorize(z, stretch = 1):
    if stretch != 1:
        z = np.repeat(np.repeat(z,stretch, axis=0), stretch, axis=1)
    n,m = z.shape
    c = np.zeros((n,m,3))
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    idx = ~(np.isinf(z) + np.isnan(z))
    A = (np.angle(z[idx]) + np.pi) / (2*np.pi)
    A = (A + 0.5) % 1.0
    B = 1.0 - 1.0/(1.0+abs(z[idx])**0.3)
    c[idx] = [hsl_to_rgb(a, 0.8, b) for a,b in zip(A,B)]
    return c