"""Information measures used in the manuscript (all in bits).

A finite pulse defines a binary channel from the intervention X (0 = no pulse, 1 = pulse) to the long-run fate Y
(0 = the proliferative attractor is kept, 1 = absorption in an arrested fixed point). Without a pulse the network
stays in its proliferative attractor with probability 1, and with a pulse it is arrested with probability a, so the
channel is a Z-channel with error probability 1 - a. Its capacity is

    C(a) = log2(1 + a (1 - a)^((1 - a) / a)),        C(0) = 0, C(1) = 1,

attained for the input probability P(X = 1) = 1 / (a (1 + 2^(H2(a) / a))), where H2 is the binary entropy.
"""

import numpy as np


def binary_entropy(p):
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    out[m] = -p[m] * np.log2(p[m]) - (1 - p[m]) * np.log2(1 - p[m])
    return out if out.ndim else float(out)


def entropy(probabilities):
    p = np.asarray(probabilities, float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def z_capacity(a):
    """Capacity (bits) of the Z-channel written by a pulse with arrest probability a."""
    a = np.clip(np.asarray(a, float), 0.0, 1.0)
    out = np.zeros_like(a)
    one = a >= 1.0
    out[one] = 1.0
    m = (a > 0) & (a < 1)
    e = 1.0 - a[m]
    out[m] = np.log2(1.0 + a[m] * e ** (e / a[m]))
    return out if out.ndim else float(out)


def z_optimal_input(a):
    """Capacity-achieving probability of applying the pulse."""
    a = float(a)
    if a <= 0:
        return float("nan")
    if a >= 1:
        return 0.5
    return 1.0 / (a * (1.0 + 2.0 ** (binary_entropy(a) / a)))


def z_mutual_information(a, q):
    """I(X;Y) of the Z-channel for P(X = 1) = q."""
    return float(binary_entropy(q * a) - q * binary_entropy(a))


def arrest_for_capacity(c):
    """Arrest probability a at which the capacity equals c bits (inverse of z_capacity)."""
    from scipy.optimize import brentq

    return brentq(lambda a: z_capacity(a) - c, 1e-9, 1 - 1e-12)
