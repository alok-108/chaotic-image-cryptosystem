"""Logistic Map Pseudo-Random Number and Permutation Generator.

The 1D chaotic logistic map is defined by:
    x_{n+1} = r * x_n * (1 - x_n)
Where:
    x0 in (0, 1) is the initial condition (seed)
    r in (0, 4] is the control parameter (chaotic regime for r in [3.57, 4.0])
"""

import numpy as np


def iterate_logistic(x0: float, r: float, steps: int, discard: int = 0) -> np.ndarray:
    """Iterate the 1D logistic map for a given number of steps.

    Args:
        x0: Initial condition, x0 in (0, 1).
        r: Bifurcation / control parameter, r in (0, 4].
        steps: Number of states to record and return.
        discard: Number of initial transient iterations to discard (burn-in).

    Returns:
        np.ndarray: 1D array of float64 chaotic values in (0, 1).
    """
    if steps <= 0:
        return np.empty(0, dtype=np.float64)

    if not (0.0 < x0 < 1.0):
        raise ValueError(f"Initial state x0 must be in (0, 1), got {x0}")
    if not (0.0 < r <= 4.0):
        raise ValueError(f"Control parameter r must be in (0, 4], got {r}")

    x = float(x0)
    # Discard transient states if requested
    for _ in range(discard):
        x = r * x * (1.0 - x)

    seq = np.empty(steps, dtype=np.float64)
    for i in range(steps):
        x = r * x * (1.0 - x)
        seq[i] = x

    return seq


def generate_sequence(x0: float, r: float, length: int, discard: int = 0) -> np.ndarray:
    """Generate pseudo-random byte sequence (uint8) using the logistic map.

    Formula from paper:
        random_number = int(abs(x_new * 10^7)) % 256

    Args:
        x0: Initial condition in (0, 1).
        r: Control parameter in (0, 4].
        length: Number of bytes to generate.
        discard: Transient states to discard.

    Returns:
        np.ndarray: 1D array of uint8 values in [0, 255].
    """
    if length <= 0:
        return np.empty(0, dtype=np.uint8)

    float_seq = iterate_logistic(x0, r, length, discard=discard)
    # Vectorized conversion: int(abs(x * 10^7)) % 256
    scaled = np.floor(np.abs(float_seq * 1e7)).astype(np.int64)
    bytes_seq = (scaled % 256).astype(np.uint8)
    return bytes_seq


def generate_permutation(x0: float, r: float, size: int, discard: int = 0) -> np.ndarray:
    """Generate a pseudo-random permutation of indices [0, 1, ..., size - 1].

    Generates 'size' floating-point chaotic states from the logistic map
    and sorts them using argsort to produce a collision-free bijection.

    Args:
        x0: Initial condition in (0, 1).
        r: Control parameter in (0, 4].
        size: Size of the permutation vector.
        discard: Transient states to discard.

    Returns:
        np.ndarray: 1D array of int64 indices representing permutation.
    """
    if size <= 0:
        return np.empty(0, dtype=np.int64)

    float_seq = iterate_logistic(x0, r, size, discard=discard)
    permutation = np.argsort(float_seq)
    return permutation
