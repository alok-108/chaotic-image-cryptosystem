"""Security and Performance Analysis Metrics for Image Cryptosystem.

Provides functions to compute:
- Information Entropy (Shannon entropy)
- Pixel Intensity Histogram
- Adjacent Pixel Correlation Coefficients (Horizontal, Vertical, Diagonal)
- Number of Pixels Change Rate (NPCR)
- Unified Average Changing Intensity (UACI)
- Key Space Analysis (log2 key space)
- Encryption/Decryption Execution Time
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def entropy(image: np.ndarray) -> float:
    """Compute Shannon information entropy of an image.

    Ideal entropy for an 8-bit random cipher image is 8.0 bits.

    Args:
        image: Image array of shape (H, W) or (H, W, C), uint8.

    Returns:
        float: Shannon entropy in bits [0.0, 8.0].
    """
    flat = image.flatten()
    total_pixels = flat.size
    if total_pixels == 0:
        return 0.0

    counts = np.bincount(flat, minlength=256)
    probabilities = counts[counts > 0] / total_pixels
    h = -np.sum(probabilities * np.log2(probabilities))
    return float(h)


def histogram(
    image: np.ndarray,
    title: str = "Pixel Intensity Distribution",
    figsize: Tuple[int, int] = (6, 3.5),
) -> plt.Figure:
    """Generate and return a Matplotlib Figure showing pixel intensity distribution.

    Args:
        image: Image array of shape (H, W) or (H, W, C), uint8.
        title: Plot title.
        figsize: Figure dimensions in inches.

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor("#1e1e2e")
    ax.set_facecolor("#181825")
    ax.tick_params(colors="#cdd6f4")
    for spine in ax.spines.values():
        spine.set_color("#45475a")

    ax.set_title(title, color="#cdd6f4", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Pixel Intensity [0 - 255]", color="#bac2de", fontsize=9)
    ax.set_ylabel("Frequency", color="#bac2de", fontsize=9)
    ax.grid(color="#313244", linestyle="--", linewidth=0.5, alpha=0.7)

    if image.ndim == 3 and image.shape[2] == 3:
        colors = [("#f38ba8", "Red"), ("#a6e3a1", "Green"), ("#89b4fa", "Blue")]
        for c, (col_hex, name) in enumerate(colors):
            channel_data = image[:, :, c].flatten()
            counts = np.bincount(channel_data, minlength=256)
            ax.plot(counts, color=col_hex, label=name, alpha=0.85, linewidth=1.5)
        ax.legend(facecolor="#1e1e2e", edgecolor="#45475a", labelcolor="#cdd6f4", fontsize=8)
    else:
        counts = np.bincount(image.flatten(), minlength=256)
        ax.bar(range(256), counts, color="#89b4fa", width=1.0, alpha=0.85)

    ax.set_xlim(0, 255)
    fig.tight_layout()
    return fig


def correlation_coefficient(
    image: np.ndarray,
    direction: str = "horizontal",
    num_pairs: Optional[int] = None,
) -> float:
    """Calculate adjacent pixel correlation coefficient in a given direction.

    Correlation formula:
        r_xy = Cov(x, y) / (sigma_x * sigma_y)

    Args:
        image: Image array (H, W) or (H, W, C).
        direction: 'horizontal', 'vertical', or 'diagonal'.
        num_pairs: Optional number of randomly sampled pairs (default None = all adjacent pairs).

    Returns:
        float: Pearson correlation coefficient in [-1.0, 1.0].
    """
    if image.ndim == 3:
        # Convert to grayscale for adjacent spatial correlation analysis
        img = (0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]).astype(np.float64)
    else:
        img = image.astype(np.float64)

    H, W = img.shape[:2]

    if direction == "horizontal":
        x = img[:, :-1].flatten()
        y = img[:, 1:].flatten()
    elif direction == "vertical":
        x = img[:-1, :].flatten()
        y = img[1:, :].flatten()
    elif direction == "diagonal":
        x = img[:-1, :-1].flatten()
        y = img[1:, 1:].flatten()
    else:
        raise ValueError(f"Unknown direction '{direction}'. Must be 'horizontal', 'vertical', or 'diagonal'.")

    if num_pairs is not None and num_pairs < len(x):
        indices = np.random.choice(len(x), size=num_pairs, replace=False)
        x = x[indices]
        y = y[indices]

    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cov = np.mean((x - x_mean) * (y - y_mean))
    std_x = np.std(x)
    std_y = np.std(y)

    if std_x == 0.0 or std_y == 0.0:
        return 0.0

    return float(cov / (std_x * std_y))


def npcr(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """Calculate Number of Pixels Change Rate (NPCR) between two cipher images.

    NPCR measures percentage of differing pixels:
        NPCR = (sum_{i,j} D(i,j) / total_pixels) * 100%
        where D(i,j) = 1 if C1(i,j) != C2(i,j) else 0

    Ideal theoretical value: ~99.6094% (paper reports ~99.81%).

    Args:
        cipher1: First encrypted image.
        cipher2: Second encrypted image.

    Returns:
        float: NPCR percentage in [0.0, 100.0].
    """
    if cipher1.shape != cipher2.shape:
        raise ValueError(f"Cipher images must have identical shapes, got {cipher1.shape} vs {cipher2.shape}")

    diff = (cipher1 != cipher2).astype(np.float64)
    return float(np.mean(diff) * 100.0)


def uaci(cipher1: np.ndarray, cipher2: np.ndarray) -> float:
    """Calculate Unified Average Changing Intensity (UACI) between two cipher images.

    UACI measures average difference in intensity:
        UACI = (1 / (total_pixels * 255)) * sum_{i,j} |C1(i,j) - C2(i,j)| * 100%

    Ideal theoretical value: ~33.4635% (paper reports ~33.46%).

    Args:
        cipher1: First encrypted image.
        cipher2: Second encrypted image.

    Returns:
        float: UACI percentage in [0.0, 100.0].
    """
    if cipher1.shape != cipher2.shape:
        raise ValueError(f"Cipher images must have identical shapes, got {cipher1.shape} vs {cipher2.shape}")

    abs_diff = np.abs(cipher1.astype(np.float64) - cipher2.astype(np.float64))
    return float(np.mean(abs_diff) / 255.0 * 100.0)


def key_space(x0: float = 0.123456789, r: float = 3.9999) -> float:
    """Calculate log2 of the cryptosystem key space.

    Assuming precision 10^-16 for floating-point values and 6 independent parameters:
        Key space size = (10^16)^6 = 10^96
        log2(10^96) = 96 * log2(10) ≈ 318.907 bits ≈ 2^319.

    Args:
        x0: Initial condition.
        r: Control parameter.

    Returns:
        float: Equivalent key length in bits (log2 of key space), ≈ 318.9 bits.
    """
    num_parameters = 6
    precision_exponent = 16
    total_decimal_combinations = num_parameters * precision_exponent  # 96
    bits = total_decimal_combinations * np.log2(10.0)
    return float(bits)


def encryption_time(func: Callable, *args: Any, **kwargs: Any) -> Tuple[float, Any]:
    """Measure the execution time of a cryptographic function.

    Args:
        func: Callable function to time.
        *args: Positional arguments to pass to func.
        **kwargs: Keyword arguments to pass to func.

    Returns:
        Tuple[float, Any]: Elapsed execution time in seconds, and return value of func.
    """
    t_start = time.perf_counter()
    result = func(*args, **kwargs)
    t_end = time.perf_counter()
    elapsed = float(t_end - t_start)
    return elapsed, result
