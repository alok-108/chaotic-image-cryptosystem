"""Unit tests for Cryptosystem Performance and Security Metrics."""

import numpy as np
import pytest
from src.metrics import (
    entropy,
    correlation_coefficient,
    npcr,
    uaci,
    key_space,
    encryption_time,
)


def test_entropy_random_noise():
    """Verify that uniform random noise has Shannon entropy very close to 8.0."""
    np.random.seed(123)
    noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    h = entropy(noise)
    assert 7.99 <= h <= 8.0, f"Expected entropy ~8.0, got {h}"


def test_entropy_constant_image():
    """Verify that a single-color image has zero information entropy."""
    constant_img = np.zeros((100, 100), dtype=np.uint8)
    h = entropy(constant_img)
    assert h == 0.0


def test_correlation_random_noise():
    """Verify that random noise has near-zero adjacent pixel correlation."""
    np.random.seed(456)
    noise = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    for direction in ["horizontal", "vertical", "diagonal"]:
        corr = correlation_coefficient(noise, direction=direction)
        assert abs(corr) < 0.05, f"Expected near zero correlation for {direction}, got {corr}"


def test_correlation_gradient_image():
    """Verify that a smooth continuous gradient has correlation near 1.0."""
    gradient = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    corr_h = correlation_coefficient(gradient, direction="horizontal")
    assert corr_h > 0.99


def test_npcr_and_uaci_random_noise():
    """Verify NPCR and UACI between two independent random images match theoretical values."""
    np.random.seed(789)
    c1 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    c2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    val_npcr = npcr(c1, c2)
    val_uaci = uaci(c1, c2)

    # Theoretical: NPCR ~ 99.6094%, UACI ~ 33.4635%
    assert 99.0 <= val_npcr <= 100.0, f"Expected NPCR ~99.6%, got {val_npcr}"
    assert 32.5 <= val_uaci <= 34.5, f"Expected UACI ~33.5%, got {val_uaci}"


def test_npcr_and_uaci_identical_images():
    """Verify NPCR and UACI are 0.0 for identical images."""
    img = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
    assert npcr(img, img) == 0.0
    assert uaci(img, img) == 0.0


def test_key_space():
    """Verify key space calculation matches ~318.9 bits (≈ 2^319)."""
    ks = key_space()
    assert 318.0 <= ks <= 320.0, f"Expected key space ~319 bits, got {ks}"


def test_encryption_time_helper():
    """Verify execution time measurement returns positive elapsed time."""
    def dummy_func(x):
        return x * 2

    elapsed, result = encryption_time(dummy_func, 21)
    assert result == 42
    assert elapsed >= 0.0
