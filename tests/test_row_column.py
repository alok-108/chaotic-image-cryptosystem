"""Unit tests for Row and Column Index Manipulator."""

import numpy as np
from src.row_column_manipulator import (
    row_column_manipulator,
    inverse_row_column_manipulator,
)


def test_row_column_manipulator_grayscale_exact_recovery():
    """Verify lossless recovery for 2D grayscale random images."""
    np.random.seed(42)
    original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    x0 = 0.123456789
    r = 3.9999

    shuffled = row_column_manipulator(original, x0=x0, r=r)
    # Check that positions have indeed changed
    assert not np.array_equal(original, shuffled)
    # Check histogram/values multiset is preserved (pure permutation)
    assert np.array_equal(np.sort(original.flatten()), np.sort(shuffled.flatten()))

    recovered = inverse_row_column_manipulator(shuffled, x0=x0, r=r)
    assert np.array_equal(original, recovered)


def test_row_column_manipulator_color_exact_recovery():
    """Verify lossless recovery for 3D color images."""
    np.random.seed(101)
    original = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    x0 = 0.654321
    r = 3.999

    shuffled = row_column_manipulator(original, x0=x0, r=r)
    assert not np.array_equal(original, shuffled)

    recovered = inverse_row_column_manipulator(shuffled, x0=x0, r=r)
    assert np.array_equal(original, recovered)


def test_row_column_manipulator_rectangular():
    """Verify correct operation on non-square dimensions."""
    original = np.random.randint(0, 256, (64, 128, 3), dtype=np.uint8)
    x0 = 0.2468
    r = 3.995

    shuffled = row_column_manipulator(original, x0=x0, r=r)
    assert shuffled.shape == original.shape
    recovered = inverse_row_column_manipulator(shuffled, x0=x0, r=r)
    assert np.array_equal(original, recovered)
