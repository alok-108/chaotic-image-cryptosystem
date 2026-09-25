"""Unit tests for Split and Join Algorithm."""

import numpy as np
import pytest
from src.split_join import split_join, inverse_split_join


def test_split_join_grayscale_256():
    """Verify lossless recovery for 256x256 grayscale image (32x32 blocks)."""
    np.random.seed(77)
    original = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    x0 = 0.123456789
    r = 3.9999

    confused = split_join(original, x0=x0, r=r)
    assert not np.array_equal(original, confused)
    # Total pixel value multiset must be preserved
    assert np.array_equal(np.sort(original.flatten()), np.sort(confused.flatten()))

    restored = inverse_split_join(confused, x0=x0, r=r)
    assert np.array_equal(original, restored)


def test_split_join_color_64():
    """Verify lossless recovery for 64x64x3 color image (8x8 blocks)."""
    np.random.seed(88)
    original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    x0 = 0.7654321
    r = 3.998

    confused = split_join(original, x0=x0, r=r)
    assert not np.array_equal(original, confused)

    restored = inverse_split_join(confused, x0=x0, r=r)
    assert np.array_equal(original, restored)


def test_split_join_invalid_dimensions():
    """Verify ValueError is raised if dimensions are not divisible by 8 or not square."""
    img_not_div8 = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(ValueError):
        split_join(img_not_div8)

    img_not_square = np.zeros((64, 128), dtype=np.uint8)
    with pytest.raises(ValueError):
        split_join(img_not_square)
