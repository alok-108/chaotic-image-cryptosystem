"""Unit tests for Logistic Map PRNG and permutation generator."""

import numpy as np
import pytest
from src.logistic_map import iterate_logistic, generate_sequence, generate_permutation


def test_iterate_logistic_bounds():
    """Verify generated float values stay strictly in (0, 1)."""
    x0 = 0.123456789
    r = 3.9999
    seq = iterate_logistic(x0, r, 1000)
    assert len(seq) == 1000
    assert np.all(seq > 0.0)
    assert np.all(seq < 1.0)


def test_iterate_logistic_invalid_inputs():
    """Verify ValueError is raised on out-of-bound parameters."""
    with pytest.raises(ValueError):
        iterate_logistic(0.0, 3.9, 100)
    with pytest.raises(ValueError):
        iterate_logistic(1.0, 3.9, 100)
    with pytest.raises(ValueError):
        iterate_logistic(0.5, 4.5, 100)


def test_generate_sequence_properties():
    """Verify sequence length, uint8 data type, and byte range [0, 255]."""
    x0 = 0.3541
    r = 3.9876
    length = 50000
    seq = generate_sequence(x0, r, length)

    assert isinstance(seq, np.ndarray)
    assert seq.dtype == np.uint8
    assert len(seq) == length
    assert seq.min() >= 0
    assert seq.max() <= 255
    # Distinct values coverage
    unique_vals = np.unique(seq)
    assert len(unique_vals) > 250, "PRNG should cover nearly all 256 byte states"


def test_generate_sequence_determinism():
    """Verify that identical keys produce bit-for-bit identical sequences."""
    x0 = 0.123456789
    r = 3.9999
    seq1 = generate_sequence(x0, r, 10000)
    seq2 = generate_sequence(x0, r, 10000)
    assert np.array_equal(seq1, seq2)


def test_generate_sequence_sensitivity():
    """Verify chaotic butterfly effect: tiny perturbation yields different sequence."""
    x0 = 0.123456789
    r = 3.9999
    delta = 1e-15
    seq1 = generate_sequence(x0, r, 5000)
    seq2 = generate_sequence(x0 + delta, r, 5000)
    # The sequences should rapidly diverge
    assert not np.array_equal(seq1, seq2)
    # Difference rate after transient iterations should be high
    diff_rate = np.mean(seq1[100:] != seq2[100:])
    assert diff_rate > 0.95


def test_generate_permutation():
    """Verify permutation indices form a strict bijection over [0..size-1]."""
    x0 = 0.456789
    r = 3.9999
    size = 256
    perm = generate_permutation(x0, r, size)

    assert len(perm) == size
    assert sorted(perm.tolist()) == list(range(size))
