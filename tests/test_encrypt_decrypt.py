"""Unit tests for Full Encryption and Decryption Pipelines."""

import os
import numpy as np
import pytest
from src.encrypt import encrypt
from src.decrypt import decrypt
from src.utils import load_image

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_images")


@pytest.mark.parametrize("image_name", ["lena.png", "baboon.png", "barbara.png"])
def test_full_pipeline_sample_images_color(image_name):
    """Test full encryption & decryption pipeline on standard 256x256 color benchmark images."""
    path = os.path.join(DATA_DIR, image_name)
    if not os.path.exists(path):
        pytest.skip(f"Sample image {image_name} not found at {path}")

    original = load_image(path, as_gray=False)
    assert original.shape[:2] == (256, 256)

    x0 = 0.123456789
    r = 3.9999

    # Encrypt
    cipher = encrypt(original, x0=x0, r=r)
    assert cipher.shape == original.shape
    assert not np.array_equal(original, cipher)

    # Decrypt
    decrypted = decrypt(cipher, x0=x0, r=r)
    assert np.array_equal(original, decrypted), f"Decryption failed for {image_name} (color)"


@pytest.mark.parametrize("image_name", ["lena.png", "baboon.png", "barbara.png"])
def test_full_pipeline_sample_images_grayscale(image_name):
    """Test full encryption & decryption pipeline on grayscale images."""
    path = os.path.join(DATA_DIR, image_name)
    if not os.path.exists(path):
        pytest.skip(f"Sample image {image_name} not found at {path}")

    original = load_image(path, as_gray=True)
    assert original.ndim == 2

    x0 = 0.432109876
    r = 3.9995

    cipher = encrypt(original, x0=x0, r=r)
    assert cipher.shape == original.shape
    assert not np.array_equal(original, cipher)

    decrypted = decrypt(cipher, x0=x0, r=r)
    assert np.array_equal(original, decrypted), f"Decryption failed for {image_name} (grayscale)"


def test_key_sensitivity_decryption():
    """Verify that decryption with a key differing by 10^-14 fails to recover the image."""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    x0_correct = 0.123456789
    r_correct = 3.9999

    cipher = encrypt(img, x0=x0_correct, r=r_correct)

    # Attempt decryption with slightly wrong x0
    x0_wrong = x0_correct + 1e-14
    decrypted_wrong = decrypt(cipher, x0=x0_wrong, r=r_correct)

    assert not np.array_equal(img, decrypted_wrong)
    # The error should be massive (> 95% pixels different)
    diff_rate = np.mean(img != decrypted_wrong)
    assert diff_rate > 0.95


def test_intermediate_stages_extraction():
    """Verify return_stages returns plain, rcm, split_join, and cipher."""
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    cipher, stages = encrypt(img, return_stages=True)

    assert "plain" in stages
    assert "rcm" in stages
    assert "split_join" in stages
    assert "cipher" in stages
    assert np.array_equal(cipher, stages["cipher"])
