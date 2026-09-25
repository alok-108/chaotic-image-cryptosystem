"""Encryption Pipeline for Chaotic Image Cryptosystem.

Follows the 4-step pipeline:
Step 1: Apply Row and Column Index Manipulator -> image_value (confusion stage 1)
Step 2: Apply Split and Join Algorithm -> image_updated (confusion stage 2)
Step 3: Generate pseudo-random sequence R1 using logistic map
Step 4: XOR image_updated with R1 -> cipher image C (diffusion stage)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from .row_column_manipulator import row_column_manipulator
from .split_join import split_join
from .logistic_map import generate_sequence
from .utils import pad_to_divisible_by_8


def encrypt(
    image: np.ndarray,
    x0: float = 0.123456789,
    r: float = 3.9999,
    auto_pad: bool = True,
    return_stages: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Encrypt an image using the two-stage confusion and diffusion chaotic cryptosystem.

    Args:
        image: Plain image array of shape (H, W) or (H, W, C), uint8.
        x0: Initial condition for logistic map, x0 in (0, 1). Default 0.123456789.
        r: Control bifurcation parameter, r in (0, 4]. Default 3.9999.
        auto_pad: If True, pads image to square dimensions divisible by 8 if needed.
        return_stages: If True, returns (cipher_image, stages_dict) for visualization.

    Returns:
        cipher_image: uint8 encrypted array.
        (Optional) stages_dict: dictionary containing 'plain', 'rcm', 'split_join', and 'cipher'.
    """
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    orig_shape = image.shape
    if auto_pad:
        work_img, _ = pad_to_divisible_by_8(image, make_square=True)
    else:
        work_img = image.copy()

    # Step 1: Row and Column Index Manipulator (first confusion)
    image_value = row_column_manipulator(work_img, x0=x0, r=r)

    # Step 2: Split and Join Algorithm (second confusion)
    image_updated = split_join(image_value, x0=x0, r=r)

    # Step 3: Generate pseudo-random sequence R1 using logistic map
    total_bytes = work_img.size
    r1_1d = generate_sequence(x0=x0, r=r, length=total_bytes)
    r1 = r1_1d.reshape(work_img.shape)

    # Step 4: XOR image_updated with R1 (diffusion)
    cipher_image = np.bitwise_xor(image_updated, r1)

    if return_stages:
        stages = {
            "plain": work_img,
            "rcm": image_value,
            "split_join": image_updated,
            "diffusion_mask": r1,
            "cipher": cipher_image,
            "orig_shape": orig_shape,
        }
        return cipher_image, stages

    return cipher_image
