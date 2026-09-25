"""Decryption Pipeline for Chaotic Image Cryptosystem.

Follows the reverse 4-step pipeline:
Step 1: Generate the exact same pseudo-random sequence R1 using logistic map
Step 2: XOR cipher C with R1 -> confused image (inverse diffusion)
Step 3: Apply inverse Split and Join -> image_value (inverse second confusion)
Step 4: Apply inverse Row and Column Index Manipulator -> plain image P (inverse first confusion)
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from .row_column_manipulator import inverse_row_column_manipulator
from .split_join import inverse_split_join
from .logistic_map import generate_sequence
from .utils import unpad_image


def decrypt(
    cipher_image: np.ndarray,
    x0: float = 0.123456789,
    r: float = 3.9999,
    orig_shape: Optional[Tuple[int, ...]] = None,
    return_stages: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Decrypt a cipher image using the inverse chaotic cryptosystem pipeline.

    Args:
        cipher_image: Cipher image array of shape (H, W) or (H, W, C), uint8.
        x0: Initial condition for logistic map, x0 in (0, 1). Default 0.123456789.
        r: Control bifurcation parameter, r in (0, 4]. Default 3.9999.
        orig_shape: If image was padded, original shape before padding to unpad.
        return_stages: If True, returns (decrypted_image, stages_dict) for visualization.

    Returns:
        decrypted_image: uint8 decrypted image array.
        (Optional) stages_dict: dictionary with intermediate reversed states.
    """
    if cipher_image.dtype != np.uint8:
        cipher_image = np.clip(cipher_image, 0, 255).astype(np.uint8)

    # Step 1: Generate same pseudo-random sequence R1 using logistic map
    total_bytes = cipher_image.size
    r1_1d = generate_sequence(x0=x0, r=r, length=total_bytes)
    r1 = r1_1d.reshape(cipher_image.shape)

    # Step 2: XOR C with R1 -> image_updated (inverse diffusion)
    image_updated = np.bitwise_xor(cipher_image, r1)

    # Step 3: Apply inverse Split and Join -> image_value (inverse confusion 2)
    image_value = inverse_split_join(image_updated, x0=x0, r=r)

    # Step 4: Apply inverse Row and Column Index Manipulator -> plain image (inverse confusion 1)
    decrypted_full = inverse_row_column_manipulator(image_value, x0=x0, r=r)

    if orig_shape is not None and orig_shape != decrypted_full.shape:
        decrypted_image = unpad_image(decrypted_full, orig_shape)
    else:
        decrypted_image = decrypted_full

    if return_stages:
        stages = {
            "cipher": cipher_image,
            "post_xor": image_updated,
            "post_inverse_split_join": image_value,
            "decrypted": decrypted_image,
        }
        return decrypted_image, stages

    return decrypted_image
