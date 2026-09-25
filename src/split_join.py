"""Split and Join Algorithm (Second Confusion Stage).

Algorithm steps:
1. Divide image into 64 equal parts (8x8 grid). For 256x256, each part is 32x32.
2. Label parts 0 to 63 in row-major order.
3. Rotate even-indexed parts by 180 degrees, odd-indexed parts by 90 degrees.
4. Generate chaotic sequence of length 64 (vector_value) to distribute the 64 parts
   into 8 vectors a0..a7 (8 parts per vector).
5. Horizontally join the 8 parts in each vector a0..a7 to form 8 horizontal strips.
6. Generate chaotic sequence of length 8 (vector_value1) to permute the 8 horizontal strips.
7. Vertically join the permuted strips to form the updated confused image.
"""

from typing import List, Tuple
import numpy as np
from .logistic_map import iterate_logistic


def _generate_split_join_permutations(x0: float, r: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the 64-block permutation and 8-strip permutation from logistic map.

    Generates 64 + 8 = 72 continuous states from the chaotic logistic map.
    First 64 states determine the block distribution into vectors a0..a7.
    Next 8 states determine the vertical arrangement of the 8 horizontal strips.
    """
    seq = iterate_logistic(x0, r, 64 + 8)
    vector_value = seq[:64]
    vector_value1 = seq[64:]

    # Permutations obtained by argsort
    part_permutation = np.argsort(vector_value)
    strip_permutation = np.argsort(vector_value1)
    return part_permutation, strip_permutation


def split_join(
    image: np.ndarray,
    x0: float = 0.123456789,
    r: float = 3.9999,
) -> np.ndarray:
    """Apply the Split and Join confusion algorithm to an image.

    Args:
        image: Array of shape (H, W) or (H, W, C). H and W must be divisible by 8 and equal.
        x0: Initial condition for logistic map.
        r: Control parameter for logistic map.

    Returns:
        np.ndarray: Updated image array of same shape and dtype.
    """
    H, W = image.shape[:2]
    if H % 8 != 0 or W % 8 != 0:
        raise ValueError(f"Image dimensions must be divisible by 8, got ({H}, {W})")
    if H != W:
        raise ValueError(
            f"Image must be square (H == W) for 90-degree block rotation; got ({H}, {W}). "
            "Please pad or resize the image before calling split_join."
        )

    h_block = H // 8
    w_block = W // 8

    # Step 1 & 2: Divide image into 64 equal parts (8x8 grid)
    parts: List[np.ndarray] = []
    for idx in range(64):
        r_idx = idx // 8
        c_idx = idx % 8
        part = image[
            r_idx * h_block : (r_idx + 1) * h_block,
            c_idx * w_block : (c_idx + 1) * w_block,
            ...,
        ].copy()

        # Step 3: Rotate even-indexed parts by 180 degrees, odd-indexed parts by 90 degrees
        if idx % 2 == 0:
            rotated_part = np.rot90(part, k=2, axes=(0, 1))
        else:
            rotated_part = np.rot90(part, k=1, axes=(0, 1))
        parts.append(rotated_part)

    # Step 4: Generate chaotic permutations
    part_permutation, strip_permutation = _generate_split_join_permutations(x0, r)

    # Distribute into 8 vectors a0..a7 and join horizontally to form 8 strips
    strips: List[np.ndarray] = []
    for j in range(8):
        a_j = part_permutation[j * 8 : (j + 1) * 8]
        # Join parts horizontally
        strip_j = np.concatenate([parts[p_idx] for p_idx in a_j], axis=1)
        strips.append(strip_j)

    # Step 6 & 7: Rearrange strips vertically according to strip_permutation
    image_updated = np.concatenate([strips[strip_permutation[k]] for k in range(8)], axis=0)

    return image_updated


def inverse_split_join(
    image_updated: np.ndarray,
    x0: float = 0.123456789,
    r: float = 3.9999,
) -> np.ndarray:
    """Reverse the Split and Join algorithm to recover original image before split/join.

    Args:
        image_updated: Scrambled image array of shape (H, W) or (H, W, C).
        x0: Initial condition used during encryption.
        r: Control parameter used during encryption.

    Returns:
        np.ndarray: Restored image array identical to the pre-split_join input.
    """
    H, W = image_updated.shape[:2]
    if H % 8 != 0 or W % 8 != 0 or H != W:
        raise ValueError(f"Image dimensions must be square and divisible by 8, got ({H}, {W})")

    h_block = H // 8
    w_block = W // 8

    part_permutation, strip_permutation = _generate_split_join_permutations(x0, r)

    # Step 1: Invert vertical strip rearrangement
    # Slices from top to bottom correspond to strips[strip_permutation[k]]
    recovered_strips: List[np.ndarray] = [None] * 8  # type: ignore
    for k in range(8):
        vertical_slice = image_updated[k * h_block : (k + 1) * h_block, :, ...]
        recovered_strips[strip_permutation[k]] = vertical_slice

    # Step 2: Invert horizontal strips to recover 64 rotated parts
    recovered_rotated_parts: List[np.ndarray] = [None] * 64  # type: ignore
    for j in range(8):
        a_j = part_permutation[j * 8 : (j + 1) * 8]
        strip_j = recovered_strips[j]
        for m in range(8):
            block_part = strip_j[:, m * w_block : (m + 1) * w_block, ...]
            orig_part_idx = a_j[m]
            recovered_rotated_parts[orig_part_idx] = block_part

    # Step 3: Invert rotations and place parts back to their grid positions
    restored_image = np.zeros_like(image_updated)
    for idx in range(64):
        rot_part = recovered_rotated_parts[idx]
        # Invert rotation: 180° is self-inverse (k=2). Inverse of 90° (k=1) is 270° (k=3).
        if idx % 2 == 0:
            unrotated = np.rot90(rot_part, k=2, axes=(0, 1))
        else:
            unrotated = np.rot90(rot_part, k=3, axes=(0, 1))

        r_idx = idx // 8
        c_idx = idx % 8
        restored_image[
            r_idx * h_block : (r_idx + 1) * h_block,
            c_idx * w_block : (c_idx + 1) * w_block,
            ...,
        ] = unrotated

    return restored_image
