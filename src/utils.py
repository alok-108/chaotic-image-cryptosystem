"""Utility functions for image I/O, padding, and transformations."""

from typing import Tuple, Union
import numpy as np
import cv2
from PIL import Image
import io


def load_image(source: Union[str, bytes, bytearray, Image.Image, np.ndarray], as_gray: bool = False) -> np.ndarray:
    """Load an image from various sources into a uint8 NumPy array in RGB or Grayscale format.

    Args:
        source: Filepath string, bytes/bytearray, PIL Image, or NumPy array.
        as_gray: If True, convert image to 2D grayscale array.

    Returns:
        np.ndarray: uint8 image array of shape (H, W) if as_gray else (H, W, 3) or (H, W).
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    if isinstance(source, Image.Image):
        if as_gray:
            img_pil = source.convert("L")
            return np.array(img_pil, dtype=np.uint8)
        else:
            img_pil = source.convert("RGB")
            return np.array(img_pil, dtype=np.uint8)

    if isinstance(source, (bytes, bytearray)):
        nparr = np.frombuffer(source, np.uint8)
        flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        img = cv2.imdecode(nparr, flag)
        if img is None:
            raise ValueError("Failed to decode image from bytes.")
        if not as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    if isinstance(source, str):
        flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        img = cv2.imread(source, flag)
        if img is None:
            raise FileNotFoundError(f"Could not load image at path: {source}")
        if not as_gray and img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    raise TypeError(f"Unsupported image source type: {type(source)}")


def save_image(image: np.ndarray, path: str) -> None:
    """Save an image (RGB or Grayscale) to disk.

    Args:
        image: uint8 NumPy array (H, W) or (H, W, 3) in RGB.
        path: Destination file path.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        # Convert RGB back to BGR for cv2.imwrite
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    else:
        cv2.imwrite(path, image)


def pad_to_divisible_by_8(image: np.ndarray, make_square: bool = True) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Pad image so both height and width are divisible by 8 (and optionally square).

    To allow 90-degree rotations in 8x8 block splitting without changing block dimensions,
    each block must be square (h_block == w_block), which requires H == W.
    If image dimensions are not equal or not divisible by 8, symmetric reflection padding is added.

    Args:
        image: Input image array (H, W) or (H, W, C).
        make_square: If True, pads to equal dimensions max(H, W) rounded up to a multiple of 8.

    Returns:
        padded_image: The padded uint8 image.
        orig_shape: Original shape tuple before padding.
    """
    orig_shape = image.shape
    H, W = orig_shape[:2]

    if make_square:
        target_size = max(H, W)
        if target_size % 8 != 0:
            target_size = ((target_size // 8) + 1) * 8
        target_H, target_W = target_size, target_size
    else:
        target_H = H if H % 8 == 0 else ((H // 8) + 1) * 8
        target_W = W if W % 8 == 0 else ((W // 8) + 1) * 8

    pad_h = target_H - H
    pad_w = target_W - W

    if pad_h == 0 and pad_w == 0:
        return image.copy(), orig_shape

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if image.ndim == 3:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="reflect",
        )
    else:
        padded = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="reflect",
        )

    return padded, orig_shape


def unpad_image(padded_image: np.ndarray, orig_shape: Tuple[int, ...]) -> np.ndarray:
    """Reverse padding applied by pad_to_divisible_by_8.

    Args:
        padded_image: Padded image array.
        orig_shape: Original image shape before padding.

    Returns:
        np.ndarray: Cropped image with original dimensions.
    """
    target_H, target_W = padded_image.shape[:2]
    orig_H, orig_W = orig_shape[:2]

    pad_h = target_H - orig_H
    pad_w = target_W - orig_W

    if pad_h == 0 and pad_w == 0:
        return padded_image.copy()

    pad_top = pad_h // 2
    pad_left = pad_w // 2

    if padded_image.ndim == 3:
        return padded_image[pad_top : pad_top + orig_H, pad_left : pad_left + orig_W, :].copy()
    else:
        return padded_image[pad_top : pad_top + orig_H, pad_left : pad_left + orig_W].copy()


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """Encode image array to bytes for download."""
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()
