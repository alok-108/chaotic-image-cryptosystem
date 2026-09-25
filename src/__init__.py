"""Chaotic Image Cryptosystem package.

A high-performance image encryption and decryption cryptosystem
using Row-Column Index Manipulation, Split-Join Algorithm, and Logistic Map Diffusion.
"""

from .logistic_map import (
    iterate_logistic,
    generate_sequence,
    generate_permutation,
)
from .row_column_manipulator import (
    row_column_manipulator,
    inverse_row_column_manipulator,
)
from .split_join import (
    split_join,
    inverse_split_join,
)
from .encrypt import encrypt
from .decrypt import decrypt
from .metrics import (
    entropy,
    histogram,
    correlation_coefficient,
    npcr,
    uaci,
    key_space,
    encryption_time,
)

__all__ = [
    "iterate_logistic",
    "generate_sequence",
    "generate_permutation",
    "row_column_manipulator",
    "inverse_row_column_manipulator",
    "split_join",
    "inverse_split_join",
    "encrypt",
    "decrypt",
    "entropy",
    "histogram",
    "correlation_coefficient",
    "npcr",
    "uaci",
    "key_space",
    "encryption_time",
]
