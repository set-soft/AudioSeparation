# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# From various UVR clones adapted by Gemini 2.5 Pro
import hashlib
import re

HASH_REGEX = re.compile(r'^[\da-z]{32}$')
# The size to seek from the end of the file, in bytes.
# 10000 * 1024 bytes = 10,000 KB = ~9.77 MB
SEEK_SIZE = 10000 * 1024


def get_hash(filepath):
    """
    Calculates the MD5 hash of a file using a special method.

    It tries to hash only the last SEEK_SIZE bytes of the file. This is a
    shortcut used by some communities (e.g., for large AI models) to get a
    quick, unique hash without processing the entire file.

    If the file is smaller than SEEK_SIZE, it falls back to hashing the
    entire file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The calculated MD5 hexdigest.
    """
    try:
        with open(filepath, 'rb') as f:
            # Seek to SEEK_SIZE bytes from the end of the file (whence=2)
            f.seek(-SEEK_SIZE, 2)
            file_hash = hashlib.md5(f.read()).hexdigest()
    except (IOError, OSError):
        # This will happen if the file is smaller than SEEK_SIZE.
        # In that case, hash the entire file.
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()

    return file_hash


def is_hash(value):
    return HASH_REGEX.match(value)
