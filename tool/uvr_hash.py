#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to get the hash used by UVR for a model i.e:
# python tool/uvr_hash.py model.onnx
import argparse
import os
# Local imports
import bootstrap  # noqa: F401
from source.utils.logger import main_logger, logger_set_standalone
from source.db.hash import get_hash


def main():
    """Main function to run the command-line tool."""
    parser = argparse.ArgumentParser(
        description="""Compute a special MD5 hash of a file.

        This tool calculates the hash of the last ~10MB of a file.
        If the file is smaller than that, it hashes the entire file.
        This method is often used for quick identification of large model files.
        """,
        # Makes the description formatting look nicer in the help text
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        'files',
        metavar='FILE',
        nargs='+',  # Accepts one or more file arguments
        help='Path to the file(s) to hash.'
    )

    args = parser.parse_args()
    args.verbose = 0
    logger_set_standalone(args)

    # Process each file provided on the command line
    for filepath in args.files:
        if not os.path.exists(filepath):
            main_logger.error(f"File not found at '{filepath}'")
            continue  # Skip to the next file

        if not os.path.isfile(filepath):
            main_logger.error(f"Path '{filepath}' is a directory, not a file.")
            continue  # Skip to the next file

        try:
            file_hash = get_hash(filepath)
            # Print in a standard format, similar to md5sum
            main_logger.info(f"{file_hash}  {filepath}")
        except Exception as e:
            main_logger.error(f"while processing file '{filepath}': {e}")


if __name__ == "__main__":
    main()
