#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to get hashes for all the files in a dir, using a cache
# Can be used as an standalone tool for testing
# Code by Gemini 2.5 Pro
import os
import csv
import logging
import sys

from .hash import get_hash
from ..utils.misc import NODES_NAME, debugl

# Set up the logger as specified
logger = logging.getLogger(f"{NODES_NAME}.hash_dir")

# Constants for clarity
MIN_FILE_SIZE_MB = 10
MIN_FILE_SIZE_BYTES = MIN_FILE_SIZE_MB * 1024 * 1024
CATALOG_FILENAME = ".catalog.csv"


def hash_dir(directory_path: str) -> dict:
    """
    Computes hashes for large files in a directory, using a local cache
    to avoid re-computation.

    Args:
        directory_path: The path to the directory to scan.

    Returns:
        A dictionary mapping {file_hash: file_name} for all files in the
        directory that are 10MB or larger.
    """
    logger.debug(f"Starting hash process for directory: '{directory_path}'")

    # 1. Ensure the target directory exists.
    try:
        os.makedirs(directory_path, exist_ok=True)
    except OSError as e:
        logger.error(f"FATAL: Could not create directory '{directory_path}'. Error: {e}")
        # This is a fatal error, so we raise the exception to stop execution.
        raise

    catalog_path = os.path.join(directory_path, CATALOG_FILENAME)
    cached_data = {}

    # 2. Load existing cache from ".catalog.csv" if it exists.
    try:
        with open(catalog_path, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header
            next(reader, None)
            for row in reader:
                if len(row) == 3:
                    filename, file_hash, timestamp = row
                    cached_data[filename] = (file_hash, float(timestamp))
            logger.debug(f"Successfully loaded {len(cached_data)} entries from cache: {catalog_path}")
    except FileNotFoundError:
        logger.debug(f"Cache file '{catalog_path}' not found. A new one will be created.")
    except Exception as e:
        logger.warning(f"Could not read cache file '{catalog_path}'. Proceeding without cache. Error: {e}")

    final_hashes = {}
    updated_cache = {}
    has_updates = False

    # 3. Iterate through files in the directory.
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        # Skip subdirectories and the cache file itself
        if not os.path.isfile(file_path) or filename == CATALOG_FILENAME:
            continue

        # 4. Filter by file size.
        try:
            file_size = os.path.getsize(file_path)
            if file_size < MIN_FILE_SIZE_BYTES:
                logger.debug(f"Skipping small file: '{filename}' ({file_size / 1024**2:.2f}MB)")
                continue
        except OSError as e:
            logger.warning(f"Could not get size of file '{filename}'. Skipping. Error: {e}")
            continue

        current_mtime = os.path.getmtime(file_path)
        file_hash = None

        # 5. Check cache for a valid, up-to-date entry.
        if filename in cached_data:
            cached_hash, cached_mtime = cached_data[filename]
            if current_mtime == cached_mtime:
                debugl(logger, 2, f"Cache hit for '{filename}'. Using stored hash.")
                file_hash = cached_hash
            else:
                logger.debug(f"File '{filename}' has been modified. Re-calculating hash.")

        # 6. If no valid cache entry, compute the hash.
        if file_hash is None:
            logger.debug(f"Computing hash for '{filename}'...")
            try:
                file_hash = get_hash(file_path)
                has_updates = True
            except IOError as e:
                logger.warning(f"Could not read file '{filename}' to compute hash. Skipping. Error: {e}")
                continue

        # Add to the results and prepare for caching
        final_hashes[file_hash] = file_path
        updated_cache[filename] = (file_hash, current_mtime)

    # 7. Write the updated cache back to disk.
    if has_updates:
        try:
            with open(catalog_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Filename', 'Hash', 'Time stamp'])
                for filename, (file_hash, mtime) in updated_cache.items():
                    writer.writerow([filename, file_hash, mtime])
            logger.debug(f"Successfully wrote {len(updated_cache)} entries to cache file '{catalog_path}'.")
        except IOError as e:
            # This is a non-fatal warning as per requirements.
            logger.warning(f"Failed to write to cache file '{catalog_path}'. Hashes were computed but not saved. Error: {e}")

    logger.debug(f"Hash process finished. Found {len(final_hashes)} valid files.")
    return final_hashes


# ==============================================================================
# Command-Line Tool for Testing and Operation
# python -m source.db.hash_dir
# ==============================================================================
if __name__ == "__main__":
    # Local imports to avoid top-level pollution
    import argparse
    from pathlib import Path
    import pprint

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Calculate and cache hashes for large files in a directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'directory',
        nargs='?',  # Makes the argument optional; we'll validate it manually.
        help='The directory to scan. Required unless --test is used.'
    )
    parser.add_argument(
        '--test',
        action='store_true',  # This is a flag; if present, args.test will be True.
        help='Run in test mode. Creates a temporary directory with dummy files and runs a test sequence.'
    )
    args = parser.parse_args()

    # --- Setup Logging ---
    # Enable debug logging for command-line execution to provide useful feedback.
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # --- Main Logic ---
    try:
        if args.test:
            # --- TEST MODE ---
            # This mode ignores the 'directory' argument and uses a fixed test path.

            # Helper function only needed for test mode
            def create_test_file(path: Path, size_in_bytes: int):
                """Helper to create a dummy file of a specific size."""
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "wb") as f:
                    f.write(os.urandom(size_in_bytes))
                logger.info(f"Created test file: {path} ({size_in_bytes / 1024**2:.2f}MB)")

            test_dir = Path("./temp_hash_dir_test_cli")
            print("-" * 60)
            print("Running in TEST mode.")
            print(f"Test directory: '{test_dir.resolve()}'")
            print("-" * 60)

            # Create test files
            create_test_file(test_dir / "large_file_1.bin", 11 * 1024 * 1024)  # > 10MB
            create_test_file(test_dir / "large_file_2.bin", 15 * 1024 * 1024)  # > 10MB
            create_test_file(test_dir / "small_file.txt", 1 * 1024 * 1024)   # < 10MB
            create_test_file(test_dir / "edge_case_file.bin", 10 * 1024 * 1024)  # exactly 10MB

            # Run the test sequence
            print("\n>>> FIRST RUN: Computing all hashes...")
            result_dict = hash_dir(str(test_dir))
            pprint.pprint(result_dict)

            print("\n>>> SECOND RUN: Should use cache for all files...")
            result_dict_cached = hash_dir(str(test_dir))
            pprint.pprint(result_dict_cached)

            print("\n>>> THIRD RUN: Modifying a file and re-running to test cache invalidation...")
            create_test_file(test_dir / "large_file_1.bin", 12 * 1024 * 1024)
            result_dict_modified = hash_dir(str(test_dir))
            pprint.pprint(result_dict_modified)

        else:
            # --- NORMAL OPERATION MODE ---

            # In normal mode, the 'directory' argument is required.
            if not args.directory:
                parser.error("The 'directory' argument is required when not using --test.")

            target_directory = args.directory
            print("-" * 60)
            print(f"Running in NORMAL mode on directory: '{target_directory}'")
            print("-" * 60)

            # Just run the function once on the specified directory
            result_dict = hash_dir(target_directory)

            print("\n>>> Resulting Hashes:")
            pprint.pprint(result_dict)

    except Exception as e:
        logger.critical(f"A critical error occurred during execution: {e}")
        sys.exit(1)
