#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to convert all ONNX MDX-Net files into a safetensors
# Run it using: python tool/batch_convert.py
#
# Created using Gemini 2.5 Pro
import argparse
import json
import os
import re
import subprocess
import sys
# Local imports
import bootstrap  # noqa: F401
from source.db.hash import get_hash
from source.db.models_db import load_known_models, save_known_models, get_db_filename
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.misc import cli_add_verbose


def parse_converter_output(output):
    """Parses the verbose output of onnx2safetensors.py to extract parameters."""
    params = {}
    try:
        params['dim_f'] = int(re.search(r"Frequency Dimension \(dim_f\):\s*(\d+)", output).group(1))
        params['channels'] = int(re.search(r"Base Channels \(ch\):\s*(\d+)", output).group(1))
        params['stages'] = int(re.search(r"U-Net Stages:\s*(\d+)", output).group(1))
        params['params'] = int(re.search(r"Total parameters:\s*(\d+)", output).group(1))
    except (AttributeError, TypeError):
        # This happens if the output did not contain the expected lines
        main_logger.error("No detection from convert.")
        sys.exit(3)
    return params


def main(args):
    logger_set_standalone(args)
    # 1. Load the JSON metadata file
    model_db = load_known_models(args.json_file)
    if model_db is None:
        main_logger.error("No model database available.")
        model_db = {}

    # Ensure source and destination directories exist
    if not os.path.isdir(args.source_dir):
        main_logger.error(f"Source directory not found at '{args.source_dir}'.")
        sys.exit(3)
    os.makedirs(args.dest_dir, exist_ok=True)

    # 2. Get list of files to convert
    onnx_files = sorted([f for f in os.listdir(args.source_dir) if f.endswith('.onnx')])

    if args.test:
        main_logger.info("\n--- RUNNING IN TEST MODE ---")
        onnx_files = ["Kim_Vocal_2.onnx", "kuielab_b_drums.onnx"]
        for fn in onnx_files:
            if not os.path.exists(os.path.join(args.source_dir, fn)):
                main_logger.error(f"Test file '{fn}' not found in source directory.")
                sys.exit(3)

    main_logger.info(f"\nFound {len(onnx_files)} ONNX files to process.")

    # 3. Process each file
    for filename in onnx_files:
        main_logger.info(f"\n{'='*50}")
        main_logger.info(f"Processing '{filename}'...")
        source_path = os.path.join(args.source_dir, filename)

        # Check if destination file already exists
        dest_filename = os.path.splitext(filename)[0] + ".safetensors"
        dest_path = os.path.join(args.dest_dir, dest_filename)
        if os.path.exists(dest_path):
            main_logger.info(f"Destination file '{dest_path}' already exists. Skipping.")
            continue

        # Get file hash
        file_hash = get_hash(source_path)

        # Verify hash exists in JSON database
        if file_hash not in model_db:
            main_logger.error(f"Hash '{file_hash}' for file '{filename}' not found in JSON database.")
            main_logger.error("Please add the model's metadata to the JSON file before converting.")
            sys.exit(3)

        model_info = model_db[file_hash]
        main_logger.info(f"Found metadata for '{model_info.get('name', 'N/A')}'. Hash: {file_hash}")

        # Prepare the command to call the converter tool
        command = [
            sys.executable,
            args.converter_script,
            source_path,
            "-o", dest_path,
            "-m", args.model_location,
            "-j", json.dumps(model_info),
            "-v"  # Always use verbose mode to capture the parameters
        ]

        main_logger.debug(f"Executing command: {' '.join(command)}")

        # 4. Call the converter and validate its output
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')

            main_logger.debug("--- Converter Output ---")
            main_logger.debug(result.stdout)
            if result.stderr:
                main_logger.error("--- Converter Errors ---")
                main_logger.error(result.stderr)

            # 5. Parse output and compare parameters
            detected_params = parse_converter_output(result.stdout)

            # Get expected params from JSON, using defaults for missing values
            expected_params = {
                'dim_f': model_info.get("mdx_dim_f_set", 3072),
                'channels': model_info.get("channels", 48),
                'stages': model_info.get("stages", 5),
                'params': model_info.get("params", 0)
            }

            main_logger.info(f"Comparing parameters: Expected {expected_params} vs Detected {detected_params}")

            if detected_params == expected_params:
                main_logger.info(f"✅ SUCCESS: Parameters match. Conversion for '{filename}' is valid.")

                # 6. Compute the hash for the new safetensors file
                main_logger.info(f"Computing hash for new file '{dest_path}'...")
                new_hash = get_hash(dest_path)
                main_logger.info(f"New file hash: {new_hash}")

                # Find and remove any old entry that points to the same safetensors filename
                old_hash_to_remove = None
                for hash_key, data in model_db.items():
                    if data.get("name") == dest_filename:
                        old_hash_to_remove = hash_key
                        break

                if old_hash_to_remove:
                    main_logger.info(f"Removing old database entry for '{dest_filename}' (hash: {old_hash_to_remove}).")
                    del model_db[old_hash_to_remove]

                # Add the new entry, copying relevant metadata from the ONNX entry
                main_logger.info(f"Adding new entry to database for hash: {new_hash}")
                # 1. Start with a deep copy of the original ONNX model's metadata
                new_entry = model_info.copy()
                # 2. Update the keys with our new, verified information
                new_entry["name"] = dest_filename  # Update the filename
                new_entry["mdx_dim_f_set"] = detected_params['dim_f']  # Update with detected value
                new_entry["channels"] = detected_params['channels']  # Update with detected value
                new_entry["stages"] = detected_params['stages']  # Update with detected value
                new_entry["params"] = detected_params['params']  # Update with detected value
                # Adjust some values to match the conversion
                new_entry["file_t"] = "safetensors"
                new_entry["download"] = "Main/MDX"
                # 3. Add the new hash to the database
                model_db[new_hash] = new_entry
            else:
                main_logger.error(f"Parameter mismatch for '{filename}'.")
                main_logger.error("Deleting incorrect output file.")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                sys.exit(3)

        except subprocess.CalledProcessError as e:
            main_logger.error(f"onnx2safetensors.py script failed for '{filename}'.")
            main_logger.error("--- STDOUT ---")
            print(e.stdout)
            main_logger.error("--- STDERR ---")
            print(e.stderr)
            main_logger.error("----------------")

    if not args.test or args.always_save_db:  # Only save the database if not in test mode
        main_logger.info("\n--- Saving updated JSON database ---")
        try:
            save_known_models(model_db, args.json_file)
        except Exception:
            sys.exit(3)
    else:
        main_logger.info("\n--- Test mode finished. JSON database was NOT modified. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A batch processing tool to convert ONNX models to Safetensors using a metadata file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source_dir', type=str, default='models', help="Directory containing the input .onnx files.")
    parser.add_argument('--dest_dir', type=str, default='models/new', help="Directory to save the output .safetensors files.")
    parser.add_argument('--json_file', type=str, default=get_db_filename(), help="Path to the metadata JSON file.")
    parser.add_argument('--test', action='store_true', help="Run in test mode, converting only 'Kim_Vocal_2.onnx'.")
    parser.add_argument('--always_save_db', action='store_true', help="Save updated database even while in test mode.")

    # Paths to the other tools in the same directory
    parser.add_argument('--converter_script', type=str, default='tool/onnx2safetensors.py',
                        help="Source to convert the files.")
    parser.add_argument('--model_location', type=str, default='source/inference/MDX_Net.py:MDX_Net',
                        help="Python path to the model class.")
    cli_add_verbose(parser)

    args = parser.parse_args()
    main(args)
