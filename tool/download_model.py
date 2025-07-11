#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to download a model from the models Data Base.
# Run it using: python tool/download_model.py HASH
import argparse
import sys
# Local imports
import bootstrap  # noqa: F401
from source.db.hash_dir import hash_dir
from source.db.models_db import load_known_models, get_download_url, cli_add_models_and_db
from source.utils.downloader import download_model
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.misc import cli_add_verbose


def main(args):
    logger_set_standalone(args)
    # Load the JSON metadata file
    model_db = load_known_models(args.json_file)
    if model_db is None:
        main_logger.error("No model database available.")
        sys.exit(3)

    # Is a valid hash?
    if args.hash not in model_db:
        main_logger.error(f"Nothing known about `{args.hash}`")
        sys.exit(4)
    d = model_db[args.hash]

    # Check what we have
    downloaded = hash_dir(args.models_dir)
    if args.hash in downloaded:
        main_logger.error(f"`{args.hash}` already downloaded as `{downloaded[args.hash]}`")
        sys.exit(5)

    # Check we can download it
    url = get_download_url(d)
    if url is None:
        main_logger.error(f"`{args.hash}` can't be downloaded")
        sys.exit(6)

    # Download the file
    name = d['name']
    try:
        download_model(url, args.models_dir, name, force_urllib=False)
    except Exception as e:
        main_logger.error(f"Failed to download {name} from {url}\n{e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downloads a model from the database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- File/Path Arguments ---
    parser.add_argument('hash', type=str, help="Hash for the file to download.")
    cli_add_models_and_db(parser)

    # --- Control Arguments ---
    cli_add_verbose(parser)

    args = parser.parse_args()
    main(args)
