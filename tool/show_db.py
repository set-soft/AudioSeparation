# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to show the models Data Base.
# Is used to do adjusts to the data base.
# Run it using: python tool/show_db.py
import argparse
import pprint
import sys
# Local imports
import bootstrap  # noqa: F401
from source.db.hash_dir import hash_dir
from source.db.models_db import load_known_models, cli_add_models_and_db, save_known_models, get_models
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.misc import cli_add_verbose


# Do nothing, you can apply some change here
def apply_process(model_db):
    return model_db, False

# Example of processing
# PARAMS = {"3072/48/5": 16684228,
#           "2560/48/5": 14763012,
#           "2048/48/5": 13191108,
#           "2048/32/5": 7420548,
#           "2048/32/4": 5478276}
#
# def apply_process(model_db):
#     for k, v in model_db.items():
#         if 'name' not in v:
#             continue
#         main_logger.info(f"Processing {v['desc']}")
#         key = f"{v['mdx_dim_f_set']}/{v['channels']}/{v['stages']}"
#         try:
#             v['params'] = PARAMS[key]
#         except KeyError:
#             print(f"No {key}")
#             raise
#     return model_db, True

# Example of processing
# def apply_process(model_db):
#     modified = False
#     for k, v in model_db.items():
#         if 'name' not in v:
#             continue
#         main_logger.info(f"Processing {v['desc']}")
#         if 'channels' not in v:
#             v['channels'] = 48
#             main_logger.info("- Explicit 48 channels")
#         if 'stages' not in v:
#             v['stages'] = 5
#             main_logger.info("- Explicit 5 stages")
#         v['model_t'] = 'MDX'
#         main_logger.info("- Explicit model_t MDX")
#         v['file_t'] = os.path.splitext(v['name'])[1][1:]
#         main_logger.info(f"- Explicit file_t {v['file_t']}")
#         modified = True
#     return model_db, modified


def main(args):
    logger_set_standalone(args)
    # 1. Load the JSON metadata file
    model_db = load_known_models(args.json_file)
    if model_db is None:
        main_logger.error("No model database available.")
        sys.exit(3)

    main_logger.info("\n--- Current DB ---\n")
    pprint.pprint(model_db)

    # 2. Apply any hardcoded processing changes (if any)
    model_db, modified = apply_process(model_db)

    # 3. Apply command-line filters if any were provided
    filters_active = args.primary_stem or args.model_t or args.file_t
    if filters_active:
        main_logger.info("\n--- Hashing downloaded models ---\n")
        hashes = hash_dir(args.models_dir)
        pprint.pprint(hashes)

        main_logger.info("\n--- Filtered DB ---\n")
        # The get_models function can handle lists/sets of values directly
        models_data, filtered_results = get_models(
            primary_stem=set(args.primary_stem) if args.primary_stem is not None else None,
            model_t=set(args.model_t) if args.model_t is not None else None,
            file_t=set(args.file_t) if args.file_t is not None else None,
            json_path=args.json_file,
            downloaded=hashes
        )
        pprint.pprint(filtered_results)
        pprint.pprint(models_data)

    # 4. If the apply_process function modified the DB, show it and save it
    if modified:
        main_logger.info("\n--- Modified DB (to be saved) ---\n")
        pprint.pprint(model_db)

        main_logger.info("\n--- Saving updated JSON database ---")
        try:
            save_known_models(model_db, args.json_file)
            main_logger.info("Save successful.")
        except Exception as e:
            main_logger.error(f"Failed to save database: {e}")
            sys.exit(3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show the content of the models database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- File/Path Arguments ---
    cli_add_models_and_db(parser)

    # --- Filtering Arguments ---
    parser.add_argument('--primary_stem', action='append',
                        help="Filter by primary stem. Can be used multiple times (e.g., --primary_stem Vocals).")
    parser.add_argument('--model_t', action='append',
                        help="Filter by model type. Can be used multiple times (e.g., --model_t MDX --model_t Demucs).")
    parser.add_argument('--file_t', action='append',
                        help="Filter by file type. Can be used multiple times (e.g., --file_t safetensors).")

    # --- Control Arguments ---
    cli_add_verbose(parser)

    args = parser.parse_args()
    main(args)
