# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to show the safetensors metadata
# python tool/show_metadata.py model.safetensors
import argparse
import json
from pprint import pprint
# Local imports
import bootstrap  # noqa: F401
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.misc import cli_add_verbose, json_object_hook
from source.inference.get_model import get_metadata


def show_metadata(args):
    d = get_metadata(args.input_file)
    main_logger.info(f"Metadata information for `{args.input_file}`")
    if 'model_t' not in d:
        main_logger.warning("Missing model_t key, this isn't an AudioSeparation file")
    expanded = {k: json.loads(v, object_hook=json_object_hook) if v[0] in '{[' else v for k, v in d.items()}
    pprint(expanded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shows the safetensors metadata",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the input ONNX model file.")
    cli_add_verbose(parser)

    args = parser.parse_args()
    logger_set_standalone(args)
    show_metadata(args)
