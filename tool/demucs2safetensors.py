#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to convert a Demucs v3/4 model into safetensors
# python tool/tool/demucs2safetensors.py --yaml DEMUCS.YAML
# You must manually download the .th files and copy them to the YAML dir
# First version by Gemini 2.5 Pro
from contextlib import contextmanager
from copy import deepcopy
import argparse
import json
from pathlib import Path
from safetensors.torch import save_file
import sys
import torch
from typing import Dict
import yaml

try:
    # We need the original Demucs library to dequantize
    from demucs.states import set_state  # noqa: F401
    with_demuc_lib = True
except Exception:
    with_demuc_lib = False
import bootstrap  # noqa: F401
from source.utils.misc import cli_add_verbose, FractionEncoder, debugl
from source.utils.logger import main_logger, logger_set_standalone
from source.db.models_db import cli_add_db, get_download_url, ModelsDB
from source.db.hash import get_hash
import source.inference.Demucs as local_demucs_module
import source.inference.HDemucs as local_hdemucs_module
import source.inference.HTDemucs as local_htdemucs_module
MODULES_MAP = {'demucs': local_demucs_module,
               'demucs.demucs': local_demucs_module,
               'demucs.hdemucs': local_hdemucs_module,
               'demucs.htdemucs': local_htdemucs_module}
MAP = {'freq_encoder': 'encoder',
       'freq_decoder': 'decoder',
       'time_encoder': 'tencoder',
       'time_decoder': 'tdecoder'}
logger = main_logger


@contextmanager
def remap_module(modules_map):
    """
    A context manager to temporarily remap an old module name to a new one.
    This is useful for loading pickled objects that depend on old paths.
    """
    original_modules = {}
    for old_name, new_module in modules_map.items():
        original_modules[old_name] = sys.modules.get(old_name)
        sys.modules[old_name] = new_module
    try:
        yield
    finally:
        # Restore the original state
        for old_name, original_module in original_modules.items():
            if original_module is not None:
                sys.modules[old_name] = original_module
            else:
                # If the module wasn't there before, remove our patch
                del sys.modules[old_name]


def solve_simple_pt(yaml_data, pkg):
    """ PyTorch Audio lib has some raw models, we store the metadata in the YAML """
    klass = yaml_data.get('klass')
    if klass is None:
        main_logger.error("No `klass` in YAML")
        sys.exit(4)
    if klass == 'Demucs':
        klass = local_demucs_module.Demucs
    elif klass == 'HDemucs':
        klass = local_hdemucs_module.HDemucs
    elif klass == 'HTDemucs':
        klass = local_htdemucs_module.HTDemucs
    else:
        main_logger.error("Unknown model `klass` {klass}")
        sys.exit(4)
    args = yaml_data.get('args', {})
    kwargs = yaml_data.get('kwargs', {})

    # For PyTorch Audio model (very old code?)
    new_dict = {}
    for k, v in pkg.items():
        parts = k.split('.')
        gr = parts[0]
        if gr in MAP:
            k = MAP[gr] + '.' + '.'.join(parts[1:])
        new_dict[k] = v

    return klass, args, kwargs, new_dict


def convert_demucs_model(yaml_path_str: str, model_paths: list[str], output_path_str: str, all_metadata, data: Dict):
    """
    Loads an original Demucs model bag, extracts weights and all necessary metadata
    from the YAML and .th files, and saves it to a single, secure .safetensors file.
    """
    yaml_path = Path(yaml_path_str)
    output_path = Path(output_path_str)

    # 1. Load and parse the YAML file
    main_logger.info(f"\n--- 🚀 Starting conversion for {yaml_path.name} ---\n")
    main_logger.info(f"- Loading YAML definition from: {yaml_path}")
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)

    signatures = yaml_data['models']
    main_logger.info(f"- Found {len(signatures)} model signatures in YAML: {signatures}")

    # 2. Match the provided .th files to their signatures
    model_file_map = {}
    for path_str in model_paths:
        path = Path(path_str)
        # The signature is the part of the filename before the first '-' or '.'
        sig = path.stem.split('-')[0]
        if sig not in signatures:
            main_logger.warning(f"File {path.name} with signature {sig} is not listed in the YAML file. Skipping.")
            continue
        if sig in model_file_map:
            raise ValueError(f"Duplicate files found for signature {sig}.")
        model_file_map[sig] = path

    # Verify that all signatures from the YAML have a corresponding file
    if len(model_file_map) != len(set(signatures)):
        missing_sigs = set(signatures) - set(model_file_map.keys())
        raise FileNotFoundError(f"Missing model files for signatures: {missing_sigs}")

    main_logger.info("- Successfully mapped all signatures to model files.")

    # 3. Load model packages and extract metadata and state dicts
    # all_metadata = {}
    consolidated_state_dict = {}

    for sig in signatures:
        path = model_file_map[sig]
        main_logger.info(f"\n- Processing model '{sig}' from '{path.name}'...")

        # Use the context manager to perform the remap
        with remap_module(MODULES_MAP):
            # This is the only "unsafe" part, loading the original pickle file
            pkg = torch.load(path, map_location='cpu', weights_only=False)

        debugl(logger, 2, f"PyTorch data type is {type(pkg)}")
        if isinstance(pkg, dict) and 'klass' in pkg:
            klass, args, kwargs, state = pkg["klass"], pkg["args"], pkg["kwargs"], pkg["state"]
        else:
            klass, args, kwargs, state = solve_simple_pt(yaml_data, pkg)
        main_logger.info(f"  - Model class: {klass.__module__}.{klass.__name__}")

        # Dequantize if necessary by letting the original code handle it
        if state.get('__quantized'):
            if not with_demuc_lib:
                main_logger.error("Don't use quantized models. Look for the same model without `_q`")
                main_logger.error("Alternatively install the demucs Python module")
                sys.exit(3)
            else:
                main_logger.info("  - Model is quantized. Dequantizing weights...")
                model_instance = klass(*args, **kwargs)
                set_state(model_instance, state)
                clean_state_dict = model_instance.state_dict()
        else:
            clean_state_dict = state

        # Store this model's metadata, keyed by its signature
        all_metadata[sig] = json.dumps({
            'class_module': klass.__module__,
            'class_name': klass.__name__,
            'args': args,
            'kwargs': kwargs,
        }, cls=FractionEncoder)  # Use the custom encoder here

        # Add the weights to the consolidated dict, prefixed by signature
        for key, value in clean_state_dict.items():
            consolidated_state_dict[f"{sig}.{key}"] = value

    # 4. Add the top-level YAML data to the metadata
    all_metadata['is_bag_of_models'] = json.dumps(len(set(signatures)) > 1)
    all_metadata['signatures'] = json.dumps(signatures)
    if 'weights' in yaml_data:
        all_metadata['weights'] = json.dumps(yaml_data['weights'])
    if 'segment' in yaml_data:
        all_metadata['segment'] = str(yaml_data['segment'])

    # 5. Calculate the total number of parameters from the final state dict
    total_params = sum(p.numel() for p in consolidated_state_dict.values())
    main_logger.info(f"- Total number of parameters in the model: {total_params:,}")
    # Add the count as a string to the metadata dictionary
    data['params'] = all_metadata['params'] = str(total_params)

    # 6. Save the final .safetensors file
    main_logger.info(f"- \U0001F4BE Saving consolidated model and metadata to: {output_path}")
    save_file(consolidated_state_dict, output_path, metadata=all_metadata)
    main_logger.info("\n--- 🎉 Conversion complete! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Demucs .th models to a single .safetensors file.")

    parser.add_argument('--yaml', required=True, type=str, help="Path to the Demucs .yaml file.")
    parser.add_argument('--models', nargs='*', default=None, type=str,
                        help="Optional. Paths to .th files. If not provided, assumes they are "
                        "in the same directory as the YAML.")
    parser.add_argument('--output', default=None, type=str,
                        help="Optional. Path for the output .safetensors file. If not provided, "
                        "it's saved next to the YAML with the same name.")
    cli_add_db(parser)
    cli_add_verbose(parser)

    args = parser.parse_args()
    logger_set_standalone(args)
    main_logger.info("⚙️  PyTorch Demucs to Safetensors converter\n")

    # Get information about the YAML file in our database
    yaml_path = Path(args.yaml)
    db = ModelsDB(yaml_path.parent)
    known = db.get_filtered(model_t="Demucs")
    hash = get_hash(yaml_path)
    d = known.get_by_hash(hash)
    if d is None:
        # We don't have the hash for it
        d = known.get_by_file_name(yaml_path.name)
        if d is None:
            # We use some information from the database to populate the metadata so is better if we have the
            # information in the database
            main_logger.error(f"{yaml_path} not in data base, please add it first")
            sys.exit(3)
        else:
            main_logger.error(f"{yaml_path} in database, but with different hash")

    logger.info(f"Model description: {d['desc']}")

    # Logic for optional --models
    model_files = args.models
    if not model_files:
        main_logger.debug("No --models provided. Searching for .th files alongside the YAML...")
        yaml_dir = yaml_path.parent
        with open(yaml_path, 'r') as f:
            signatures = yaml.safe_load(f)['models']

        model_files = []
        for sig in set(signatures):  # Use set to avoid redundant searches
            found = list(yaml_dir.glob(f'{sig}*.th')) + list(yaml_dir.glob(f'{sig}*.pt'))
            if not found:
                raise FileNotFoundError(f"Could not automatically find a model file for signature '{sig}' in {yaml_dir}")
            if len(found) > 1:
                full_match = yaml_dir / (sig + '.th')  # UVR Demucs uses it
                if full_match in found:
                    found = [full_match]
                else:
                    main_logger.warning(f"Found multiple files for signature '{sig}', using the first one: {found[0]}")
            model_files.append(str(found[0]))
        main_logger.info(f"Automatically found model files: {model_files}")

    # Logic for optional --output
    output_file = args.output
    if not output_file:
        output_file = yaml_path.with_suffix('.safetensors')
        main_logger.info(f"Defaulting to: `{output_file}` (No --output provided)")
    else:
        output_file = Path(output_file)

    # Adjust the data to the converted version
    metadata = {}
    d = deepcopy(d)
    d["download"] = "Main/Demucs"
    d["name"] = Path(d["name"]).stem + ".safetensors"
    d["file_t"] = "safetensors"
    # Add it to the safetensors
    metadata["desc"] = d["desc"]
    metadata["download"] = get_download_url(d)
    metadata["file_t"] = "safetensors"
    metadata["model_t"] = "Demucs"
    metadata["name"] = d["name"]
    metadata["primary_stem"] = json.dumps(d["primary_stem"])
    metadata["project"] = "https://github.com/set-soft/AudioSeparation"

    convert_demucs_model(args.yaml, model_files, output_file, metadata, d)

    # Now update the DB
    db.remove(known.get_by_file_name(d["name"]))
    db.add(get_hash(output_file), d)
    db.save()
