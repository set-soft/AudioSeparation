# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Helper to get a model from the correct class
import importlib
import json
import logging
from safetensors import safe_open
from .MDX_Net import MDX_Net
from ..utils.misc import NODES_NAME, json_object_hook, get_debug_level
# Demucs class imports
from .demucs_api import BagOfModels
from .demucs_log_helper import DemucsModelInfo

logger = logging.getLogger(f"{NODES_NAME}.get_model")


def get_metadata(file_path, d=None):
    """ Read the metadata from a safetensors file """
    logger.debug(f"Reading metadata from {file_path}")
    metadata = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if not metadata:
        raise ValueError(f"Could not read metadata from safetensors file: {file_path}")

    if d is None:
        return metadata
    # Is this a child model?
    parent = d.get('parent')
    if parent:
        # Ok, this is a child model changing details of a parent model
        # Currently used by Demucs models to create simplified versions of the same model
        metadata['is_bag_of_models'] = d.get('is_bag_of_models', 'false')
        try:
            metadata['signatures'] = d['signatures']
        except KeyError:
            logger.error("Child model without signatures")
            raise
        metadata['segment'] = d.get('segment', '0')

    return metadata


def get_hyperparameter(metadata, parameter, as_type, default=None, warn_diff=True):
    value = metadata.get(parameter)
    if value is None:
        if default is None:
            raise ValueError(f"Missing `{parameter}` hyperparameter")
        return default
    if as_type == "int":
        value = int(value)
    if warn_diff and value != default:
        logger.warning(f"Hyperparameter mismatch: database = {default}, metadata = {value}")
    return value


def get_mdx_model(d):
    """ Create an MDX_Net object with the specified parameters """
    # Check the file is consistent we our data base
    metadata = get_metadata(d['model_path'], d)
    dim_f = get_hyperparameter(metadata, 'mdx_dim_f_set', "int", d['mdx_dim_f_set'])
    channels = get_hyperparameter(metadata, 'channels', "int", d['channels'])
    stages = get_hyperparameter(metadata, 'stages', "int", d['stages'])
    # Create a class with this parameters
    return MDX_Net(dim_f=dim_f, ch=channels, num_stages=stages)


def get_model_path(d):
    parent = d.get("parent")
    if parent is None:
        return d.get('model_path')
    return parent.get('model_path')


def get_demucs_model(d):
    """ Create a Demucs, HDemucs (Hybrid) or HTDemucs (Hybrid Transformer) object.
        All metadata comes from the safetensors """
    file_path = get_model_path(d)

    # 1. First, open the file safely to read only the metadata header.
    metadata = get_metadata(file_path, d)

    is_bag = json.loads(metadata.get('is_bag_of_models', 'false'))
    signatures = json.loads(metadata['signatures'])

    sub_models = []
    for sig in set(signatures):  # Use set to only instantiate each unique architecture once
        model_meta_str = metadata.get(sig)
        if not model_meta_str:
            raise ValueError(f"Metadata for signature '{sig}' not found in safetensors file.")

        # Use the object_hook here to reconstruct Fraction objects automatically
        model_meta = json.loads(model_meta_str, object_hook=json_object_hook)

        class_module, class_name = model_meta['class_module'], model_meta['class_name']
        args, kwargs = model_meta['args'], model_meta['kwargs']

        logger.debug(f"  - Reconstructing architecture for '{sig}': {class_module}.{class_name}")
        assert '.' not in class_name, "Security check failed, won't import a file outside my directory"
        # Import the class from a module in this dir with the same name as the class
        local_class_module = '.'.join(__name__.split('.')[:-1]) + "." + class_name
        logger.debug(f"  - Redirecting class: {class_module} -> {local_class_module}")
        module = importlib.import_module(local_class_module)
        klass = getattr(module, class_name)

        sub_models.append({'sig': sig, 'model': klass(*args, **kwargs)})

    # Create a mapping from signature to model instance
    model_map = {m['sig']: m['model'] for m in sub_models}

    # Re-order the models to match the YAML's signature list
    ordered_models = [model_map[sig] for sig in signatures]
    segment = float(metadata.get('segment', '0'))

    if not is_bag:
        weights = None
        final_model = ordered_models[0]
        final_model.signatures = signatures
    else:
        logger.debug("Rebuilding BagOfModels container...")
        weights = json.loads(metadata.get('weights', 'null'))
        final_model = BagOfModels(ordered_models, weights=weights, segment=segment)
        final_model.signatures = signatures

    final_model.config_segment = segment

    debug_level = get_debug_level(logger)
    if debug_level >= 1:
        # Show some information of the resulting model
        logger.debug("Model information:")
        logger.debug(f"- Total models {len(ordered_models)}")
        if weights is not None and len(weights) != len(ordered_models):
            raise ValueError(f"Invalid weights for {len(ordered_models)} models: {weights}")
        for n, m in enumerate(ordered_models):
            tp = m.__class__.__name__
            kwargs = model_meta['kwargs']
            w = weights[n] if weights else None
            if n == 0:
                ref_sources = m.sources
            else:
                if m.sources != ref_sources:
                    logger.error("The sub-model outputs doesn't match")
            if w and len(m.sources) != len(w):
                raise ValueError(f"Invalid {w} weights for {m.sources} sources")
            num = -1 if len(ordered_models) == 1 else n
            DemucsModelInfo(num, tp, kwargs, logger.debug, w, extra=debug_level > 1)

    return final_model


def get_model(d):
    model_t = d['model_t'].lower()
    if model_t == "mdx":
        return get_mdx_model(d)
    elif model_t == "demucs":
        return get_demucs_model(d)
    msg = f"Unknown model type `{model_t}`"
    logger.error(msg)
    raise ValueError(msg)
