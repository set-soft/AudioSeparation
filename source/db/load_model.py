# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Model loader helper
# Original code from Gemini 2.5 Pro
import logging
# Local imports
from .models_db import download_model
from ..inference.get_model import get_model
from ..utils.misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.load_model")


def show_model_parameters(d):
    logger.debug("Using model parameters:")
    logger.debug(f"  Frequency Dimension (dim_f): {d['mdx_dim_f_set']}")
    logger.debug(f"  Base Channels (ch): {d['channels']}")
    logger.debug(f"  U-Net Stages: {d['stages']}")


def load_model(d, device, models_dir):
    file_t = d['file_t'].lower()
    show_model_parameters(d)

    # Get the file name, download if necessary
    model_path = d.get('model_path')
    if model_path is None:
        # It means it wasn't on disk
        model_path = download_model(d, models_dir)

    # ONNX
    if file_t == "onnx":
        from ..utils.load_onnx import load_onnx
        model = load_onnx(model_path, device)
        # Store the same information we have in the PyTorch version
        model.dim_f = d['mdx_dim_f_set']
        model.ch = d['channels']
        model.num_stages = d['stages']
        return model

    # Safetensors
    if file_t == "safetensors":
        from ..utils.load_safetensors import load_safetensors
        return load_safetensors(model_path, get_model(d), device)

    # Other
    msg = f"Unknown file type {file_t}"
    logger.error(msg)
    raise ValueError(msg)
