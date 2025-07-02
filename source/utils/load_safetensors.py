# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Model loader helper
# Original code from Gemini 2.5 Pro
import logging
from safetensors.torch import load_file
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.load_safetensors")


def load_safetensors(model_path, model_run, device):
    logger.info("Loading PyTorch model from .safetensors file...")
    # 1. Load the state_dict from the file, EXPLICITLY forcing all tensors onto the CPU.
    state_dict = load_file(model_path, device="cpu")
    # 2. Load the CPU state_dict into the CPU model. This is now a safe operation.
    try:
        missing_keys, unexpected_keys = model_run.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing keys in state_dict for model_run: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in state_dict for model_run: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            logger.debug("All keys matched successfully.")
    except RuntimeError as e:
        logger.error(f"RuntimeError during model_run.load_state_dict: {e}")
        logger.error("This might indicate a mismatch between saved weights and model architecture.")
        raise
    return model_run
