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


def load_state_dict(model_run, state_dict):
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


def load_safetensors(model_path, model_run, device):
    logger.info("Loading PyTorch model from .safetensors file...")
    # 1. Load the state_dict from the file, EXPLICITLY forcing all tensors onto the CPU.
    state_dict = load_file(model_path, device="cpu")

    # 2. Load the CPU state_dict into the CPU model. This is now a safe operation.
    if hasattr(model_run, 'signatures'):
        # This is the way we store Demucs models
        signatures = model_run.signatures
        if len(signatures) == 1:
            # Single model
            sig = signatures[0]
            logger.debug(f"Single model with filtered keys, prefix: {sig}")
            # Just remove the prefix
            state_dict = {k[len(sig)+1:]: v for k, v in state_dict.items()}
            # The rest is as a regular model
        else:
            # Bag of models, load the keys for each sub-model
            logger.debug("Multiple models with filtered keys")
            for sub_model, sig in zip(model_run.models, signatures):
                prefix = f"{sig}."
                logger.debug(f"  - Filtering {prefix}")
                sub_state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                load_state_dict(sub_model, sub_state_dict)
            # Finished
            return model_run

    load_state_dict(model_run, state_dict)
    model_run.target_device = device

    return model_run
