# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: CC BY-NC-SA 4.0
# Project: ComfyUI-Float_Optimized
import contextlib  # For context manager
import logging
import torch
try:
    import comfy.model_management as mm
    with_comfy = True
except Exception:
    with_comfy = False
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.torch")


def get_torch_device_options():
    # We always have CPU
    default = "cpu"
    options = [default]
    # Do we have CUDA?
    if torch.cuda.is_available():
        default = "cuda"
        options.append(default)
        for i in range(torch.cuda.device_count()):
            options.append(f"cuda:{i}")  # Specific CUDA devices
    # Is this a Mac?
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        options.append("mps")
        if default == "cpu":
            default = "mps"
    return options, default


def get_offload_device():
    return mm.unet_offload_device() if with_comfy else torch.device("cpu")


def get_canonical_device(device: str | torch.device) -> torch.device:
    """Converts a device string or object into a canonical torch.device object with an explicit index."""
    if not isinstance(device, torch.device):
        device = torch.device(device)

    # If it's a CUDA device and no index is specified, get the default one.
    if device.type == 'cuda' and device.index is None:
        # NOTE: This adds a dependency on torch.cuda.current_device()
        # The first solution is often better as it doesn't need this.
        return torch.device(f'cuda:{torch.cuda.current_device()}')
    return device


# ##################################################################################
# # Helper for inference (Target device, offload, eval, no_grad and cuDNN Benchmark)
# ##################################################################################

@contextlib.contextmanager
def model_to_target(model):
    """
    Consolidated context manager for model device placement and inference state.

    - Moves the model to its designated `model.target_device`.
    - Sets `torch.backends.cudnn.benchmark` based on `model.cudnn_benchmark_setting` if available.
    - Sets the model to `eval()` mode.
    - Wraps the operation in a `torch.no_grad()` context.
    - Offloads the model to the CPU (`mm.unet_offload_device()`) afterwards.
    """
    if not isinstance(model, torch.nn.Module):
        with torch.no_grad():
            yield  # The code inside the 'with' statement runs here
        return

    # 1. Determine target device from the model object
    try:
        target_device = model.target_device
        assert isinstance(target_device, torch.device)
    except Exception as e:
        logger.warning(f"model_to_target: Could not get 'target_device' from model ({e}). "
                       "Defaulting to model's current device.")
        target_device = next(model.parameters()).device

    # 2. Get CUDNN benchmark setting from the model object (optional)
    # Use hasattr as this is an optional setting that not all models might have.
    cudnn_benchmark_enabled = None  # Default is to keep the current setting
    if hasattr(model, 'cudnn_benchmark_setting'):
        cudnn_benchmark_enabled = model.cudnn_benchmark_setting

    original_device = next(model.parameters()).device
    original_cudnn_benchmark_state = None
    is_cuda_target = target_device.type == 'cuda'

    try:
        # 3. Manage cuDNN benchmark state
        if (cudnn_benchmark_enabled is not None and is_cuda_target and hasattr(torch.backends, 'cudnn') and
           torch.backends.cudnn.is_available()):
            if torch.backends.cudnn.benchmark != cudnn_benchmark_enabled:
                original_cudnn_benchmark_state = torch.backends.cudnn.benchmark
                torch.backends.cudnn.benchmark = cudnn_benchmark_enabled
                logger.debug(f"Temporarily set cuDNN benchmark to {torch.backends.cudnn.benchmark}")

        # 4. Move model to target device if not already there
        if original_device != target_device:
            logger.debug(f"Moving model from `{original_device}` to target device `{target_device}` for inference.")
            model.to(target_device)

        # 5. Set to eval mode and disable gradients for the operation
        model.eval()
        with torch.no_grad():
            yield  # The code inside the 'with' statement runs here

    finally:
        # 6. Restore original cuDNN benchmark state
        if original_cudnn_benchmark_state is not None:
            # This check is sufficient because it will only be not None if we set it inside the try block
            torch.backends.cudnn.benchmark = original_cudnn_benchmark_state
            logger.debug(f"Restored cuDNN benchmark to {original_cudnn_benchmark_state}")

        # 7. Offload model back to CPU
        if with_comfy:
            offload_device = get_offload_device()
            current_device_after_yield = next(model.parameters()).device
            if current_device_after_yield != offload_device:
                logger.debug(f"Offloading model from `{current_device_after_yield}` to offload device `{offload_device}`.")
                model.to(offload_device)
                # Clear cache if we were on a CUDA device
                if 'cuda' in str(current_device_after_yield):
                    torch.cuda.empty_cache()
