# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Wrappers for the model and inference
import logging
import torch
# ComfyUI imports
try:
    import comfy.utils
    with_comfy = True
except Exception:
    with_comfy = False
# Local imports
from .stft import stft_chunk_process, stft_get_chunks
from ..db.load_model import load_model
from ..utils.misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.demixer")
SAMPLE_RATE = 44100


def show_inference_parameters(d):
    logger.debug("Using inference parameters:")
    logger.debug(f"  Frequency Bins (n_fft/2): {d['mdx_n_fft_scale_set']//2}")
    logger.debug(f"  Amplitude Compensation: {d['compensate']}")


class DemixerMDX(object):
    def __init__(self, d, device, models_dir):
        self.d = d
        self.model_run = load_model(d, device, models_dir)
        self.device = device
        show_inference_parameters(d)
        self.sr = SAMPLE_RATE
        self.ch = 2

    def __call__(self, waveform, segments=1):
        dim_t = (2 ** self.d['mdx_dim_t_set']) * segments
        try:
            # --- 1. Normalize input shape to handle both batched and non-batched data ---
            if waveform.ndim == 2:
                # Input is [C, samples], add a batch dimension to make it [1, C, samples]
                logger.debug("Input is not batched. Adding a temporary batch dimension.")
                waveform = waveform.unsqueeze(0)
                input_was_batched = False
            elif waveform.ndim == 3:
                # Input is already batched [B, C, samples]
                input_was_batched = True
            else:
                raise ValueError(f"Unsupported waveform shape: {waveform.shape}. Expected 2 or 3 dimensions.")

            batch_size = waveform.shape[0]
            logger.info("🎛️  Performing demix...")

            # Lists to store the separated stems from each item in the batch
            list_of_main_stems = []
            list_of_complement_stems = []

            # ComfyUI progress bar
            progress_bar_ui = None
            if with_comfy:
                chunks = stft_get_chunks(waveform.shape[2], self.d['mdx_n_fft_scale_set'], segment_size=dim_t)
                chunks *= batch_size
                progress_bar_ui = comfy.utils.ProgressBar(chunks)

            # --- 2. Iterate through the batch ---
            for i, single_waveform in enumerate(waveform):
                # single_waveform has shape [C, samples]
                logger.debug(f"Processing item {i+1}/{batch_size}...")

                # Process this single waveform
                main_wav = stft_chunk_process(single_waveform, self.d, self.model_run, self.device, segment_size=dim_t,
                                              progress_bar_ui=progress_bar_ui)
                complement_wav = single_waveform - main_wav

                # Add the results to our lists
                list_of_main_stems.append(main_wav)
                list_of_complement_stems.append(complement_wav)

            # --- 3. Stack the results into single batch tensors ---
            # torch.stack creates a new dimension (the batch dimension) from a list of tensors
            stacked_main_stems = torch.stack(list_of_main_stems, dim=0)
            stacked_complement_stems = torch.stack(list_of_complement_stems, dim=0)
            # Both will now have shape [B, C, samples]

            # --- 4. Denormalize output shape if original input was not batched ---
            if not input_was_batched:
                logger.debug("Squeezing batch dimension from output to match non-batched input.")
                stacked_main_stems = stacked_main_stems.squeeze(0)
                stacked_complement_stems = stacked_complement_stems.squeeze(0)

            return [{'waveform': stacked_main_stems, 'sample_rate': SAMPLE_RATE, 'stem': self.d['primary_stem']},
                    {'waveform': stacked_complement_stems, 'sample_rate': SAMPLE_RATE, 'stem': 'Complement'}]

        except Exception as e:
            logger.error(f"Error during separation: {str(e)}")
            raise e


def get_demixer(d, device, models_dir):
    # Currently just MDX
    return DemixerMDX(d, device, models_dir)
