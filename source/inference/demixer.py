# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Wrappers for the model and inference
import logging
import math
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
from ..utils.torch import model_to_target, get_offload_device
from .demucs_api import apply_model, BagOfModels

logger = logging.getLogger(f"{NODES_NAME}.demixer")
SAMPLE_RATE = 44100


class DemixerGeneric(object):
    def __init__(self, d, device, models_dir):
        super().__init__()
        self.d = d
        self.model_run = load_model(d, device, models_dir)
        self.device = device

    def set_device(self, device):
        if device == self.device:
            return
        self.device = device
        if hasattr(self.model_run, "target_device"):
            self.model_run = device


# ############################################################################################################################
# MDX-Net
# ############################################################################################################################


def show_inference_parameters(d):
    logger.debug("Using inference parameters:")
    logger.debug(f"  Frequency Bins (n_fft/2): {d['mdx_n_fft_scale_set']//2}")
    logger.debug(f"  Amplitude Compensation: {d['compensate']}")


class DemixerMDX(DemixerGeneric):
    def __init__(self, d, device, models_dir):
        super().__init__(d, device, models_dir)
        show_inference_parameters(d)
        self.sr = SAMPLE_RATE
        self.ch = 2

    def __call__(self, waveform, segments=None):
        if segments is None:
            # To make it compatible with Demucs
            segments = 1
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


# ############################################################################################################################
# Demucs
# ############################################################################################################################


def get_steps_for_demucs(model, wav, segment, shifts, overlap):
    segment = segment or model.segment
    segment_length = int(model.samplerate * segment)
    stride = int((1 - overlap) * segment_length)
    return math.ceil(wav.shape[-1] / stride) * (shifts + 1)


class DemixerDemucs(DemixerGeneric):
    def __init__(self, d, device, models_dir):
        super().__init__(d, device, models_dir)
        self.sr = self.model_run.samplerate
        # Demucs code will move the model to and from the device
        # The advantage is that it will be do it for the sub_models
        # So here we keep it offloaded and tell Demucs code to do the work
        self.model_run.target_device = get_offload_device()

    def get_steps(self, wav, segment, shifts, overlap):
        """ Tries to figure out how much steps we will need for inference """
        if isinstance(self.model_run, BagOfModels):
            total = 0
            for sub_model in self.model_run.models:
                steps = get_steps_for_demucs(sub_model, wav, segment, shifts, overlap)
                total += steps
                logger.debug(f"- Steps {steps}")
            logger.debug(f"Total steps {total}")
            return total
        steps = get_steps_for_demucs(self.model_run, wav, segment, shifts, overlap)
        logger.debug(f"Steps {steps}")
        return steps

    def demucs_callback(self, v):
        self.comfy_progress_bar.update(1)

    def __call__(self, waveform_tensor, segment=None, shifts=0, overlap=0.25):
        try:
            # --- 1. Normalize input shape to handle both batched and non-batched data ---
            if waveform_tensor.ndim == 2:
                # Input is [C, samples], add a batch dimension to make it [1, C, samples]
                logger.debug("Input is not batched. Adding a temporary batch dimension.")
                waveform_tensor = waveform_tensor.unsqueeze(0)
                input_was_batched = False
            elif waveform_tensor.ndim == 3:
                # Input is already batched [B, C, samples]
                input_was_batched = True
            else:
                raise ValueError(f"Unsupported waveform shape: {waveform_tensor.shape}. Expected 2 or 3 dimensions.")

            batch_size = waveform_tensor.shape[0]
            logger.info("🎛️  Performing demix...")

            model = self.model_run
            input_tensor_on_device = waveform_tensor.to(self.device)
            # Determine the segment size
            # 1. User selection
            # 2. Value in the config
            # 3. Auto: from each model (when None)
            forced_segment = segment
            if forced_segment is None:
                forced_segment = self.model_run.config_segment
                if forced_segment:
                    logger.debug(f"Using model provided segment size {forced_segment} s")
                else:
                    logger.debug("Using default segment size")
                    forced_segment = None
            else:
                logger.debug(f"Using user provided segment size {forced_segment} s")
            if with_comfy:
                comfy_progress_bar = comfy.utils.ProgressBar(self.get_steps(waveform_tensor, forced_segment, shifts, overlap))

            with model_to_target(model):
                separated_tensors = apply_model(
                    model,
                    input_tensor_on_device,
                    device=self.device,
                    segment=forced_segment,
                    shifts=shifts + 1,  # Shifts 0 is disabled, 1 is just one pass, 2 is 2 passes
                    overlap=overlap,
                    split=True,       # Enable chunking
                    progress=True,    # Show a progress bar in the console
                    callback=lambda x: (x.get('state') == 'end') and comfy_progress_bar.update(1) if with_comfy else None,
                )

            # Move the final result tensor back to the CPU before creating the output dicts.
            # This is good practice to free up VRAM for subsequent nodes.
            separated_tensors = separated_tensors.cpu()
            assert batch_size == separated_tensors.shape[0]

            # The output is [batch, sources, channels, samples].
            # But we will separate the stems and return a batch for each stem, so we need
            # [sources, batch, channels, samples].
            separated_sources = separated_tensors.permute(1, 0, 2, 3)

            # The model object tells us the names of the stems it produced
            model_stems = model.sources.copy()
            # UVR model is a 2 stems model with vocals and non_vocals, map it gracefully
            if model_stems[1] == 'non_vocals':
                model_stems[1] = 'other'
            logger.debug(f"Model produced {len(model_stems)} stems: {model_stems}")

            # Create a dictionary mapping the stem name to the audio tensor
            output_map = dict(zip(model_stems, separated_sources))

            # --- Silent audio for missing stems
            reference_tensor = None
            # Find the first valid tensor from our output to use as a shape reference
            for tensor in output_map.values():
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    reference_tensor = tensor
                    break

            # Determine the shape and sample rate for our silent audio fallback
            if reference_tensor is not None:
                # If we have a successful stem, use its properties
                ref_batch_size, ref_channels, ref_samples = reference_tensor.shape
            else:
                # EDGE CASE: All stems failed. Fall back to the input audio's properties.
                logger.warning("All stems failed. Using input audio shape for silence.")
                # input_audio['waveform'] has shape [batch, channels, samples]
                ref_batch_size, ref_channels, ref_samples = waveform_tensor.shape
            # ---

            # --- Gracefully create the 6 outputs ---
            # Iterate through our fixed RETURN_NAMES and get the corresponding tensor.
            # If a stem name doesn't exist in our output_map (e.g., 'guitar' for a 4-stem model),
            # the .get() method will return None, which ComfyUI handles correctly.
            final_outputs = []
            for stem_name in ("vocals", "drums", "bass", "other", "guitar", "piano"):
                output_tensor = output_map.get(stem_name, None)

                # If the stem was produced, wrap it back into an AUDIO dict.
                # If not, append None. ComfyUI handles None outputs correctly.
                if output_tensor is not None:
                    if not input_was_batched:
                        output_tensor = output_tensor.squeeze(0)
                    output_dict = {
                        "waveform": output_tensor,
                        "sample_rate": model.samplerate,
                        "stem": stem_name.capitalize()
                    }
                    final_outputs.append(output_dict)
                else:
                    logger.debug(f"Stem '{stem_name}' not produced by model. Returning silence.")
                    # Create a silent tensor with the correct shape, device, and dtype
                    silent_waveform = torch.zeros(ref_batch_size, ref_channels, ref_samples, dtype=torch.float32)

                    # Wrap the silent tensor in the ComfyUI AUDIO dictionary format
                    silent_dict = {
                        "waveform": silent_waveform,
                        "sample_rate": model.samplerate,
                        "stem": stem_name.capitalize()
                    }
                    final_outputs.append(silent_dict)

            return tuple(final_outputs)

        except Exception as e:
            logger.error(f"Error during separation: {str(e)}")
            raise e


def get_demixer(d, device, models_dir):
    model_t = d['model_t'].lower()
    if model_t == "mdx":
        return DemixerMDX(d, device, models_dir)
    elif model_t == "demucs":
        return DemixerDemucs(d, device, models_dir)
    msg = f"Unknown model type `{model_t}`"
    logger.error(msg)
    raise ValueError(msg)
