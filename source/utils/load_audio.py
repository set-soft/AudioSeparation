# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Audio load helper
# Original code from Gemini 2.5 Pro
import logging
import torch
import torchaudio
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.load_audio")


def audio_get_channels(waveform):
    dim_c = 0 if waveform.ndim == 2 else 1
    return waveform.shape[dim_c]


def force_stereo(waveform):
    dim_c = 0 if waveform.ndim == 2 else 1
    if waveform.shape[dim_c] == 1:
        logger.debug("Audio is mono, converting to fake stereo.")
        return torch.cat([waveform, waveform], dim=dim_c)
    return waveform


def force_sample_rate(waveform, orig_freq, new_freq):
    logger.debug(f"Resampling from {orig_freq} Hz to {new_freq} Hz.")
    resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
    return resampler(waveform)


def load_audio(file_path, force_sr=None, force_stereo=False):
    """ Loads an audio file, optionally converts it to stereo float, and resamples to force_sr. """
    logger.info(f"🎵 Loading audio file: {file_path}")
    try:
        waveform, sample_rate = torchaudio.load(file_path, normalize=True)
        # Ensure stereo
        if force_stereo and audio_get_channels(waveform) == 1:
            waveform = force_stereo(waveform)
        # Ensure 44.1 kHz or other S/R
        if force_sr is not None and sample_rate != force_sr:
            waveform = force_sample_rate(waveform, sample_rate, force_sr)
        return waveform, sample_rate
    except Exception as e:
        logger.error(f"💥 Failed to load audio file: {e}")
        raise
