# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Audio save helper
# Original code from Gemini 2.5 Pro
import logging
import os
import torchaudio
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.save_audio")


def save_audio(tensor, sample_rate, file_path, output_format):
    """
    Saves a tensor as an audio file, using the most basic and compatible
    torchaudio.save signature to avoid all version-specific errors.
    """
    logger.info(f"💾 Saving audio to: {file_path}")

    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        # The most compatible signature is simply:
        # torchaudio.save(filepath, src, sample_rate, format)
        # We pass the format string directly. The ffmpeg backend will use
        # a reasonable default quality for MP3 encoding.
        torchaudio.save(file_path, tensor.cpu(), sample_rate, format=output_format.lower())

        logger.info("✅ Save complete.")
    except Exception as e:
        if "ffmpeg" in str(e).lower() and "Unknown encoder" not in str(e):
            logger.error("💥 Failed to save audio file. This might be because the 'ffmpeg' backend is not available.")
            logger.error("Please ensure FFmpeg is installed and accessible in your system's PATH.")
        else:
            logger.error(f"💥 Failed to save audio file: {e}")
        raise
