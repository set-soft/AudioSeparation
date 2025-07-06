# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
import os
from typing import Dict
# ComfyUI imports
import folder_paths  # ComfyUI's way to access model paths
# Local imports
from .source.utils.logger import main_logger
from .source.utils.load_audio import audio_get_channels, force_stereo, force_sample_rate
from .source.utils.torch import get_torch_device_options, get_canonical_device
from .source.utils.comfy_node_action import send_node_action
from .source.db.models_db import ModelsDB
from .source.inference.demixer import get_demixer

DEF_MODEL = 'Kim_Vocal_2.safetensors'
DEF_ENTRY = 'Default'
MODELS_DIR = os.path.join(folder_paths.models_dir, "audio", "MDX")
DEMUCS_DIR = os.path.join(folder_paths.models_dir, "audio", "Demucs")
models_db_mdx = ModelsDB(MODELS_DIR)
models_db_demucs = ModelsDB(DEMUCS_DIR)
logger = main_logger


class AudioSeparateVocals:
    PRIMARY_STEM = 'Vocals'
    MODEL_T = 'MDX'
    FILE_T = 'safetensors'
    DEFAULT_MODEL = "Kim_Vocal_2.safetensors"
    models_db = models_db_mdx

    @classmethod
    def _get_available_audio_models(cls):
        # Refresh the database
        cls.models_db.refresh()
        # Filter the models this node can handle
        cls.models_filtered = cls.models_db.get_filtered(primary_stem=cls.PRIMARY_STEM, model_t=cls.MODEL_T, file_t=cls.FILE_T,
                                                         default=cls.DEFAULT_MODEL, repeat_dl=True)
        # We add any model downloaded and memorized by the GUI
        return cls.models_filtered.get_display_names()

    @classmethod
    def INPUT_TYPES(cls):
        device_options, default_device = get_torch_device_options()
        return {
            "required": {
                "input_sound": ("AUDIO",),
                "model": (cls._get_available_audio_models(),),  # Dropdown for model selection
                "segments": ("INT", {
                    "default": 1,        # Default value
                    "min": 1,            # Minimum allowed value
                    "max": 64,           # Maximum allowed value (set a reasonable practical max)
                    "step": 1,           # Step for slider/spinbox
                    "display": "slider"  # How to display: "number" or "slider"
                }),
                "target_device": (device_options, {
                    "default": default_device,
                    "tooltip": "The device (CPU or CUDA) to which the projection layer will be assigned for computation."}),
            }
        }

    RETURN_TYPES = ("AUDIO", "AUDIO",)
    RETURN_NAMES = (PRIMARY_STEM, "Complement",)
    FUNCTION = "execute"
    CATEGORY = "audio/separation"
    DESCRIPTION = "Separates vocals using MDX-Net networks"
    UNIQUE_NAME = "AudioSeparateVocals"
    DISPLAY_NAME = "Vocals using MDX"

    def __init__(self):
        super().__init__()
        self.demixer = None

    def execute(self, input_sound: Dict, model: str, segments: int, target_device: str):
        # Get information for the selected model
        main_logger.info(f"Selected model: {model}")
        model_data = self.models_filtered.get_by_display_name(model)
        if model_data is None:
            raise ValueError("Unknown model selected, please refresh pressing `R` and select another")
        model_path = model_data.get('model_path')

        # Create or recycle a demixer
        device = get_canonical_device(target_device)
        if self.demixer is None or self.demixer.d['hash'] != model_data['hash']:
            # New demixer
            logger.debug("Creating a new demixer object")
            # This will load the model, optionally downloading it
            self.demixer = get_demixer(model_data, device, MODELS_DIR)
        else:
            # Update the device
            self.demixer.set_device(device)

        # Handle a change in the icon of the model name
        if model_path is None:
            # Was downloaded
            send_node_action("change_widget", "model", model_data['indicator'] + model_data['filtered_name'])

        # Match channels and S/R
        waveform = input_sound['waveform']
        sample_rate = input_sound['sample_rate']
        if audio_get_channels(waveform) == 1 and self.demixer.ch == 2:
            waveform = force_stereo(waveform)
        if sample_rate != self.demixer.sr:
            waveform = force_sample_rate(waveform, sample_rate, self.demixer.sr)

        # Demix
        wavs = self.demixer(waveform, segments)

        return (wavs[0], wavs[1],)


class AudioSeparateInstrumental(AudioSeparateVocals):
    PRIMARY_STEM = 'Instrumental'
    DEFAULT_MODEL = "Kim_Inst.safetensors"
    DESCRIPTION = "Separates instruments using MDX-Net networks"
    UNIQUE_NAME = "AudioSeparateInstrumental"
    DISPLAY_NAME = "Instrumental using MDX"
    RETURN_NAMES = (PRIMARY_STEM, "Complement",)


class AudioSeparateBass(AudioSeparateVocals):
    PRIMARY_STEM = 'Bass'
    DEFAULT_MODEL = "kuielab_b_bass.safetensors"
    DESCRIPTION = "Separates bass using MDX-Net networks"
    UNIQUE_NAME = "AudioSeparateBass"
    DISPLAY_NAME = "Bass using MDX"
    RETURN_NAMES = (PRIMARY_STEM, "Complement",)


class AudioSeparateDrums(AudioSeparateVocals):
    PRIMARY_STEM = 'Drums'
    DEFAULT_MODEL = "kuielab_b_drums.safetensors"
    DESCRIPTION = "Separates drums using MDX-Net networks"
    UNIQUE_NAME = "AudioSeparateDrums"
    DISPLAY_NAME = "Drums using MDX"
    RETURN_NAMES = (PRIMARY_STEM, "Complement",)


class AudioSeparateVarious(AudioSeparateVocals):
    PRIMARY_STEM = ["Other", "Reverb"]
    DEFAULT_MODEL = "Reverb_HQ_By_FoxJoy.safetensors"
    DESCRIPTION = "Misc. separators using MDX-Net networks"
    UNIQUE_NAME = "AudioSeparateVarious"
    DISPLAY_NAME = "Various using MDX"
    RETURN_NAMES = ("Main", "Complement",)


class AudioSeparateDemucs(AudioSeparateVocals):
    PRIMARY_STEM = None
    MODEL_T = 'Demucs'
    FILE_T = 'safetensors'
    DEFAULT_MODEL = "htdemucs_ft.safetensors"
    models_db = models_db_demucs

    @classmethod
    def INPUT_TYPES(cls):
        device_options, default_device = get_torch_device_options()
        return {
            "required": {
                "input_sound": ("AUDIO",),
                "model": (cls._get_available_audio_models(),),  # Dropdown for model selection
                "shifts": ("INT", {
                    "default": 0, "min": 0, "max": 16, "step": 1, "display": "slider",
                    "tooltip": "Number of random shifts for equivariant stabilization.\n"
                               "Higher values improve quality but are slower. 0 disables it."
                }),
                "overlap": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 0.99, "step": 0.01, "display": "slider",
                    "tooltip": "Amount of overlap between audio chunks.\n"
                               "Higher values can reduce stitching artifacts but are slower."
                }),
                "custom_segment": ("BOOLEAN", {
                    "default": False,
                    "label_on": "enabled",
                    "label_off": "disabled",
                    "tooltip": "Enable to override the model's default segment length.\n"
                               "Disabling uses the recommended length from the model file.\n"
                               "Useful for HDemucs and Demucs models, not much for HTDemucs."
                }),
                "segment": ("INT", {
                    "default": 44, "min": 10, "max": 120, "step": 1, "display": "slider",
                    "tooltip": "Length of audio chunks to process at a time (in seconds).\n"
                               "Higher values need more VRAM but can improve quality."
                }),
                "target_device": (device_options, {
                    "default": default_device,
                    "tooltip": "The device (CPU or CUDA) to which the projection layer will be assigned for computation."}),
            }
        }

    # Define the output names. We define all 6 possible stems.
    # The execution logic will handle returning 'None' for unused outputs.
    RETURN_NAMES = ("vocals", "drums", "bass", "other", "guitar", "piano")
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO", "AUDIO")
    DESCRIPTION = "Demucs Audio Separator (4/6 stems)"
    UNIQUE_NAME = "AudioSeparateDemucs"
    DISPLAY_NAME = "Demucs Audio Separator"

    def execute(self, input_sound: Dict, model: str, shifts: int, overlap: float, custom_segment: bool, segment: int,
                target_device: str):
        # Get information for the selected model
        main_logger.info(f"Selected model: {model}")
        model_data = self.models_filtered.get_by_display_name(model)
        if model_data is None:
            raise ValueError("Unknown model selected, please refresh pressing `R` and select another")
        model_path = model_data.get('model_path')

        # Create or recycle a demixer
        device = get_canonical_device(target_device)
        if self.demixer is None or self.demixer.d['hash'] != model_data['hash']:
            # New demixer
            logger.debug("Creating a new demixer object")
            # This will load the model, optionally downloading it
            self.demixer = get_demixer(model_data, device, DEMUCS_DIR)
        else:
            # Update the device
            self.demixer.set_device(device)

        # Handle a change in the icon of the model name
        if model_path is None:
            # Was downloaded
            send_node_action("change_widget", "model", model_data['indicator'] + model_data['filtered_name'])

        # Match channels and S/R
        waveform = input_sound['waveform']
        sample_rate = input_sound['sample_rate']
        if audio_get_channels(waveform) == 1 and self.demixer.ch == 2:
            waveform = force_stereo(waveform)
        if sample_rate != self.demixer.sr:
            waveform = force_sample_rate(waveform, sample_rate, self.demixer.sr)

        # Demix
        wavs = self.demixer(waveform, shifts=shifts, overlap=overlap, segment=segment if custom_segment else None)

        return tuple(wavs)
