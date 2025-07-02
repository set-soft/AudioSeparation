# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
# This helper is used to load a class from an arbitrary file
# Gemini 2.5 Pro code
import importlib
import logging
import os
import sys
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.load_class")


# Helper to dynamically import the target PyTorch model class
def import_model_class(location_string: str):
    """
    Dynamically imports a PyTorch model class from a file path and class name.

    The location_string is expected to be in the format:
    'path/to/your/file.py:ClassName'
    """
    module_dir = None
    try:
        # 1. Split the input string into a file path and a class name
        filepath, class_name = location_string.split(':')

        # Check if the file exists before proceeding
        if not os.path.exists(filepath):
            logger.error(f"File not found at '{filepath}'.")
            sys.exit(1)

        # 2. Get the directory and the module name from the file path
        module_dir, module_file = os.path.split(filepath)
        module_name = os.path.splitext(module_file)[0]

        # Add the directory to sys.path to allow Python to find it
        # Add it to the beginning to ensure it's checked first
        sys.path.insert(0, module_dir)

        # 3. Import the module
        logger.info(f"Importing module '{module_name}' from '{module_dir}'...")
        module = importlib.import_module(module_name)

        # 4. Get the class from the imported module
        model_class = getattr(module, class_name)

    except (ValueError, ImportError, AttributeError, FileNotFoundError) as e:
        logger.error(f"Could not import model class from '{location_string}'.")
        logger.error("Please ensure the format is 'path/to/file.py:ClassName'.")
        logger.error(f"Original error: {e}")
        sys.exit(1)

    finally:
        # 5. Clean up by removing the path we added.
        # This is crucial to avoid polluting the user's environment.
        if module_dir is not None and module_dir in sys.path:
            sys.path.pop(0)

    logger.info(f"Successfully imported class '{class_name}'.")
    return model_class
