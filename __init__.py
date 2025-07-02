# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
import inspect
import logging
from .source.utils.misc import NODES_NAME
from . import nodes  # noqa: E402

init_logger = logging.getLogger(f"{NODES_NAME}.__init__")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def register_nodes(module):
    suffix = " " + module.SUFFIX if hasattr(module, "SUFFIX") else ""
    if suffix:
        suffix = " " + suffix
    for name, obj in inspect.getmembers(module):
        if not inspect.isclass(obj) or not hasattr(obj, "INPUT_TYPES"):
            continue
        assert hasattr(obj, "UNIQUE_NAME"), f"No name for {obj.__name__}"
        NODE_CLASS_MAPPINGS[obj.UNIQUE_NAME] = obj
        NODE_DISPLAY_NAME_MAPPINGS[obj.UNIQUE_NAME] = obj.DISPLAY_NAME + suffix


register_nodes(nodes)

init_logger.info(f"Registering {len(NODE_CLASS_MAPPINGS)} node(s).")
init_logger.debug(f"{list(NODE_DISPLAY_NAME_MAPPINGS.values())}")

WEB_DIRECTORY = "./js"
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
