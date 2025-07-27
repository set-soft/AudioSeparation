# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation

# This is our first import so we initialize SeCoNoHe
from .src.nodes import nodes, main_logger, __version__
from seconohe.register_nodes import register_nodes
from seconohe import JS_PATH


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes(main_logger, [nodes], version=__version__)
WEB_DIRECTORY = JS_PATH
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
