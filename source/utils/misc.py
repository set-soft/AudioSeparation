# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
import logging

NODES_NAME = "AudioSeparation"
NODES_DEBUG_VAR = NODES_NAME.upper() + "_NODES_DEBUG"


def debugl(logger, level, msg):
    if logger.getEffectiveLevel() <= logging.DEBUG - (level - 1):
        logger.debug(msg)


def cli_add_verbose(parser):
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose output to see details of the process.")
