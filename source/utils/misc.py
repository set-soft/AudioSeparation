# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
from fractions import Fraction
import json
import logging

NODES_NAME = "AudioSeparation"
NODES_DEBUG_VAR = NODES_NAME.upper() + "_NODES_DEBUG"


def debugl(logger, level, msg):
    if logger.getEffectiveLevel() <= logging.DEBUG - (level - 1):
        logger.debug(msg)


def cli_add_verbose(parser):
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose output to see details of the process.")


class FractionEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Fraction):
            # Represent the Fraction as a dictionary with a type hint
            return {'_type': 'Fraction', 'numerator': obj.numerator, 'denominator': obj.denominator}
        return super().default(obj)


def json_object_hook(d):
    """The decoder hook for our custom Fraction serialization."""
    if d.get('_type') == 'Fraction':
        return Fraction(d['numerator'], d['denominator'])
    return d
