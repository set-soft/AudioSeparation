# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
import argparse
from fractions import Fraction
import json
from .. import __version__, __copyright__, __license__, __author__


def cli_add_verbose(parser):
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Enable verbose output to see details of the process.")


class PrintVersionAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Format the version information
        version_info = f"""{parser.prog} (Audio Separation) {__version__}
{__copyright__}
{__license__}
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.

Written by {__author__}"""
        print(version_info)
        # Exit the parser
        parser.exit()


def cli_add_version(parser, prog_name):
    parser.add_argument('-V', '--version', help="Show version and copyright information and exit",
                        action=PrintVersionAction)


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
