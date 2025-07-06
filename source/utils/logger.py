# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
import os
import sys
import logging
from .misc import NODES_NAME, NODES_DEBUG_VAR

no_colorama = False
try:
    from colorama import init as colorama_init, Fore, Back, Style
except ImportError:
    no_colorama = True
# If colorama isn't installed use an ANSI basic replacement
if no_colorama:
    from .ansi import Fore, Back, Style  # noqa: F811
else:
    colorama_init()

# Used for tools
standalone_mode = False

white = Fore.WHITE + Style.BRIGHT
yellow = Fore.YELLOW + Style.BRIGHT
red = Fore.RED + Style.BRIGHT
red_alarm = Fore.RED + Back.WHITE + Style.BRIGHT
cyan = Fore.CYAN + Style.BRIGHT
reset = Style.RESET_ALL
# format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s "
#          "(%(filename)s:%(lineno)d)"
format = f"[{NODES_NAME} %(levelname)s] %(message)s (%(name)s - %(filename)s:%(lineno)d)"
format_simple = f"[{NODES_NAME}] %(message)s"
FORMATS = {
    logging.DEBUG: cyan + format + reset,
    logging.INFO: white + format_simple + reset,
    logging.WARNING: yellow + format + reset,
    logging.ERROR: red + format + reset,
    logging.CRITICAL: red_alarm + format + reset
}
format = "[%(levelname)s] %(message)s (%(name)s - %(filename)s:%(lineno)d)"
format_simple = "%(message)s"
if not sys.stdout.isatty():
    white = yellow = red = red_alarm = cyan = reset = ""
FORMATS_STANDALONE = {
    logging.DEBUG: cyan + format + reset,
    logging.INFO: white + format_simple + reset,
    logging.WARNING: yellow + format + reset,
    logging.ERROR: red + format + reset,
    logging.CRITICAL: red_alarm + format + reset
}


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    def __init__(self):
        super(logging.Formatter, self).__init__()

    def format(self, record):
        formats = FORMATS_STANDALONE if standalone_mode else FORMATS
        log_fmt = formats.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# Create a new logger
logger = logging.getLogger(NODES_NAME)
logger.propagate = False

# Add handler if we don't have one.
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CustomFormatter())
    logger.addHandler(handler)

# ######################
# Logger setup
# ######################
# 1. Determine the ComfyUI global log level (influenced by --verbose)
main_logger = logger
comfy_root_logger = logging.getLogger('comfy')
effective_comfy_level = logging.getLogger().getEffectiveLevel()
# 2. Check our custom environment variable for more verbosity
try:
    nodes_debug_env = int(os.environ.get(NODES_DEBUG_VAR, "0"))
except ValueError:
    nodes_debug_env = 0
# 3. Set node's logger level
if nodes_debug_env:
    main_logger.setLevel(logging.DEBUG - (nodes_debug_env - 1))
    final_level_str = f"DEBUG (due to {NODES_DEBUG_VAR}={nodes_debug_env})"
else:
    main_logger.setLevel(effective_comfy_level)
    final_level_str = logging.getLevelName(effective_comfy_level) + " (matching ComfyUI global)"
_initial_setup_logger = logging.getLogger(NODES_NAME + ".setup")  # A temporary logger for this message
_initial_setup_logger.debug(f"{NODES_NAME} logger level set to: {final_level_str}")


def logger_set_standalone(args):
    verbose = args.verbose
    main_logger.setLevel(logging.DEBUG - (verbose - 1) if verbose else logging.INFO)
    global standalone_mode
    standalone_mode = True
