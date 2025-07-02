# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Helper to get a model from the correct class
import logging
from .MDX_Net import MDX_Net
from ..utils.misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.get_model")


# Currently we have just one type of networks, but this is a clean way to support more, or even test replacements
def get_model(d):
    model_t = d['model_t'].lower()
    if model_t != "mdx":
        msg = f"Unknown model type `{model_t}`"
        logger.error(msg)
        raise ValueError(msg)
    return MDX_Net(dim_f=d['mdx_dim_f_set'], ch=d['channels'], num_stages=d['stages'])
