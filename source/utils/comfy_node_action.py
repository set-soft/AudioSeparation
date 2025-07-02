# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# ComfyUI Node actions
import logging
# ComfyUI imports
try:
    from server import PromptServer
    with_comfy = True
except Exception:
    with_comfy = False
# Local imports
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.comfy_node_action")


def send_node_action(action: str, arg1: str = None, arg2: str = None, sid: str = None):
    """
    Sends a node action event to the ComfyUI client.

    Args:
        action (str): Action to be performed.
        arg1 (str): First argument
        arg2 (str): Second argument
        sid (str, optional): The session ID of the client to send to.
                            If None, broadcasts to all clients. Defaults to None.
    """
    if not with_comfy:
        return
    try:
        PromptServer.instance.send_sync(
            "set-audioseparation-node",  # This is our custom event name
            {
                'action': action,
                'arg1': arg1,
                'arg2': arg2
            },
            sid
        )
    except Exception as e:
        logger.error(f"when trying to use ComfyUI PromptServer: {e}")
