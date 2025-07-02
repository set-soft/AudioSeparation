# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# ONNX model loader helper
# Original code from Gemini 2.5 Pro
import logging
try:
    import onnxruntime as ort
    with_onnx = True
except Exception:
    with_onnx = False
from .misc import NODES_NAME

logger = logging.getLogger(f"{NODES_NAME}.load_onnx")

if with_onnx:
    import torch

    class ONNXWrapper:
        """
        A wrapper class for an ONNX Runtime InferenceSession to provide a
        PyTorch-like __call__ interface.
        """
        def __init__(self, session: ort.InferenceSession, device: torch.device):
            self.session = session
            self.device = device
            # Get the name of the input tensor from the model's graph
            self.input_name = self.session.get_inputs()[0].name

        def __call__(self, input_tensor: torch.Tensor):
            """
            Performs inference using the ONNX session.

            Args:
                input_tensor: A PyTorch tensor already on the correct device.

            Returns:
                A PyTorch tensor on the same device as the input.
            """
            # 1. Convert the input PyTorch tensor to a CPU NumPy array
            input_numpy = input_tensor.cpu().numpy()
            # 2. Run the ONNX session
            result_numpy = self.session.run(None, {self.input_name: input_numpy})[0]
            # 3. Convert the output NumPy array back to a PyTorch tensor on the original device
            result_tensor = torch.from_numpy(result_numpy).to(self.device)

            return result_tensor

    def load_onnx(model_path, device):
        logger.info("Loading ONNX model for runtime inference...")
        providers = ['CUDAExecutionProvider' if 'cuda' in str(device) else 'CPUExecutionProvider']
        session = ort.InferenceSession(model_path, providers=providers)
        model_w = ONNXWrapper(session, device)
        return model_w
else:
    def load_onnx(model_path, device):
        raise ValueError("No ONNX support, please install `onnxruntime`")
