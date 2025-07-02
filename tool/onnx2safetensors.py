# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to convert an ONNX MDX-Net file into a safetensors file
# Run it using: python tool/onnx2safetensors.py ONNX_FILE
#
# Created using Gemini 2.5 Pro, but with a lot of iterations and adjusts
import argparse
import json
import numpy as np
import onnx
import os
import sys
import torch
from torch import nn
# Local imports
import bootstrap  # noqa: F401
from safetensors.torch import save_file
from source.utils.load_class import import_model_class
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.misc import cli_add_verbose
from source.db.models_db import get_download_url


class OnnxGraph:
    """A helper class to hold ONNX graph information."""
    def __init__(self, onnx_path: str):
        main_logger.info("Loading ONNX graph...")
        onnx_model = onnx.load(onnx_path)
        self.nodes = list(onnx_model.graph.node)
        self.weights = {t.name: torch.from_numpy(np.copy(onnx.numpy_helper.to_array(t))) for t in onnx_model.graph.initializer}
        self.weighted_nodes = [n for n in self.nodes if any(inp in self.weights for inp in n.input)]
        self.num_weighted_nodes = len(self.weighted_nodes)
        main_logger.info(f"ONNX graph loaded. Found {self.num_weighted_nodes} nodes with weights.")


class AutoMapper:
    """
    Performs the 'Dual Walk' using a reliable recursive traversal of the
    PyTorch model's module hierarchy, eliminating the need for a forward-pass trace.
    """
    def __init__(self, pytorch_model: nn.Module, onnx_graph: OnnxGraph, verbose: bool = False):
        self.pytorch_model = pytorch_model
        self.onnx_graph = onnx_graph
        self.verbose = verbose

        # Get an ordered list of PyTorch modules that have weights by traversing the module hierarchy.
        # This is guaranteed to be in the same order as the state_dict.
        self.pytorch_modules_with_params = [
            m for m in self.pytorch_model.modules()
            if len(list(m.children())) == 0 and len(list(m.parameters(recurse=False))) > 0
        ]

    def add_target(self, map_list, key_iter, source, transpose=False):
        target = next(key_iter)
        map_list.append({'target': target, 'source': source, 'transpose': transpose})
        if self.verbose > 1:
            main_logger.debug(f"    {source} -> {target}'")

    def generate_map(self):
        main_logger.info("Generating automatic weight map by walking module hierarchy...")
        if len(self.pytorch_modules_with_params) != self.onnx_graph.num_weighted_nodes:
            main_logger.error("🚨 Mismatch in number of weighted layers!")
            main_logger.error(f"PyTorch model defines {len(self.pytorch_modules_with_params)} layers with weights.")
            main_logger.error(f"ONNX file contains {self.onnx_graph.num_weighted_nodes} nodes with weights.")
            main_logger.error("This likely means the model's parameters (dim_f, ch, num_stages) do not match the ONNX file.")
            sys.exit(1)

        map_list = []
        # We can now reliably iterate through the state_dict keys.
        key_iter = iter([k for k in self.pytorch_model.state_dict().keys() if 'num_batches_tracked' not in k])

        for pt_module, onnx_node in zip(self.pytorch_modules_with_params, self.onnx_graph.weighted_nodes):
            if self.verbose:
                main_logger.debug(f"  Mapping ONNX '{onnx_node.op_type}' -> PyTorch '{pt_module.__class__.__name__}'")

            onnx_param_names = [name for name in onnx_node.input if name in self.onnx_graph.weights]

            if isinstance(pt_module, nn.Linear):
                self.add_target(map_list, key_iter, onnx_param_names[0], True)
            elif isinstance(pt_module, (nn.Conv2d, nn.ConvTranspose2d)):
                self.add_target(map_list, key_iter, onnx_param_names[0])
                if pt_module.bias is not None:
                    self.add_target(map_list, key_iter, onnx_param_names[1])
            elif isinstance(pt_module, nn.BatchNorm2d):
                # The order in the state_dict is weight, bias, running_mean, running_var
                # The order in the ONNX node is scale, B, mean, var
                # They correspond 1-to-1
                for i in range(4):
                    self.add_target(map_list, key_iter, onnx_param_names[i])

        main_logger.info("Automatic map generated successfully.")
        return map_list


def verify_parameter(params, name, value, description):
    """ Check if a parameter is already known.
        If known check our detection is ok.
        Otherwise add it """
    in_params = params.get(name)
    if in_params:
        if in_params != value:
            main_logger.error(f"{description} mismatch ({in_params} vs {value})")
            sys.exit(3)
    else:
        params[name] = value


def convert(args):
    """Main conversion function driven by command line arguments."""
    main_logger.info(f"Starting conversion for '{args.input_file}'...")

    onnx_graph = OnnxGraph(args.input_file)

    # Auto-detect parameters
    first_matmul_weight_name = next((n.input[1] for n in onnx_graph.nodes if n.op_type == 'MatMul'), None)
    if not first_matmul_weight_name:
        raise ValueError("Could not find a MatMul node to detect dim_f.")
    dim_f = onnx_graph.weights[first_matmul_weight_name].shape[0]

    first_conv_weight_name = next((n.input[1] for n in onnx_graph.nodes if n.op_type == 'Conv'), None)
    if not first_conv_weight_name:
        raise ValueError("Could not find a Conv node to detect channels.")
    ch = onnx_graph.weights[first_conv_weight_name].shape[0]

    num_stages = 5 if onnx_graph.num_weighted_nodes > 80 else 4

    total_weight_params = 0
    for tensor in onnx_graph.weights.values():
        total_weight_params += tensor.numel()  # numel() gives the total number of elements

    main_logger.info("\n--- Detected Model Parameters ---")
    main_logger.info(f"  Frequency Dimension (dim_f): {dim_f}")
    main_logger.info(f"  Base Channels (ch): {ch}")
    main_logger.info(f"  U-Net Stages: {num_stages}")
    main_logger.info(f"  Total parameters: {total_weight_params}")

    TargetModelClass = import_model_class(args.model_location)
    target_model = TargetModelClass(dim_f=dim_f, ch=ch, num_stages=num_stages)

    auto_mapper = AutoMapper(target_model, onnx_graph, args.verbose)
    mapping = auto_mapper.generate_map()

    main_logger.info("\n--- Starting Final Weight Transfer ---")
    new_state_dict = target_model.state_dict()
    transfer_count = 0

    for item in mapping:
        target_key, source_key, needs_transpose = item['target'], item['source'], item['transpose']

        if target_key in new_state_dict:
            if source_key in onnx_graph.weights:
                source_tensor = onnx_graph.weights[source_key]
                if needs_transpose:
                    source_tensor = source_tensor.t()
                if new_state_dict[target_key].shape == source_tensor.shape:
                    new_state_dict[target_key].data.copy_(source_tensor)
                    transfer_count += 1
                else:
                    main_logger.warning(f"  [!] SHAPE MISMATCH for '{target_key}': Target {new_state_dict[target_key].shape} "
                                        f"vs Source {source_tensor.shape}")
            else:
                main_logger.error(f"  [!] ERROR: Raw weight key '{source_key}' not found.")
        else:
            main_logger.warning(f"  [!] Warning: Target key '{target_key}' not found.")

    # Finalize and Save
    state_dict_to_save = target_model.state_dict()
    final_state_dict = {
        key: tensor for key, tensor in state_dict_to_save.items()
        if 'num_batches_tracked' not in key
    }

    total_params = len(new_state_dict)
    num_non_param_buffers = total_params - len(final_state_dict)
    loadable_params = len(final_state_dict)

    main_logger.info(f"\nSuccessfully transferred {transfer_count} / {loadable_params} tensors.")
    if transfer_count < loadable_params:
        main_logger.warning("Some weights were not transferred. Check for errors above.")
    else:
        main_logger.info("  All loadable parameters and buffers were successfully transferred.")
    main_logger.info(f"  ({num_non_param_buffers} non-persistent buffers like 'num_batches_tracked' "
                     "will be excluded from the final file.)")

    output_path = args.output_file or os.path.splitext(args.input_file)[0] + ".safetensors"

    # Add some metadata to the file
    hyperparameters = args.metadata if args.metadata is not None else {}
    # Verify dim_f
    verify_parameter(hyperparameters, "mdx_dim_f_set", dim_f, "Frequency Dimension (dim_f)")
    # Verify channels
    verify_parameter(hyperparameters, "channels", ch, "Base Channels (ch)")
    # Verify stages
    verify_parameter(hyperparameters, "stages", num_stages, "U-Net Stages")
    # Verify total parameters
    verify_parameter(hyperparameters, "params", total_weight_params, "Total parameters")
    # Fix entries to match the conversion
    hyperparameters["name"] = os.path.basename(output_path)
    hyperparameters["download"] = "Main/MDX"
    hyperparameters["download"] = get_download_url(hyperparameters)  # Convert to something usable outside our project
    hyperparameters["project"] = "https://github.com/set-soft/AudioSeparation"
    hyperparameters["file_t"] = "safetensors"
    # They must be strings
    model_hyperparameters = {k: str(v) for k, v in hyperparameters.items()}

    main_logger.info(f"\nSaving model weights to '{output_path}'...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the filtered state_dict
    save_file(tensors=final_state_dict, filename=output_path, metadata=model_hyperparameters)
    main_logger.info("\n✅ Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart tool to convert specific MDX-Net ONNX models to PyTorch Safetensors.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the input ONNX model file.")
    parser.add_argument('-o', '--output_file', type=str, default=None,
                        help="Path for the output .safetensors file.\n(default: same as input with .safetensors extension)")
    parser.add_argument('-m', '--model_location', type=str, default='source/inference/MDX_Net.py:MDX_Net',
                        help="Location of the PyTorch model class, in 'filename:ClassName' format.\n"
                        "(default: source/inference/MDX_Net.py:MDX_Net)")
    parser.add_argument('-j', '--metadata', type=str, help="Metadata to include in the hyperparameters, JSON format")
    cli_add_verbose(parser)

    args = parser.parse_args()
    logger_set_standalone(args)
    if args.metadata is not None:
        try:
            args.metadata = json.loads(args.metadata)
        except Exception as e:
            main_logger.error(f"Failed to parse the metadata: {e}")
            sys.exit(2)
        if not isinstance(args.metadata, dict):
            main_logger.error(f"Metadata must be a dict, not {type(args.metadata)}")
            sys.exit(2)
    convert(args)
