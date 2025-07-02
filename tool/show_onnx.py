# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to show the ONNX structure, i.e:
# python tool/show_onnx.py model.onnx
import argparse
import numpy as np
import onnx
from onnx import shape_inference  # Import the shape inference module
import torch
import sys
# Local imports
import bootstrap  # noqa: F401
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.misc import cli_add_verbose


def print_onnx_nodes_and_weights(onnx_model_path):
    """
    Analyzes and prints a detailed map of an ONNX model, including
    type and shape information for all tensors, populated via shape inference.
    """
    original_model = onnx.load(onnx_model_path)

    # --- 1. Run Shape Inference to populate value_info ---
    # This is the crucial step that adds missing type/shape info.
    main_logger.info("Running ONNX shape inference...")
    inferred_model = shape_inference.infer_shapes(original_model)
    graph = inferred_model.graph
    main_logger.info("Shape inference complete.")

    # --- 2. Create a comprehensive map of all tensors in the graph ---
    value_info_all = {
        value_info.name: value_info
        for value_info in list(graph.input) + list(graph.value_info) + list(graph.output)
    }

    main_logger.info("\n--- Model Inputs ---")
    for input_tensor in graph.input:
        tensor_type = onnx.helper.printable_type(input_tensor.type)
        main_logger.info(f"Name: {input_tensor.name}, Type: {tensor_type}")

    main_logger.info("\n--- Model Outputs ---")
    for output_tensor in graph.output:
        tensor_type = onnx.helper.printable_type(output_tensor.type)
        main_logger.info(f"Name: {output_tensor.name}, Type: {tensor_type}")

    # --- 3. Prepare data for analysis ---
    nodes = list(graph.node)
    weights = {t.name: torch.from_numpy(np.copy(onnx.numpy_helper.to_array(t))) for t in graph.initializer}
    node_outputs = {out: n.name for n in nodes for out in n.output}

    main_logger.info(f"\n--- Map for the ONNX graph with {len(nodes)} nodes ---")
    for ni, n in enumerate(nodes):
        has_weight = " (W)" if any(inp in weights for inp in n.input) else ""
        main_logger.info(f"[{ni:03d}] {n.name} [{n.op_type}]{has_weight}")

        # Print detailed inputs with type info
        main_logger.info("    Inputs:")
        for ninp, inp_name in enumerate(n.input):
            from_txt, type_txt = "", ""
            if inp_name in weights:
                from_txt = f"weight initializer [name: {inp_name}]"
                type_txt = str(weights[inp_name].shape)
            elif inp_name in node_outputs:
                from_txt = f"output of node '{node_outputs[inp_name]}'"
            elif inp_name in [i.name for i in graph.input]:
                from_txt = "graph input"
            else:
                from_txt = "unknown source"

            if inp_name in value_info_all:
                type_txt = onnx.helper.printable_type(value_info_all[inp_name].type)
            main_logger.info(f"      - [{ninp}] '{inp_name}' (from {from_txt}) -> Type: {type_txt}")

        # Print detailed outputs with type info
        main_logger.info("    Outputs:")
        for nout, out_name in enumerate(n.output):
            type_txt = (onnx.helper.printable_type(value_info_all.get(out_name, "").type)
                        if out_name in value_info_all else "N/A")
            main_logger.info(f"      - [{nout}] '{out_name}' -> Type: {type_txt}")

    final_output_name = graph.output[0].name
    main_logger.info(f"\nThe final graph output '{final_output_name}' is from node "
                     f"'{node_outputs.get(final_output_name, 'N/A')}'\n")

    # --- Create a dummy input tensor by directly reading the graph's structured data ---
    primary_input_info = graph.input[0]

    # 1. Get the shape
    shape = []
    # The shape is stored in the 'dim' attribute of the tensor type
    for dimension in primary_input_info.type.tensor_type.shape.dim:
        # Check if the dimension has a fixed integer value or is a dynamic parameter
        if dimension.HasField('dim_value'):
            shape.append(dimension.dim_value)
        else:
            # For dynamic dimensions (like 'batch_size'), use 1 as a placeholder
            shape.append(1)

    # 2. Get the data type
    try:
        # Get the integer enum for the element type (e.g., 1 for FLOAT)
        elem_type_enum = primary_input_info.type.tensor_type.elem_type
        # Use ONNX's official mapping to get the corresponding NumPy dtype
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(elem_type_enum)
        # Convert the NumPy dtype to a PyTorch dtype
        torch_dtype = torch.from_numpy(np.array(0, dtype=np_dtype)).dtype
    except (KeyError, AttributeError):
        # Fallback to float32 if type is not specified or recognized
        main_logger.warning("Could not determine input dtype from ONNX graph. Defaulting to float32.")
        torch_dtype = torch.float32

    if not shape:
        main_logger.error("Could not determine input shape from ONNX graph.")
        return None

    main_logger.info(f"Generating a random tensor of shape {shape} and type {torch_dtype} for inference.")

    # 3. Create the random tensor
    torch.manual_seed(0)
    dummy_input_tensor = torch.randn(shape, dtype=torch_dtype)

    # --- 4. Calculate and Print Model Statistics ---
    num_weighted_nodes = len([n for n in nodes if any(inp in weights for inp in n.input)])
    num_weight_tensors = len(weights)

    # Calculate the total number of parameters (like "13B")
    total_params = 0
    for tensor in weights.values():
        total_params += tensor.numel()  # numel() gives the total number of elements

    # Format the total parameters into a human-readable string (e.g., 1.2M, 2.5B)
    if total_params > 1_000_000_000:
        params_str = f"{total_params / 1_000_000_000:.2f}B"
    elif total_params > 1_000_000:
        params_str = f"{total_params / 1_000_000:.2f}M"
    elif total_params > 1_000:
        params_str = f"{total_params / 1_000:.2f}K"
    else:
        params_str = f"{total_params}"

    main_logger.info("\n--- Model Statistics ---\n")
    main_logger.info(f"Number of layers with weights: {num_weighted_nodes}")
    main_logger.info(f"Total number of weight/bias/parameter tensors: {num_weight_tensors}")
    main_logger.info(f"Total model parameters: {total_params:,} (~{params_str})")

    return dummy_input_tensor


def convert_model(file):
    try:
        from onnx2pytorch import ConvertModel
    except Exception:
        main_logger.error("Install onnx2pytorch")
        sys.exit(3)

    main_logger.info("--- Loading the ONNX ---\n")
    # Load the ONNX model
    onnx_model = onnx.load(file)

    main_logger.info("--- Converting to PyTorch ---\n")
    # Convert the ONNX model to a PyTorch model
    pytorch_model = ConvertModel(onnx_model)

    return pytorch_model


def show_converted(model):
    main_logger.info(f"\n--- PyTorch Model Structure ---\n\n{model}")


def show_keys(model):
    main_logger.info("\n--- PyTorch Model Keys ---\n\n")
    for key in model.state_dict().keys():
        main_logger.info(key)


def run_model(model, input):
    main_logger.info("\n--- Running Forward Pass ---")
    if input is None:
        main_logger.error("Missing random input for inference run")
        return

    # Run the forward pass to trigger our interception prints
    with torch.no_grad():
        _ = model(input)

    main_logger.info("\n--- Run Complete ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shows the ONNX layers",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input_file', type=str, help="Path to the input ONNX model file.")
    parser.add_argument('-c', '--show_converted', action='store_true', help="Show the class converted using onnx2pytorch.")
    parser.add_argument('-k', '--keys', action='store_true', help="Print the keys for the converted state_dict.")
    parser.add_argument('-r', '--run', action='store_true',
                        help="Run un inference using the converted model.\n"
                        "Incompatible with -S")
    parser.add_argument('-S', '--no_show', action='store_false', help="Don't print the ONNX structure.")
    cli_add_verbose(parser)

    args = parser.parse_args()
    logger_set_standalone(args)
    if args.run and not args.no_show:
        main_logger.error("-r can't be used when -S is specified")
        sys.exit(1)
    dummy_input_tensor = print_onnx_nodes_and_weights(args.input_file) if args.no_show else None
    if args.show_converted or args.keys or args.run:
        model = convert_model(args.input_file)
        if args.show_converted:
            show_converted(model)
        if args.keys:
            show_keys(model)
        if args.run:
            run_model(model, dummy_input_tensor)
