#!/usr/bin/env python3
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to show PyTorch class i.e:
# python tool/show_class.py -m source/inference/MDX_Net.py:MDX_Net
import argparse
import sys
from torch import nn
# Local imports
import bootstrap  # noqa: F401
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.load_class import import_model_class
from source.utils.misc import cli_add_verbose


def load_class(args):
    """
    Loads the structure of a specified PyTorch model class
    """
    main_logger.info("--- Loading the class ---\n")
    main_logger.info(f"Model Class: {args.model_location}")
    main_logger.info(f"Parameters: dim_f={args.dim_f}, ch={args.ch}, num_stages={args.num_stages}")

    # 1. Dynamically import the specified model class
    TargetModelClass = import_model_class(args.model_location)

    # 2. Instantiate the model with the provided parameters
    try:
        pytorch_model = TargetModelClass(dim_f=args.dim_f, ch=args.ch, num_stages=args.num_stages)
    except Exception as e:
        main_logger.error("Could not instantiate the model class with the given parameters.")
        main_logger.error(f"Please check if the class '{args.model_location.split(':')[1]}' "
                          "accepts 'dim_f', 'ch', and 'num_stages'.")
        main_logger.error(f"Original error: {e}")
        sys.exit(1)

    return pytorch_model


def show(model):
    """
    Prints the structure of a specified PyTorch model class
    """
    # Print the PyTorch model structure to the log
    # The f-string ensures the multi-line output of the model is captured in the log
    main_logger.info(f"\n--- PyTorch Model Structure ---\n\n{model}")


def export_onnx(model, dim_f, file):
    import torch

    main_logger.info("\n--- Exporting to ONNX ---")

    # 1. Create a dummy input with the correct shape
    dummy_input = torch.randn(1, 4, dim_f, 256)
    model.eval()

    # 2. Export the model
    try:
        torch.onnx.export(
            model,
            dummy_input,
            file,
            export_params=False,  # We only care about the graph structure, not weights
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        main_logger.info(f"\n✅ SUCCESS! Model successfully exported to '{file}'")

    except Exception as e:
        main_logger.error(f"\n❌ FAILURE during export: {e}")


def show_keys(model):
    main_logger.info("\n--- PyTorch Model Keys ---\n\n")
    for key in model.state_dict().keys():
        main_logger.info(key)


def show_compact(module, parent_name='model', indent=0, index=0):
    """
    Recursively walks the PyTorch model and prints its structure
    in a format similar to our ONNX analysis.
    """
    # Get the module's class name and a formatted name with its parent
    module_name = module.__class__.__name__
    full_name = f"{parent_name}.{module_name}" if parent_name else module_name

    # Print containers like ModuleList or Sequential differently
    if len(list(module.children())) > 0 and not isinstance(module, nn.Sequential):
        main_logger.info(f"[---] {'  ' * indent}{full_name}: {module_name}")

    # Iterate through named children to maintain order
    for name, child_module in module.named_children():
        child_full_name = f"{parent_name}.{name}"

        # If the child is a leaf node (like Conv2d, Linear, etc.)
        if len(list(child_module.children())) == 0:
            main_logger.info(f"[{index:03d}] {'  ' * (indent+1)}{child_full_name}: {child_module.__class__.__name__}")
            index += 1
        else:
            # If the child is another container, recurse
            index = show_compact(child_module, child_full_name, indent + 1, index)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shows the current class layers and/or keys",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model_location', type=str, default='source/inference/MDX_Net.py:MDX_Net',
                        help="Location of the PyTorch model class, in 'filename:ClassName' format.\n"
                        "(default: source/inference/MDX_Net.py:MDX_Net)")
    parser.add_argument('-d', '--dim_f',  type=int,  default=3072,
                        help="The frequency dimension (dim_f) of the model. (default: 3072)")
    parser.add_argument('-c', '--ch',  type=int,  default=48,
                        help="The base channel count (ch) of the model. (default: 48)")
    parser.add_argument('-n', '--num_stages', type=int, default=5,
                        choices=[2, 3, 4, 5, 6, 7],  # Restrict to known valid values
                        help="The number of U-Net stages in the model. (choices: 2 to 7, default: 5)")
    cli_add_verbose(parser)
    parser.add_argument('-o', '--export_onnx', type=str, default=None,
                        help="Path for the optional output .onnx file.\n"
                        "Only the structure is exported")
    parser.add_argument('-k', '--keys', action='store_true', help="Print the keys for the state_dict.")
    parser.add_argument('-C', '--compact', action='store_true', help="Print a compact representation.")
    parser.add_argument('-S', '--no_show', action='store_false', help="Don't print the structure.")

    args = parser.parse_args()
    logger_set_standalone(args)
    model = load_class(args)
    if args.no_show:
        show(model)
    # Optional ONNX export
    if args.export_onnx:
        export_onnx(model, args.dim_f, args.export_onnx)
    # Optional print keys
    if args.keys:
        show_keys(model)
    # Optional compact form
    if args.compact:
        main_logger.info("\n--- PyTorch Compact Model Structure ---\n")
        show_compact(model)
