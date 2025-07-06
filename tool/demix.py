# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
#
# Tool to demix audio
# Run it using: python tool/demix.py -m HASH AUDIO
import argparse
import os
import sys
import torch
# Local imports
import bootstrap  # noqa: F401
from source.db.models_db import ModelsDB, cli_add_models_and_db
from source.inference.demixer import get_demixer
from source.utils.logger import main_logger, logger_set_standalone
from source.utils.load_audio import load_audio
from source.utils.save_audio import save_audio
from source.utils.misc import cli_add_verbose

BANNER = "🎵 MDX-Net Audio Separation Tool 🎵"


# --- Main Demixing Logic ---
def demix(d, args):
    main_logger.info("🚀 Starting audio separation process...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main_logger.info(f"💻 Using device: {device}")

    # --- Load and Prepare Audio ---
    waveform, sr = load_audio(args.input_file, force_sr=44100, force_stereo=True)

    # --- Load Model ---
    demixer = get_demixer(d, device, args.models_dir)

    # --- Do inference in chunks ---
    wavs = demixer(waveform, args.segments)

    # --- Save outputs ---
    base, ext = os.path.splitext(args.input_file)
    out_ext = ext if not args.format else '.' + args.format.lower()
    out_format = args.format or out_ext[1:]
    if args.save_main:
        out_path = f"{args.out_base or base}_{wavs[0]['stem']}{out_ext}"
        save_audio(wavs[0]['waveform'], wavs[0]['sample_rate'], out_path, out_format)

    if args.save_complement:
        for wav in wavs[1:]:
            if wav is None:
                continue
            out_path = f"{args.out_base or base}_{wav['stem']}{out_ext}"
            save_audio(wav['waveform'], wav['sample_rate'], out_path, out_format)

    main_logger.info("🎉 All operations complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=BANNER,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input options ---
    parser.add_argument('input_file', type=str, nargs='?', default=None,
                        help="Path to the input audio file (wav, mp3, flac). Required unless --list is used.")
    parser.add_argument('-m', '--model', type=str, default="499a6a6bf9da6d330235a1576007ddc0",
                        help="Hash or name for known model (use --list to know all). "
                        "Or path to the model file (.safetensors or .onnx), which must be known.")
    cli_add_models_and_db(parser)

    # --- Output options ---
    parser.add_argument('--no_main', dest='save_main', action='store_false',
                        help="Do not save the main separated stem.")
    parser.add_argument('--save_complement', action='store_true',
                        help="Save all the stems, including the complement stem (input - main).")
    parser.add_argument('--out_base', type=str, default=None,
                        help="Base for the output path. No extension here, we will add the name of the stem and extension")
    parser.add_argument('--format', type=str, default=None, choices=['wav', 'flac', 'mp3'],
                        help="Output audio format. Defaults to input format.")

    # --- Control Arguments ---
    parser.add_argument('--segments', type=int, default=1,
                        help="How many audio segments to process at once")
    parser.add_argument('-l', '--list', action='store_true', help="Show available models")
    cli_add_verbose(parser)

    parser.set_defaults(save_main=True)
    args = parser.parse_args()
    logger_set_standalone(args)
    main_logger.info(BANNER)

    # Sanity check
    if not args.list:
        if not args.input_file:
            main_logger.error("💥 The following arguments are required: input_file")
            sys.exit(2)
        if not args.save_main and not args.save_complement:
            main_logger.error("💥 Nothing to save! Please don't specify --no_main (default) or use --save_complement.")
            sys.exit(2)

    # Check what we have
    db = ModelsDB(args.models_dir, args.json_file)
    models = db.get_filtered()

    # Just list models
    if args.list:
        main_logger.info("\n--- Available models ---\n")
        known = models.get_display_names(clean=True)
        for m in known:
            main_logger.info(f"`{m}` {models.get_by_display_name(m)['hash']}")
        main_logger.info(f"\n{len(known)} models known.")
        sys.exit(0)

    # Check we have a sound file
    if not os.path.isfile(args.input_file):
        main_logger.error(f"💥 `{args.input_file}` is not a file.")
        sys.exit(4)

    # Look for the selected model
    d = models.get(args.model)
    if d is None:
        main_logger.error(f"💥 Unknown model `{args.model}`. If you provided a file check it exists")
        sys.exit(3)
    try:
        main_logger.info(f"📂 Using model from `{d['model_path']}`")
    except KeyError:
        # Needs download
        pass

    demix(d, args)
