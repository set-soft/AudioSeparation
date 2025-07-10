import logging
from ..utils.misc import NODES_NAME

rlogger = logging.getLogger(f"{NODES_NAME}.demucs_log")


# A helper to make logging clean and avoid repetition.
def __log_param(kwargs, param_name, default, description="", unit="", indent="  ", extra=False):
    """
    Logs a parameter if its value is different from the default, or if it's a key parameter.

    Args:
        logger: The logging instance to use.
        kwargs (dict): The dictionary of parameters for the model.
        param_name (str): The name of the parameter to check.
        default: The default value for this parameter.
        description (str): A user-friendly description of the parameter.
        unit (str): An optional unit to display after the value (e.g., 'Hz').
        indent (str): The indentation string for the log message.
    """
    value = kwargs.get(param_name, default)
    # We log if the value is not the default, or if it's a fundamental parameter.
    important_params = ['sources', 'segment']
    if not (value != default or param_name in important_params or extra):
        return None
    unit_str = f" {unit}" if unit else ""
    desc_str = description if description else param_name.capitalize().replace('_', ' ')
    return f"     {indent}- {desc_str}: {str(value)}{unit_str} ({param_name})"


def _log_param(logger, kwargs, param_name, default, description="", unit="", indent="  ", extra=False):
    res = __log_param(kwargs, param_name, default, description, unit, indent, extra)
    if res is not None:
        logger(res)


def _add_param(params, kwargs, param_name, default, description="", unit="", indent="  ", extra=False):
    res = __log_param(kwargs, param_name, default, description, unit, indent, extra)
    if res is not None:
        params.append(res)


def _flush_section(logger, params, name):
    if params:
        _section(logger, name)
        for p in params:
            logger(p)


def _flush_sub_section(logger, params, name):
    if params:
        _sub_section(logger, name)
        for p in params:
            logger(p)


def _section(logger, name):
    logger(60*" ")
    logger(f"       {name}:")


def _sub_section(logger, name):
    logger(f"       {name}:")


def _log_type(logger, n, name):
    start = "" if n < 0 else f"{n+1}. "
    logger(f"  {start}Type: {name}")


def _convolutions(kwargs, logger, extra, advanced=False):
    params = []
    _add_param(params, kwargs, 'kernel_size', 8, extra=extra)
    _add_param(params, kwargs, 'stride', 4, extra=extra)
    if advanced:
        _add_param(params, kwargs, 'time_stride', 2, "Final time layer stride", extra=extra)
    _add_param(params, kwargs, 'context', 1, "Decoder context window size", extra=extra)
    if advanced:
        _add_param(params, kwargs, 'context_enc', 0, "Encoder context window size", extra=extra)
    _flush_section(logger, params, "Convolutions")


def _normalization(kwargs, logger, extra):
    params = []
    _add_param(params, kwargs, 'norm_starts', 4, "Start at layer", extra=extra)
    _add_param(params, kwargs, 'norm_groups', 4, "Number of groups", extra=extra)
    _flush_section(logger, params, "Normalization")


def _dconv(kwargs, logger, extra, full=True):
    if kwargs.get('dconv_mode', 1) <= 0:
        return
    params = []
    where = ['', 'In encoder', 'In decoder', 'In encoder and decoder'][kwargs.get('dconv_mode', 1)]
    _add_param(params, kwargs, 'dconv_mode', 1, where, extra=extra)
    _add_param(params, kwargs, 'dconv_depth', 2, "Number of layers in DConv branch", extra=extra)
    if full:
        comp = 4
        init = 1e-4
    else:
        comp = 8
        init = 1e-3
    _add_param(params, kwargs, 'dconv_comp', comp, "Channel compression factor", extra=extra)
    _add_param(params, kwargs, 'dconv_init', init, "Initial scale", extra=extra)
    if full:
        _add_param(params, kwargs, 'dconv_attn', 4, "Layer to start attention in DConv", extra=extra)
        _add_param(params, kwargs, 'dconv_lstm', 4, "Layer to start LSTM in DConv", extra=extra)
    _flush_section(logger, params, "DConv Residual Branch")


def _structure(logger, kwargs, extra, ch=64, depth=6, with_lstm=False, with_ch_tm=False):
    params = []
    _add_param(params, kwargs, 'channels', ch, "Initial hidden channels", extra=extra)
    _add_param(params, kwargs, 'depth', depth, "Number of U-Net layers", extra=extra)
    _add_param(params, kwargs, 'growth', 2.0, "Channel growth factor per layer", extra=extra)
    _add_param(params, kwargs, 'rewrite', True, "Use 1x1 convolutions in blocks", extra=extra)
    if with_lstm:
        _add_param(params, kwargs, 'lstm_layers', 0, "Number of main LSTM layers", extra=False)
    if with_ch_tm:
        _add_param(params, kwargs, 'channels_time', None, "Specific channels for time branch", extra=False)
    _flush_section(logger, params, "Structure")


def _stft(logger, kwargs, extra):
    params = []
    _add_param(params, kwargs, 'nfft', 4096, "Frequency Bins", extra=extra)
    _add_param(params, kwargs, 'cac', True, "Use Complex as Channels", extra=extra)
    if not kwargs.get('cac', True):
        _add_param(params, kwargs, 'wiener_iters', 0, "Wiener filter iterations", extra=extra)
    _flush_section(logger, params, "STFT")


def _freq_branch(kwargs, logger, extra):
    params = []
    def_ratio = None
    if kwargs.get('multi_freqs') == []:
        def_ratio = []
    _add_param(params, kwargs, 'multi_freqs', def_ratio, "Ratios for frequency band splitting", extra=extra)
    if kwargs.get('multi_freqs'):
        _add_param(params, kwargs, 'multi_freqs_depth', 2, "Layers to apply frequency splitting")
    _add_param(params, kwargs, 'freq_emb', 0.2, "Frequency embedding weight", extra=extra)
    if kwargs.get('freq_emb'):
        indent = "    "
        _add_param(params, kwargs, 'emb_scale', 10, "Scale", indent=indent, extra=extra)
        _add_param(params, kwargs, 'emb_smooth', True, "Smooth", indent=indent, extra=extra)
    _flush_section(logger, params, "Frequency Branch")


def _log_demucs_params(n, kwargs, logger, extra):
    """Logs the parameters for the original Demucs class."""

    _log_type(logger, n, "Classic Waveform Demucs (Demucs)")
    _log_param(logger, kwargs, 'sources', [], "Target source names", extra=extra)
    _log_param(logger, kwargs, 'segment', 40, "Segment size", unit="s", extra=extra)

    # --- Structure & Channels ---
    _structure(logger, kwargs, extra, ch=64, depth=6, with_lstm=True)

    # --- Convolutions ---
    _convolutions(kwargs, logger, extra)

    params = []
    _add_param(params, kwargs, 'gelu', True, "GeLU (not ReLU)", extra=extra)
    if kwargs.get('rewrite', True):
        _add_param(params, kwargs, 'glu', True, "GLU in 1x1 rewrite (not ReLU)", extra=extra)
    _flush_section(logger, params, "Activations")

    # --- Normalization ---
    _normalization(kwargs, logger, extra)

    # --- DConv Residual Branch ---
    _dconv(kwargs, logger, extra)

    # --- Pre/Post Processing ---
    params = []
    _log_param(params, kwargs, 'resample', True, "Use 2x resampling", extra=extra)
    _log_param(params, kwargs, 'normalize', True, "Normalize audio on-the-fly", extra=extra)
    _flush_section(logger, params, "Processing")


def _log_hdemucs_params(n, kwargs, logger, extra):
    """Logs the parameters for the HDemucs (Hybrid Spectrogram/Waveform) class."""

    _log_type(logger, n, "Hybrid Demucs (Spectrogram + Waveform) (HDemucs)")
    _log_param(logger, kwargs, 'sources', [], "Target source names", extra=extra)

    # --- Structure & Channels ---
    _structure(logger, kwargs, extra, ch=48, depth=6, with_ch_tm=True)

    # --- STFT & Spectrogram ---
    _stft(logger, kwargs, extra)

    # --- Frequency Branch ---
    _freq_branch(kwargs, logger, extra)

    # --- Convolutions ---
    _convolutions(kwargs, logger, extra, advanced=True)

    # --- Normalization ---
    _normalization(kwargs, logger, extra)

    # --- DConv Residual Branch (defaults are different from Demucs) ---
    _dconv(kwargs, logger, extra)


def _log_htdemucs_params(n, kwargs, logger, extra):
    """Logs the parameters for the HTDemucs (Hybrid Transformer) class."""

    _log_type(logger, n, "Hybrid Transformer Demucs (HTDemucs)")
    _log_param(logger, kwargs, 'sources', [], "Target source names", extra=extra)
    _log_param(logger, kwargs, 'segment', 10, "Segment size", unit="s", extra=extra)

    # --- Structure & Channels (defaults are different from HDemucs) ---
    _structure(logger, kwargs, extra, ch=48, depth=4)

    # --- STFT & Spectrogram ---
    _stft(logger, kwargs, extra)

    # --- Frequency Branch ---
    _freq_branch(kwargs, logger, extra)

    # --- Convolutions ---
    _convolutions(kwargs, logger, extra, advanced=True)

    # --- Normalization ---
    _normalization(kwargs, logger, extra)

    # --- DConv (defaults are different) ---
    _dconv(kwargs, logger, extra, full=False)

    # --- Transformer Block ---
    if kwargs.get('t_layers', 5) > 0:
        _section(logger, "Transformer")
        indent = "    "
        if kwargs.get('bottom_channels', 0):
            _log_param(logger, kwargs, 'bottom_channels', 0, "Channels forced to", extra=extra)
        _log_param(logger, kwargs, 't_hidden_scale', 4.0, "Hidden scale", extra=extra)
        _log_param(logger, kwargs, 't_layers', 5, "Number of transformer layers", extra=extra)
        _log_param(logger, kwargs, 't_heads', 8, "Number of attention heads", extra=extra)
        _log_param(logger, kwargs, 't_dropout', 0.0, "Dropout", extra=extra)

        params = []
        _add_param(params, kwargs, 't_emb', 'sin', "Type", extra=extra, indent=indent)
        _add_param(params, kwargs, 't_weight_pos_embed', 1.0, "Weight", extra=False, indent=indent)
        t_emb = kwargs.get('t_emb', 'sin')
        if t_emb == 'scaled':
            _add_param(params, kwargs, 't_max_positions', 10000, "Max positions", extra=extra, indent=indent)
        elif t_emb == 'sin':
            _add_param(params, kwargs, 't_max_period', 10000.0, "Max period", extra=extra, indent=indent)
            _add_param(params, kwargs, 't_sin_random_shift', 0, "Random shift", extra=False, indent=indent)
        elif t_emb == 'cape':
            _add_param(params, kwargs, 't_cape_mean_normalize', True, "Cape normalize", extra=extra, indent=indent)
            _add_param(params, kwargs, 't_cape_glob_loc_scale', [5000.0, 1.0, 1.4], "Cape params", extra=extra, indent=indent)
            if kwargs.get('t_cape_augment', True):
                rlogger.warning("t_cape_augment should be False")
        _flush_sub_section(logger, params, "Positional Embeddings")

        params = []
        _add_param(params, kwargs, 't_norm_first', True, "Before attention/FFN", extra=extra, indent=indent)
        _add_param(params, kwargs, 't_norm_in', True, "Before pos. embedding", extra=extra, indent=indent)
        if kwargs.get('t_norm_in', True):
            _add_param(params, kwargs, 't_norm_in_group', False, "On all timesteps", extra=extra, indent=indent)
        _add_param(params, kwargs, 't_group_norm', False, "Of encoder on all timesteps", extra=extra, indent=indent)
        _add_param(params, kwargs, 't_norm_out', True, "GroupNorm at end of layers", extra=extra, indent=indent)

        _add_param(params, kwargs, 't_cross_first', False, "Cross-attention is the first layer", extra=extra)
        _add_param(params, kwargs, 't_layer_scale', True, "Layer scale", extra=extra)
        _add_param(params, kwargs, 't_gelu', True, "GeLU (not ReLU)", extra=extra)
        _flush_sub_section(logger, params, "Normalization")

        # Log sparsity details only if sparse attention is enabled
        is_sparse = kwargs.get('t_sparse_self_attn', False)
        _log_param(logger, kwargs, 't_sparse_self_attn', False, "Use sparse self-attention", extra=extra)
        if is_sparse:
            _log_param(logger, kwargs, 't_sparse_cross_attn', False, "Sparse cross-attention", indent=indent, extra=extra)
            _log_param(logger, kwargs, 't_auto_sparsity', False, "Automatic sparsity", indent=indent, extra=extra)
            auto_sparsity = kwargs.get('t_auto_sparsity', False)
            if not auto_sparsity:
                _log_param(logger, kwargs, 't_mask_type', 'diag', "Masking pattern", indent=indent, extra=extra)
                _log_param(logger, kwargs, 't_mask_random_seed', 42, "Mask seed", indent=indent, extra=extra)
                mask_t = kwargs.get('t_mask_type', 'diag')
                if 'diag' in mask_t:
                    _log_param(logger, kwargs, 't_sparse_attn_window', 500, "Window size", indent=indent, extra=extra)
                if 'gloabal' in mask_t:
                    _log_param(logger, kwargs, 't_global_window', 100, "Window size", indent=indent, extra=extra)
                if 'random' in mask_t:
                    _log_param(logger, kwargs, 't_sparsity', 0.95, "Sparsity for random mask", indent=indent, unit="%",
                               extra=extra)

        # Training only
        # _log_param(logger, kwargs, 't_weight_decay', 0.0, "Weight decay", extra=False)
        # _log_param(logger, kwargs, 't_lr', None, "Learning rate", extra=False)
        # _log_param(logger, kwargs, 't_cape_augment', True, "Learning rate", extra=False)
        # _log_param(logger, kwargs, 'rescale', 0.1, "Rescale trick", extra=False)


# --- Main Dispatcher Function ---
def log_demucs_model_info(n, klass_name: str, kwargs: dict, logger, extra=False):
    """
    Logs a human-friendly summary of a Demucs model's configuration.

    Args:
        klass_name (str): The name of the model class (e.g., "HTDemucs").
        kwargs (dict): The keyword arguments used to initialize the model.
        logger: The logger instance to use for output.
    """
    # Always log these fundamental parameters
    sr = kwargs.get('samplerate', 44100)
    if sr != 44100:
        rlogger.warning("Model not configured for 44.1 kHz sample rate")
    a_ch = kwargs.get('audio_channels', 2)
    if a_ch != 2:
        rlogger.warning("Model not configured for stereo")

    if klass_name == "HTDemucs":
        _log_htdemucs_params(n, kwargs, logger, extra)
    elif klass_name == "HDemucs":
        _log_hdemucs_params(n, kwargs, logger, extra)
    elif klass_name == "Demucs":
        _log_demucs_params(n, kwargs, logger, extra)
    else:
        logger.warning(f"No specific logger for model class: {klass_name}. "
                       "Displaying raw kwargs.")
        for key, value in kwargs.items():
            logger(f"     - {key}: {value}")
