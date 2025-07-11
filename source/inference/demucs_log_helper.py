from fractions import Fraction
import logging
from ..utils.misc import NODES_NAME

rlogger = logging.getLogger(f"{NODES_NAME}.demucs_log")


class DemucsModelInfo(object):
    def __init__(self, index, klass_name: str, kwargs: dict, logger, weights, extra=False):
        super().__init__()
        self.index = index
        self.kwargs = kwargs
        self.logger = logger
        self.extra = extra
        self.extra_indent = ""
        self.weights = weights

        # Always log these fundamental parameters
        sr = kwargs.get('samplerate', 44100)
        if sr != 44100:
            rlogger.warning("Model not configured for 44.1 kHz sample rate")
        a_ch = kwargs.get('audio_channels', 2)
        if a_ch != 2:
            rlogger.warning("Model not configured for stereo")

        if klass_name == "HTDemucs":
            self.htdemucs()
        elif klass_name == "HDemucs":
            self.hdemucs()
        elif klass_name == "Demucs":
            self.demucs()
        else:
            logger.warning(f"No specific logger for model class: {klass_name}. "
                           "Displaying raw kwargs.")
            for key, value in kwargs.items():
                logger(f"     - {key}: {value}")

    def get(self, key, default=None):
        return self.kwargs.get(key, default)

    def log_type(self, name):
        start = "" if self.index < 0 else f"{self.index+1}. "
        self.logger(f"  {start}Type: {name}")

    def _log_param(self, param_name, default, description="", unit="", indent="  ", can_skip=False):
        """
        Logs a parameter if its value is different from the default, or if it's a key parameter.

        Args:
            param_name (str): The name of the parameter to check.
            default: The default value for this parameter.
            description (str): A user-friendly description of the parameter.
            unit (str): An optional unit to display after the value (e.g., 'Hz').
            indent (str): The indentation string for the log message.
        """
        value = self.get(param_name, default)

        # We log if the value is not the default, or if it's a fundamental parameter.
        is_default = (value == default)
        is_important = param_name in ['sources', 'segment']

        if is_default and not is_important and not self.extra:
            return None
        if can_skip and is_default:
            return None

        if param_name == 'sources' and self.weights:
            value = [s if w == 1.0 else ('' if not w else f'{w}*{s}') for s, w in zip(value, self.weights)]

        if unit == '%':
            value *= 100
        unit_str = f" {unit}" if unit else ""
        desc_str = description if description else param_name.capitalize().replace('_', ' ')
        indent += self.extra_indent
        value_str = f"{value.numerator}/{value.denominator}" if isinstance(value, Fraction) else str(value)
        n = (39 - len(desc_str) - len(value_str) - len(unit_str))*" "
        return f"     {indent}- {desc_str}: {value_str}{unit_str} {n}({param_name})"

    def log_param(self, param_name, default, description="", unit="", indent="  ", can_skip=False):
        res = self._log_param(param_name, default, description, unit, indent, can_skip)
        if res is not None:
            self.logger(res)

    def add(self, param_name, default, description="", unit="", indent="  ", can_skip=False):
        res = self._log_param(param_name, default, description, unit, indent, can_skip)
        if res is not None:
            self.params.append(res)

    def reset(self):
        self.params = []

    def sub_section(self, name):
        self.logger(f"       {name}:")

    def section(self, name):
        self.logger("       " + "-" * 40)
        self.sub_section(name)

    def flush(self, name, is_sub=False):
        if self.params:
            if is_sub:
                self.sub_section(self.extra_indent + name)
            else:
                self.section(name)
            for p in self.params:
                self.logger(p)

    def structure(self, ch=64, depth=6, with_lstm=False, with_ch_tm=False):
        self.reset()
        self.add('channels', ch, "Initial hidden channels")
        self.add('depth', depth, "Number of U-Net layers")
        self.add('growth', 2.0, "Channel growth factor per layer")
        self.add('rewrite', True, "Use 1x1 convolutions in blocks")
        if with_lstm:
            self.add('lstm_layers', 0, "Number of main LSTM layers", can_skip=True)
        if with_ch_tm:
            self.add('channels_time', None, "Specific channels for time branch", can_skip=True)
        self.flush("Structure")

    def convolutions(self, advanced=False):
        self.reset()
        self.add('kernel_size', 8)
        self.add('stride', 4)
        if advanced:
            self.add('time_stride', 2, "Final time layer stride")
        self.add('context', 1, "Decoder context window size")
        if advanced:
            self.add('context_enc', 0, "Encoder context window size")
        self.flush("Convolutions")

    def normalization(self):
        self.reset()
        self.add('norm_starts', 4, "Start at layer")
        self.add('norm_groups', 4, "Number of groups")
        self.flush("Normalization")

    def dconv(self, full=True):
        if self.get('dconv_mode', 1) <= 0:
            return
        self.reset()
        where = ['', 'In encoder', 'In decoder', 'In encoder and decoder'][self.get('dconv_mode', 1)]
        self.add('dconv_mode', 1, where)
        self.add('dconv_depth', 2, "Number of layers in DConv branch")
        if full:
            comp = 4
            init = 1e-4
        else:
            comp = 8
            init = 1e-3
        self.add('dconv_comp', comp, "Channel compression factor")
        self.add('dconv_init', init, "Initial scale")
        if full:
            self.add('dconv_attn', 4, "Layer to start attention in DConv")
            self.add('dconv_lstm', 4, "Layer to start LSTM in DConv")
        self.flush("DConv Residual Branch")

    def stft(self):
        self.reset()
        self.add('nfft', 4096, "Frequency Bins")
        # Decode the method
        cac = self.get('cac')
        niters = self.get('wiener_iters')
        if cac:
            zout = "Complex as Channels (CaC)"
        elif niters >= 0:
            zout = "Wiener filtering"
        else:
            zout = "Naive iSTFT from masking"
        self.add('___', zout, "Framework")
        self.add('cac', True, "Use Complex as Channels")
        if not self.get('cac', True):
            self.add('wiener_iters', 0, "Wiener filter iterations")
        self.flush("STFT")

    def freq_branch(self):
        self.reset()
        def_ratio = None
        if self.get('multi_freqs') == []:
            def_ratio = []
        self.add('multi_freqs', def_ratio, "Ratios for frequency band splitting")
        if self.get('multi_freqs'):
            self.add('multi_freqs_depth', 2, "Layers to apply frequency splitting")
        self.add('freq_emb', 0.2, "Frequency embedding weight")
        if self.get('freq_emb'):
            indent = "    "
            self.add('emb_scale', 10, "Scale", indent=indent)
            self.add('emb_smooth', True, "Smooth", indent=indent)
        self.flush("Frequency Branch")

    def demucs(self):
        """Logs the parameters for the original Demucs class."""
        self.log_type("Classic Waveform Demucs (Demucs)")
        self.log_param('sources', [], "Target source names")
        self.log_param('segment', 40, "Segment size", unit="s")
        # --- Structure & Channels ---
        self.structure(ch=64, depth=6, with_lstm=True)
        # --- Convolutions ---
        self.convolutions()

        self.reset()
        self.add('gelu', True, "GeLU (not ReLU)")
        if self.get('rewrite', True):
            self.add('glu', True, "GLU in 1x1 rewrite (not ReLU)")
        self.flush("Activations")
        # --- Normalization ---
        self.normalization()
        # --- DConv Residual Branch ---
        self.dconv()
        # --- Pre/Post Processing ---
        self.reset()
        self.add('resample', True, "Use 2x resampling")
        self.add('normalize', True, "Normalize audio on-the-fly")
        self.flush("Processing")

    def hdemucs(self):
        """Logs the parameters for the HDemucs (Hybrid Spectrogram/Waveform) class."""
        self.log_type("Hybrid Demucs (Spectrogram + Waveform) (HDemucs)")
        self.log_param('sources', [], "Target source names")
        self.log_param('segment', 40, "Segment size", unit="s")
        # --- Structure & Channels ---
        self.structure(ch=48, depth=6, with_ch_tm=True)
        # --- STFT & Spectrogram ---
        self.stft()
        # --- Frequency Branch ---
        self.freq_branch()
        # --- Convolutions ---
        self.convolutions(advanced=True)
        # --- Normalization ---
        self.normalization()
        # --- DConv Residual Branch (defaults are different from Demucs) ---
        self.dconv()

    def htdemucs(self):
        """Logs the parameters for the HTDemucs (Hybrid Transformer) class."""
        self.log_type("Hybrid Transformer Demucs (HTDemucs)")
        self.log_param('sources', [], "Target source names")
        self.log_param('segment', 10, "Segment size", unit="s")
        # --- Structure & Channels (defaults are different from HDemucs) ---
        self.structure(ch=48, depth=4)
        # --- STFT & Spectrogram ---
        self.stft()
        # --- Frequency Branch ---
        self.freq_branch()
        # --- Convolutions ---
        self.convolutions(advanced=True)
        # --- Normalization ---
        self.normalization()
        # --- DConv (defaults are different) ---
        self.dconv(full=False)
        # --- Transformer Block ---
        if self.get('t_layers', 5) > 0:
            self.extra_indent = "  "
            # --- Main Transformer ---
            self.reset()
            if self.get('bottom_channels', 0):
                self.add('bottom_channels', 0, "Channels forced to")
            self.add('t_hidden_scale', 4.0, "Hidden scale")
            self.add('t_layers', 5, "Number of transformer layers")
            self.add('t_heads', 8, "Number of attention heads")
            self.add('t_dropout', 0.0, "Dropout")
            self.flush("Transformer")
            # --- Positional Embeddings ---
            self.reset()
            self.add('t_emb', 'sin', "Type")
            self.add('t_weight_pos_embed', 1.0, "Weight", can_skip=True)
            t_emb = self.get('t_emb', 'sin')
            if t_emb == 'scaled':
                self.add('t_max_positions', 10000, "Max positions")
            elif t_emb == 'sin':
                self.add('t_max_period', 10000.0, "Max period")
                self.add('t_sin_random_shift', 0, "Random shift", can_skip=True)
            elif t_emb == 'cape':
                self.add('t_cape_mean_normalize', True, "Cape normalize")
                self.add('t_cape_glob_loc_scale', [5000.0, 1.0, 1.4], "Cape params")
                if self.get('t_cape_augment', True):
                    rlogger.warning("t_cape_augment is True in loaded model, should be False for inference.")
            self.flush("Positional Embeddings", is_sub=True)
            # --- Transformer Normalization ---
            self.reset()
            self.add('t_norm_first', True, "Before attention/FFN")
            self.add('t_norm_in', True, "Before pos. embedding")
            if self.get('t_norm_in', True):
                self.add('t_norm_in_group', False, "On all timesteps")
            self.add('t_group_norm', False, "Of encoder on all timesteps")
            self.add('t_norm_out', True, "GroupNorm at end of layers")
            self.flush("Normalization", is_sub=True)
            # --- Transformer Misc ---
            self.reset()
            self.add('t_cross_first', False, "Cross-attention is the first layer")
            self.add('t_layer_scale', True, "Layer scale")
            self.add('t_gelu', True, "GeLU (not ReLU)")
            self.flush("Various", is_sub=True)
            # --- Sparsity ---
            # Log sparsity details only if sparse attention is enabled
            self.reset()
            is_sparse = self.get('t_sparse_self_attn', False)
            self.add('t_sparse_self_attn', False, "Use sparse self-attention")
            if is_sparse:
                self.add('t_sparse_cross_attn', False, "Sparse cross-attention")
                self.add('t_auto_sparsity', False, "Automatic sparsity")
                auto_sparsity = self.get('t_auto_sparsity', False)
                if not auto_sparsity:
                    self.add('t_mask_type', 'diag', "Masking pattern")
                    self.add('t_mask_random_seed', 42, "Mask seed")
                    mask_t = self.get('t_mask_type', 'diag')
                    if 'diag' in mask_t:
                        self.add('t_sparse_attn_window', 500, "Window size")
                    if 'global' in mask_t:
                        self.add('t_global_window', 100, "Window size")
                    if 'random' in mask_t:
                        self.add('t_sparsity', 0.95, "Sparsity for random mask", unit="%")
            self.flush("Sparsity", is_sub=True)
            # Training only
            # self.add(logger, kwargs, 't_weight_decay', 0.0, "Weight decay", extra=False)
            # self.add(logger, kwargs, 't_lr', None, "Learning rate", extra=False)
            # self.add(logger, kwargs, 't_cape_augment', True, "Learning rate", extra=False)
            # self.add(logger, kwargs, 'rescale', 0.1, "Rescale trick", extra=False)
            self.extra_indent = ""
