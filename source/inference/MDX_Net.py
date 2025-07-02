# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnología Industrial
# License: GPLv3
# Project: ComfyUI-AudioSeparation
# Developed using:
# - Netron to inspect the topology
# - onnx2pytorch.ConvertModel to find mapping details
# - Gemini 2.5 Pro to analyze the network, write the code and debug it
# I never saw the original implementation, this is a reconstruction from Kim_Vocals_2.onnx
#
# Found geometries:
#
#  dim_f | channels | stages | ~params
# -------|----------|--------|------------
#  3072  |    48    |   5    | 16 684 228
#  2560  |    48    |   5    | 14 763 012
#  2048  |    48    |   5    | 13 191 108
#  2048  |    32    |   5    |  7 420 548
#  2048  |    32    |   4    |  5 478 276
from torch import nn


class FrequencyBranch(nn.Module):
    """ Frequency-domain branch with Linear -> BatchNorm2d -> ReLU sequences. """
    def __init__(self, channels, freq_dim, hidden_dim, bn_eps):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim, bias=False),
            # Using the standard, verified nn.BatchNorm2d
            nn.BatchNorm2d(num_features=channels, eps=bn_eps),
            nn.ReLU(True),
            nn.Linear(hidden_dim, freq_dim, bias=False),
            nn.BatchNorm2d(num_features=channels, eps=bn_eps),
            nn.ReLU(True)
        )

    def forward(self, x):
        # Relies on PyTorch's nn.Linear broadcasting over the first 3 dims of (B,C,T,F)
        # And nn.BatchNorm2d operating on the C dimension of the 4D tensor.
        return self.sequence(x)


class TimeBranch(nn.Module):
    """
    Time-domain branch using 3x3 convolutions.
    """
    def __init__(self, channels):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding='same', bias=True),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding='same', bias=True),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding='same', bias=True),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.sequence(x)


class TDF_Block(nn.Module):
    """ The main processing block, combining time and frequency branches.
        This is a sequential-residual block, as shown in the ONNX graph. """
    def __init__(self, channels, freq_dim, hidden_dim, bn_eps):
        super().__init__()
        self.time_branch = TimeBranch(channels)
        self.freq_branch = FrequencyBranch(channels, freq_dim, hidden_dim, bn_eps)

    def forward(self, x):
        # 1. The input 'x' goes through the time branch first.
        time_out = self.time_branch(x)

        # 2. The output of the time branch is then fed into the frequency branch.
        freq_out = self.freq_branch(time_out)

        # 3. The final result is a residual connection:
        #    Output of Time Branch + Output of Frequency Branch
        return time_out + freq_out


class Transpose(nn.Module):
    """ A simple nn.Module wrapper for the permute operation """
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)


class MDX_Net(nn.Module):
    """
    The complete U-Net architecture.
    Fully parametric for frequency bins, channels, and number of stages.
    This version uses your elegant interlaced ModuleList design for a clean,
    dynamic structure that correctly matches the ONNX graph order.
    """
    def __init__(self, dim_f=3072, ch=48, num_stages=5):
        super().__init__()
        # Validate input
        if num_stages < 1 or num_stages > 12:
            raise ValueError(f"num_stages must be between 1 and 12, but got {num_stages}")

        self.num_stages = num_stages

        # Define shared BatchNorm parameters
        BN_EPS = 9.999999747378752e-06
        freq_hidden_dim = dim_f // 8
        # Allow others to know our creation parameters
        self.dim_f = dim_f
        self.ch = ch
        self.num_stages = num_stages

        # --- Initial Chain (always exists) ---
        self.initial_conv = nn.Conv2d(4, ch, 1, bias=True)
        self.initial_relu = nn.ReLU(True)
        self.initial_transpose = Transpose(dims=(0, 1, 3, 2))

        # --- Encoder Path with Interlaced Layers ---
        # We create all stages and downsamplers as ModuleLists
        self.enc_stages = nn.ModuleList()

        # This list defines the channel progression
        # e.g., for ch=48: (48, 96, 144, 192, 240, 288)
        channels = [ch * (i + 1) for i in range(num_stages + 1)]

        for i in range(num_stages):
            # Append the TDF_Block stage
            self.enc_stages.append(TDF_Block(channels[i], dim_f // (2**i), freq_hidden_dim // (2**i), BN_EPS))
            # Append the downsampling block immediately after
            self.enc_stages.append(nn.Sequential(nn.Conv2d(channels[i], channels[i+1], 2, 2, bias=True), nn.ReLU(True)))

        # --- Bottleneck ---
        bottleneck_in_ch = channels[num_stages]
        self.bottleneck = TDF_Block(bottleneck_in_ch, dim_f // (2**num_stages), freq_hidden_dim // (2**num_stages), BN_EPS)

        # --- Decoder Path with Interlaced Layers ---
        self.dec_stages = nn.ModuleList()
        for i in range(num_stages):
            dec_idx = num_stages - 1 - i
            # Upsampler takes bottleneck/previous stage channels and outputs encoder stage channels
            in_ch = channels[dec_idx + 1]
            out_ch = channels[dec_idx]

            # Append the upsampling block
            seq = nn.Sequential(nn.ConvTranspose2d(in_ch, out_ch, 2, 2, bias=True),
                                nn.BatchNorm2d(out_ch, eps=BN_EPS),
                                nn.ReLU(True))
            self.dec_stages.append(seq)
            # Append the TDF_Block stage immediately after
            self.dec_stages.append(TDF_Block(out_ch, dim_f // (2**dec_idx), freq_hidden_dim // (2**dec_idx), BN_EPS))

        # --- Final Chain (always exists) ---
        self.final_transpose = Transpose(dims=(0, 1, 3, 2))
        self.final_conv = nn.Conv2d(ch, 4, 1, bias=True)

    def forward(self, x):
        # Initial processing
        x = self.initial_conv(x)
        x = self.initial_relu(x)
        x = self.initial_transpose(x)

        # --- Dynamic Encoder Path ---
        skip_connections = []
        # Encoder runs through the interlaced list
        for i in range(0, self.num_stages*2, 2):
            s = self.enc_stages[i](x)
            skip_connections.append(s)
            x = self.enc_stages[i+1](s)

        # --- Bottleneck ---
        x = self.bottleneck(x)

        # --- Dynamic Decoder Path ---
        skip_connections.reverse()  # Reverse for easy lookup
        # Decoder also runs through its interlaced list
        for i in range(0, self.num_stages * 2, 2):
            x = self.dec_stages[i](x)
            x = x * skip_connections[i//2]
            x = self.dec_stages[i+1](x)

        # Final processing
        output = self.final_transpose(x)
        output = self.final_conv(output)

        return output
