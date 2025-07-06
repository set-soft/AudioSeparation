# Short-Time Fourier Transform (STFT).
import logging
import numpy as np
import torch
from tqdm import tqdm
# Local imports
from ..utils.misc import NODES_NAME
from ..utils.torch import model_to_target

logger = logging.getLogger(f"{NODES_NAME}.stft")


class STFT:
    def __init__(self, n_fft, hop_length, dim_f, device):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.dim_f = dim_f
        self.device = device

    def __call__(self, x):
        window = self.window.to(x.device)
        batch_dims = x.shape[:-2]
        c, t = x.shape[-2:]
        x = x.reshape([-1, t])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       window=window, center=True, return_complex=True)
        x = torch.view_as_real(x)
        x = x.permute([0, 3, 1, 2])
        x = x.reshape([*batch_dims, c, 2, -1, x.shape[-1]]
                      ).reshape([*batch_dims, c * 2, -1, x.shape[-1]])

        return x[..., :self.dim_f, :]

# Original code
#     def inverse(self, x):
#         window = self.window.to(x.device)
#         batch_dims = x.shape[:-3]
#         c, f, t = x.shape[-3:]
#         n = self.n_fft // 2 + 1
#         f_pad = torch.zeros([*batch_dims, c, n - f, t]).to(x.device)
#         x = torch.cat([x, f_pad], -2)
#         x = x.reshape([*batch_dims, c // 2, 2, n, t]).reshape([-1, 2, n, t])
#         x = x.permute([0, 2, 3, 1])
#         x = x[..., 0] + x[..., 1] * 1.j
#         x = torch.istft(x, n_fft=self.n_fft,
#                         hop_length=self.hop_length, window=window, center=True)
#         x = x.reshape([*batch_dims, 2, -1])
#
#         return x

    # Annotated code
    def inverse(self, x):
        """
        Correctly performs the inverse STFT.
        x is the output of the model, shape (B, C, F, T)
        With C == 4 (L/R as complex)
        """
        window = self.window.to(x.device)
        batch_dims = x.shape[:-3]
        c, f, t = x.shape[-3:]  # c is 4 here
        assert c == 4

        n = self.n_fft // 2 + 1  # Full number of frequency bins

        # Pad the frequency dimension back to its original size
        f_pad = torch.zeros([*batch_dims, c, n - f, t], device=x.device)
        x = torch.cat([x, f_pad], -2)

        # The key is to correctly un-stack the 4 channels back into (C, 2)
        # where C=2 (stereo) and 2 is real/imag.

        # Reshape (B, 4, F, T) -> (B, 2, 2, F, T)
        # The new dimensions are (B, stereo_channels, real_imag, F, T)
        x = x.reshape([*batch_dims, 2, 2, n, t])

        # Permute to get (B, stereo_channels, F, T, real_imag)
        x = x.permute(0, 1, 3, 4, 2)

        # Ensure the tensor is contiguous in memory before the final view
        x = x.contiguous()

        # Now, view_as_complex will work on the last dimension
        x = torch.view_as_complex(x)  # Shape: (B, C, F, T) complex

        # Reshape for istft: (B, C, F, T) -> (B*C, F, T)
        x = x.reshape(-1, n, t)

        # Perform inverse STFT
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, center=True)

        # Reshape back to (B, C, num_samples)
        x = x.reshape([*batch_dims, 2, -1])

        return x


def stft_get_chunks(samples, n_fft, segment_size=256, hop_length=1024):
    chunk_size = hop_length * (segment_size - 1)
    step = chunk_size - n_fft
    return 1 + (samples - chunk_size + step - 1) // step


def stft_chunk_process(waveform, d, model_run, device, segment_size=256, hop_length=1024, progress_bar_ui=None):
    """ Inference helper for models using STFT information as input """
    n_fft = d['mdx_n_fft_scale_set']
    compensate = d['compensate']

    mix_np = waveform.numpy()

    stft = STFT(n_fft, hop_length, model_run.dim_f, device)
    trim = n_fft // 2
    chunk_size = hop_length * (segment_size - 1)
    gen_size = chunk_size - 2 * trim

    # --- Overlap-Add Loop ---
    pad = gen_size + trim - (mix_np.shape[1] % gen_size)
    # Padded mixture as a numpy array
    # mixture = np.concatenate((np.zeros((2, trim)), mix_np, np.zeros((2, pad))), axis=1)
    mixture = np.concatenate((np.zeros((2, trim), dtype=np.float32), mix_np, np.zeros((2, pad), dtype=np.float32)), axis=1)

    step = chunk_size - n_fft  # Correct step size for large overlap

    result = np.zeros((1, 2, mixture.shape[1]), dtype=np.float32)
    divider = np.zeros((1, 2, mixture.shape[1]), dtype=np.float32)

    total_chunks = 1 + (mixture.shape[1] - chunk_size + step - 1) // step
    logger.info(f"⚙️  Processing {total_chunks} chunks...")
    model_run.target_device = device

    with model_to_target(model_run):
        for i in tqdm(range(0, mixture.shape[1] - chunk_size + 1, step)):
            start = i
            end = i + chunk_size

            mix_part = mixture[:, start:end]

            # Convert just the chunk to a tensor for the model
            mix_part_tensor = torch.from_numpy(mix_part).unsqueeze(0).to(device)
            spek = stft(mix_part_tensor)
            spec_pred = model_run(spek)

            # Get the output back as a numpy array
            tar_waves_np = stft.inverse(spec_pred).cpu().detach().numpy()

            # Hanning window applied to the output before adding
            # window = np.hanning(chunk_size)
            window = np.hanning(chunk_size).astype(np.float32)
            window = np.tile(window[None, None, :], (1, 2, 1))

            result[..., start:end] += tar_waves_np * window
            divider[..., start:end] += window

            if progress_bar_ui:
                progress_bar_ui.update(1)

    # --- Final Normalization and Trimming ---
    divider[divider == 0] = 1.0
    main_wav_np = (result[0] / divider[0])  # Get the 2D array
    main_wav_np = main_wav_np[:, trim:-trim][:, :mix_np.shape[1]]
    main_wav_np *= compensate

    # Convert final result back to a torch tensor for saving
    return torch.from_numpy(main_wav_np)


# ############################################################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# License: MIT
#
# Convenience wrapper to perform STFT and iSTFT
# ############################################################################################################################


def spectro(x, n_fft=512, hop_length=None, pad=0):
    *other, length = x.shape
    x = x.reshape(-1, length)
    is_mps = x.device.type == 'mps'
    if is_mps:
        x = x.cpu()
    z = torch.stft(x,
                   n_fft * (1 + pad),
                   hop_length or n_fft // 4,
                   window=torch.hann_window(n_fft).to(x),
                   win_length=n_fft,
                   normalized=True,
                   center=True,
                   return_complex=True,
                   pad_mode='reflect')
    _, freqs, frame = z.shape
    return z.view(*other, freqs, frame)


def ispectro(z, hop_length=None, length=None, pad=0):
    *other, freqs, frames = z.shape
    n_fft = 2 * freqs - 2
    z = z.view(-1, freqs, frames)
    win_length = n_fft // (1 + pad)
    is_mps = z.device.type == 'mps'
    if is_mps:
        z = z.cpu()
    x = torch.istft(z,
                    n_fft,
                    hop_length,
                    window=torch.hann_window(win_length).to(z.real),
                    win_length=win_length,
                    normalized=True,
                    length=length,
                    center=True)
    _, length = x.shape
    return x.view(*other, length)
