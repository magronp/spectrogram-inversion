#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

'''
This script is largely adapted from the original Open Unmix code,
which is available at: https://github.com/sigsep/open-unmix-pytorch
If you use it, please acknowledge it by citing the corresponding paper:
F.-R. Stoter, S. Uhlich, A. Liutkus and Y. Mitsufuji,
"Open-Unmix - A Reference Implementation for Music Source Separation",
Journal of Open Source Software, 2019.
'''

import numpy as np
import json
from pathlib import Path
import torch
from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch.nn as nn
from helpers.data_io import load_src


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)


class OpenUnmix(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*2,
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size,
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x = self.transform(x)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        mix = x.detach().clone()

        # crop
        x = x[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x += self.input_mean
        x *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x = self.fc1(x.reshape(-1, nb_channels*self.nb_bins))
        # normalize every instance in a batch
        x = self.bn1(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x = torch.tanh(x)

        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x)

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = nn.functional.relu(x)

        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)

        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        # since our output is non-negative, we can apply RELU
        x = nn.functional.relu(x) * mix

        return x


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )
    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def load_model(target, model_dir='data/open_unmx/', device='cpu'):

    model_path = Path(model_dir).expanduser()

    # load model from disk
    with open(Path(model_path, target + '.json'), 'r') as stream:
        results = json.load(stream)

    target_model_path = next(Path(model_path).glob("%s*.pth" % target))
    state = torch.load(
        target_model_path,
        map_location=device
    )

    max_bin = bandwidth_to_max_bin(
        state['sample_rate'],
        results['args']['nfft'],
        results['args']['bandwidth']
    )

    unmix = OpenUnmix(
        n_fft=results['args']['nfft'],
        n_hop=results['args']['nhop'],
        nb_channels=results['args']['nb_channels'],
        hidden_size=results['args']['hidden_size'],
        max_bin=max_bin
    )

    unmix.load_state_dict(state)
    unmix.stft.center = True
    unmix.eval()
    unmix.to(device)

    return unmix


def estim_spectro_from_mix(audio, model_dir='data/open_unmx/', device='cpu'):

    # Create a 2-channel audio for feeding open unmix
    audio = np.repeat(audio, 2, axis=1)

    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    spectro_mag_est = []
    targets = ['speech', 'noise']

    print("Estimate spectrograms from the mix...")
    for j, target in enumerate(targets):
        unmix_target = load_model(
            target=target,
            model_dir=model_dir,
            device=device
        )
        vj = unmix_target(audio_torch).cpu().detach().numpy()
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        spectro_mag_est.append(vj[:, 0, ...])  # remove sample dim
        source_names += [target]
    print("Done")
    spectro_mag_est = np.transpose(np.array(spectro_mag_est), (1, 3, 2, 0))

    # Back to monochannel and remove extra 0 frames
    spectro_mag_est = np.transpose(spectro_mag_est[:, :, 0, :], (2, 1, 0))

    return spectro_mag_est


if __name__ == '__main__':

    # Load a mixture
    sample_rate = 16000
    audio_path = 'data/SNR_0/0/'
    _, mix = load_src(audio_path, sample_rate)
    audio = mix[:, np.newaxis]

    # Estimate the magnitude spectrograms directly
    model_dir = 'data/open_unmx/'
    spectro_mag = estim_spectro_from_mix(audio, model_dir=model_dir)

# EOF
