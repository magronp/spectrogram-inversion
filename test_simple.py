#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.algos import spectrogram_inversion
from helpers.data_io import load_src
from librosa import stft
from open_unmx.estim_spectro import estim_spectro_from_mix
from matplotlib import pyplot as plt


# Set random seed for reproducibility
np.random.seed(1234)

# Parameters
sample_rate = 16000
n_fft = 1024
win_length = 1024
hop_length = 256
win_type = 'hann'
max_iter = 20

# Load data (start from mixture 50 since the first 50 are for validation)
audio_path = 'data/SNR_0/0/'
src_ref, mix = load_src(audio_path, sample_rate)
mix_stft = stft(mix, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=win_type)

# Estimate the magnitude spectrograms
spectro_mag = estim_spectro_from_mix(mix[:, np.newaxis])

# Apply the spectrogram inversion algorithms
algos_list = ['MISI', 'Mix+Incons', 'Mix+Incons_hardMag', 'Mag+Incons_hardMix']
nalgos = len(algos_list)
sdr_all = np.zeros((max_iter+1, nalgos))

for ia, algo in enumerate(algos_list):
    _, _, sdr = spectrogram_inversion(mix_stft, spectro_mag, algo=algo, consistency_weigth=1, max_iter=max_iter,
                                        reference_sources=src_ref, win_length=win_length, hop_length=hop_length,
                                        window=win_type, compute_error=True)
    sdr_all[:, ia] = sdr

plt.figure()
plt.plot(sdr_all)
plt.ylabel('SDR (dB)', fontsize=16)
plt.xlabel('Iterations', fontsize=16)
plt.legend(algos_list)
plt.grid('on')
plt.show()
plt.tight_layout()

# EOF

