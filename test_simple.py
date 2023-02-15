#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from helpers.algos import get_score, amplitude_mask, spectrogram_inversion
from helpers.data_io import load_src
from librosa import stft
from open_unmx.estim_spectro import estim_spectro_from_mix
from matplotlib import pyplot as plt

# Set random seed for reproducibility
np.random.seed(1234)

# Parameters
params = {'sample_rate': 16000,
          'win_length': 1024,
          'hop_length': 256,
          'n_fft': 1024,
          'win_type': 'hann',
          'max_iter': 20,
          'n_mix': 50,
          'input_SNR_list': [10, 0, -10]
          }

# Define some parameters and initialize the SNR array
n_isnr = len(params['input_SNR_list'])
sdr_am = np.zeros((n_isnr, params['n_mix']))
sdr_misi = np.zeros((n_isnr, params['n_mix']))

index_isnr, isnr = 1, 0
index_mix = 0

# Load data (start from mixture 50 since the first 50 are for validation)
audio_path = 'data/SNR_' + str(isnr) + '/' + str(index_mix + params['n_mix']) + '/'
src_ref, mix = load_src(audio_path, params['sample_rate'])
mix_stft = stft(mix, n_fft=params['n_fft'], hop_length=params['hop_length'],
                   win_length=params['win_length'], window=params['win_type'])

# Estimate the magnitude spectrograms
spectro_mag = estim_spectro_from_mix(mix[:, np.newaxis])
algos_list = ['Mix+Incons', 'Mix+Incons_optweights', 'Mix+Incons_hardMag', 'Mix+Incons_hardMag_optweights', 'Mag+Incons_hardMix']

for algo in algos_list:
    _, error, sdr = spectrogram_inversion(mix_stft, spectro_mag, algo=algo, consistency_weigth=1,
                                      max_iter=params['max_iter'], reference_sources=src_ref,
                                      win_length=params['win_length'], hop_length=params['hop_length'],
                                      window=params['win_type'], compute_error=True)
    plt.plot(sdr, label=algo)
plt.legend()

# EOF

