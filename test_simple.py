#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from helpers.algos import get_score, amplitude_mask, misi, spectrogram_inversion
from helpers.data_io import load_src, record_src
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
          'input_SNR_list': [10, 0, -10],
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

# Amplitude mask
src_est_am = amplitude_mask(spectro_mag, mix_stft, win_length=params['win_length'],
                            hop_length=params['hop_length'], window=params['win_type'])
sdr_am = get_score(src_ref, src_est_am)

algos_list = ['MISI', 'Mix+Incons', 'Mix+Incons_hardMag', 'Mag+Incons_hardMix']
n_algos = len(algos_list)
sdri_all = np.zeros((params['max_iter']+1, n_algos))

for ia, algo in enumerate(algos_list):
    src_est, error, sdr = spectrogram_inversion(mix_stft, spectro_mag, params['win_length'], algo=algo,
                                                consistency_weigth=0.1,
                                                max_iter=params['max_iter'], reference_sources=src_ref,
                                                hop_length=params['hop_length'],
                                                window=params['win_type'], compute_error=True)
    sdri_all[:, ia] = sdr


_, _, sdr_icons_hardmix = spectrogram_inversion(mix_stft, spectro_mag, params['win_length'], algo='Incons_hardMix',
                                            consistency_weigth=1,
                                            max_iter=params['max_iter'], reference_sources=src_ref,
                                            hop_length=params['hop_length'],
                                            window=params['win_type'], compute_error=True)

plt.figure()
plt.plot(sdri_all)
plt.legend(algos_list)
plt.show()

# EOF

