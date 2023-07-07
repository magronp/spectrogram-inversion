#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.algos import spectrogram_inversion, amplitude_mask
import librosa
from librosa import stft
from helpers.openunmix import estim_spectro_from_mix
import soundfile


# Set random seed for reproducibility
np.random.seed(1234)

# Parameters
sample_rate = 16000
n_fft = 1024
win_length = 1024
hop_length = 256
win_type = 'hann'
max_iter = 20

# Load the noisy mixture
audio_path = 'example/'
mix = librosa.core.load(audio_path + 'mix.wav', sr=sample_rate)[0]
mix_stft = stft(mix, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=win_type)

# Estimate the magnitude spectrograms
spectro_mag = estim_spectro_from_mix(mix)

# Apply the spectrogram inversion algorithms
algos_list = ['MISI', 'Mix+Incons', 'Mix+Incons_hardMag', 'Mag+Incons_hardMix', 'Incons_hardMix']
nalgos = len(algos_list)
sdr_all = np.zeros((max_iter+1, nalgos))

# Non-iterative methods (AM and Incons_hardMix)

# Amplitude mask
src_est = amplitude_mask(spectro_mag, mix_stft, win_length, hop_length, win_type)
soundfile.write(audio_path + 'speech_est_' + 'AM' + '.wav', src_est[0, :], sample_rate)


# Iterative algorithms
for ia, algo in enumerate(algos_list):
    src_est = spectrogram_inversion(mix_stft, spectro_mag, algo=algo, consistency_weigth=10, max_iter=max_iter,
                                    reference_sources=None, win_length=win_length, hop_length=hop_length,
                                    window=win_type, compute_error=True)[0]
    soundfile.write(audio_path + 'speech_est_' + algo + '.wav', src_est[0, :], sample_rate)

# EOF

