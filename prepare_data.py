#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import librosa
from helpers.stft import my_stft, my_istft
from helpers.data_io import record_src
np.random.seed(0)  # Set random seed for reproducibility

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def adjust_noise_at_isnr(clean, noise, input_snr=0):
    """ Adjust the noise level at the given input SNR
    Args:
        clean: numpy.ndarray (nsamples, 1) - clean signal
        noise: numpy.ndarray (nsamples, 1) - noise signal
        input_snr: float - input SNR
    Returns:
        noise: numpy.ndarray (nsamples, 1) - noise signal adjusted at the input_SNR level
    """
    noise = noise * (np.linalg.norm(clean) / np.linalg.norm(noise)) * (10 ** (- input_snr / 20))

    return noise


def prep_dataset(params):
    """ Compute the mixtures at various input SNR for creating the dataset
    Args:
        params: dictionary with fields:
            'noise_data_dir': string - the path to the noise data
            'speech_data_dir': string - the path to the clean speech data
            'sample rate': int - the sampling frequency
            'n_mix': int - the number of mixtures to create
            'input_SNR_list': list - the list of input SNRs to consider
    """

    # Load the noises, keep the 1st of the 16 channels
    noise_data_list = librosa.util.find_files(params['noise_data_dir'], ext='wav')
    noise_data_list = [n for n in noise_data_list if n.__contains__('ch01')]
    noise_total = np.array([])
    for n in noise_data_list:
        noise_total = np.concatenate((noise_total, librosa.core.load(n, sr=params['sample_rate'])[0]))
    noise_beg_ind = []
    noise_total_len = noise_total.shape[0]

    # Load the list of clean speech files and shuffle it
    speech_data_list = librosa.util.find_files(params['speech_data_dir'], ext='wav')
    np.random.shuffle(speech_data_list)

    # Create the mixtures
    for n in range(params['n_mix']):
        print('Creating mix ' + str(n+1) + ' / ' + str(params['n_mix']))

        # Load the speech
        speech_data = speech_data_list[n]
        clean = librosa.core.load(speech_data, sr=params['sample_rate'])[0]
        len_clean = clean.shape[0]

        # Take a piece of the noise of same length
        rand_sample_noise_beg = np.random.randint(noise_total_len)
        noise = noise_total[rand_sample_noise_beg:rand_sample_noise_beg+len_clean]

        # Collect the noise index (to further study the results as a function of the noise type)
        noise_beg_ind.append(rand_sample_noise_beg)

        # Adjust the input SNR and record audio
        for iSNR in params['input_SNR_list']:

            # Adjust the noise at target input SNR
            noise_adj = adjust_noise_at_isnr(clean, noise, input_snr=iSNR)
            src_ref = np.concatenate((clean[:, np.newaxis], noise_adj[:, np.newaxis]), axis=1)

            # Take the STFT and iSTFT to ensure the length is fixed
            src_ref_stft = my_stft(src_ref, n_fft=params['n_fft'], hop_length=params['hop_length'],
                                   win_length=params['win_length'], win_type=params['win_type'])
            src_ref = my_istft(src_ref_stft, hop_length=params['hop_length'], win_length=params['win_length'],
                         win_type=params['win_type'])

            # Create the folder to record the wav (if necessary)
            rec_dir = 'data/SNR_' + str(iSNR) + '/' + str(n)
            if not os.path.exists(rec_dir):
                os.makedirs(rec_dir)

            # Record wav
            record_src(rec_dir + '/', src_ref, params['sample_rate'], rec_mix=True)

    # Get the indices of noise type for each mixture in the test set and record
    noise_beg_ind = np.array(noise_beg_ind)
    noise_beg_ind = noise_beg_ind[50:]
    ind_noise_1 = noise_beg_ind < noise_total_len // 3
    ind_noise_3 = noise_beg_ind > 2*noise_total_len // 3
    ind_noise_2 = 1 - (ind_noise_1+ind_noise_3)
    np.savez('data/noise_ind.npz', ind_noise_1=ind_noise_1, ind_noise_2=ind_noise_2, ind_noise_3=ind_noise_3)

    return


if __name__ == '__main__':

    # Parameters
    params = {'sample_rate': 16000,
              'n_mix': 100,
              'input_SNR_list': [10, 0, -10],
              'speech_data_dir': 'data/VoiceBank',
              'noise_data_dir': 'data/DEMAND',
              'win_length': 1024,
              'hop_length': 256,
              'n_fft': 1024,
              'win_type': 'hann'
              }

    # Prepare the whole dataset
    prep_dataset(params)

# EOF
