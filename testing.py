#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from helpers.algos import get_score, amplitude_mask, misi
from helpers.data_io import load_src, record_src
from librosa import stft
from open_unmx.estim_spectro import estim_spectro_from_mix
from helpers.plotter import plot_test_results, plot_test_results_pernoise

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def testing(params, test_sdr_path='outputs/test_sdr.npz'):
    """ Run the proposed algorithm on the test subset and the MISI and AM baselines
    Args:
        params: dictionary with fields:
            'sample rate': int - the sampling frequency
            'n_mix': int - the number of mixtures to process
            'max_iter': int - the nomber of iterations of the proposed algorithm
            'input_SNR_list': list - the list of input SNRs to consider
            'beta_range': numpy array - the beta-divergence parameter grid
            'hop_length': int - the hop size of the STFT
            'win_length': int - the window length
            'n_fft': int - the number of FFT points
            'win_type': string - the STFT window type (e.g., Hann, Hamming, Blackman...)
        test_sdr_path: string - the path where to store the test SDR
    """

    # Define some parameters and initialize the SNR array
    n_isnr = len(params['input_SNR_list'])
    sdr_am = np.zeros((n_isnr, params['n_mix']))
    sdr_misi = np.zeros((n_isnr, params['n_mix']))

    # Load the optimal step sizes from validation
    #gd_step_opt = np.load('outputs/val_gd_step.npz')['gd_step']

    # Loop over iSNRs, mixtures and parameters
    for index_isnr, isnr in enumerate(params['input_SNR_list']):
        for index_mix in range(params['n_mix']):

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
            sdr_am[index_isnr, index_mix] = get_score(src_ref, src_est_am)
            record_src(audio_path + 'am_', src_est_am, params['sample_rate'])

            # MISI
            src_est_misi = misi(mix_stft, spectro_mag, win_length=params['win_length'], hop_length=params['hop_length'],
                                max_iter=params['max_iter'])[0]
            sdr_misi[index_isnr, index_mix] = get_score(src_ref, src_est_misi)
            record_src(audio_path + 'misi_', src_est_misi, params['sample_rate'])

    # Save results
    np.savez(test_sdr_path, sdr_am=sdr_am, sdr_misi=sdr_misi)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)

    # Parameters
    params = {'sample_rate': 16000,
              'win_length': 1024,
              'hop_length': 256,
              'n_fft': 1024,
              'win_type': 'hann',
              'max_iter': 5,
              'n_mix': 50,
              'input_SNR_list': [10, 0, -10],
              }

    # Run the benchmark on the test set
    testing(params)

    # Plot the results
    #plot_test_results(params['input_SNR_list'], params['beta_range'])

# EOF

