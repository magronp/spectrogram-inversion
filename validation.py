#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from helpers.data_io import load_src
from librosa import stft, istft
from helpers.algos import spectrogram_inversion, amplitude_mask
from open_unmx.estim_spectro import estim_spectro_from_mix
from helpers.plotter import plot_val_results
from matplotlib import pyplot as plt


__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def validation(params, val_sdr_path='outputs/val_sdr.npz'):
    """ Run the proposed algorithm on the validation subset in different settings
    Args:
        params: dictionary with fields:
            'sample rate': int - the sampling frequency
            'n_mix': int - the number of mixtures to process
            'max_iter': int - the nomber of iterations of the proposed algorithm
            'input_SNR_list': list - the list of input SNRs to consider
            'grad_step_range': numpy array - the step size grid
            'beta_range': numpy array - the beta-divergence parameter grid
            'hop_length': int - the hop size of the STFT
            'win_length': int - the window length
            'n_fft': int - the number of FFT points
            'win_type': string - the STFT window type (e.g., Hann, Hamming, Blackman...)
        val_sdr_path: string - the path where to store the validation SDR
    """

    # Some parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos, n_cons = len(params['algos_list']), params['cons_weight_list'].shape[0]

    # Initialize the SDR array
    sdr_val = np.zeros((params['max_iter'] + 1, n_isnr, params['n_mix'], n_algos, n_cons))
    sdr_misi = np.zeros((params['max_iter'] + 1, n_isnr, params['n_mix']))
    error_val = np.zeros((params['max_iter'], n_isnr, params['n_mix'], n_algos, n_cons))
    error_misi= np.zeros((params['max_iter'], n_isnr, params['n_mix']))

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

            # MISI
            _, error, sdr = spectrogram_inversion(mix_stft, spectro_mag, params['win_length'], algo='MISI',
                                                  max_iter=params['max_iter'], reference_sources=src_ref,
                                                  hop_length=params['hop_length'], window=params['win_type'],
                                                  compute_error=True)
            sdr_misi[:, index_isnr, index_mix] = sdr
            error_misi[:, index_isnr, index_mix] = error

            # Validation for all algos depending on a consistency weight
            for ia, algo in enumerate(params['algos_list']):
                for ic, consistency_weight in enumerate(params['cons_weight_list']):

                    print('iSNR ' + str(index_isnr + 1) + ' / ' + str(n_isnr) +
                          ' -- Mix ' + str(index_mix + 1) + ' / ' + str(params['n_mix']) +
                          ' -- Algo ' + str(ia + 1) + ' / ' + str(n_algos) +
                          ' -- Cons weight ' + str(ic + 1) + ' / ' + str(n_cons))
                    
                    src_est, error, sdr = spectrogram_inversion(mix_stft, spectro_mag, params['win_length'],
                                                                algo=algo,
                                                                consistency_weigth=consistency_weight,
                                                                max_iter=params['max_iter'],
                                                                reference_sources=src_ref,
                                                                hop_length=params['hop_length'],
                                                                window=params['win_type'], compute_error=True)
                    sdr_val[:, index_isnr, index_mix, ia, ic] = sdr
                    error_val[:, index_isnr, index_mix, ia, ic] = error

    # Save results
    np.savez(val_sdr_path, sdr_val=sdr_val, sdr_misi=sdr_misi, error_val=error_val, error_misi=error_misi)

    return


def plot_val(params, val_sdr_path='outputs/val_sdr.npz'):
    """ Compute the optimal step size from the validation set SDR results
    Args:
        grad_step_range: numpy array - the step size grid
        val_sdr_path: string - the path where to load the validation SDR
    """

    # Load the validation SDR and average over mixtures
    loader = np.load(val_sdr_path)
    sdr_val, sdr_misi = np.nanmean(loader['sdr_val'], axis=2), np.nanmean(loader['sdr_misi'], axis=2)

    plt.figure(0)
    plt.plot(sdr_misi)
    for index_isnr in range(3):
        plt.subplot(1, 3, index_isnr + 1)
        plt.plot(sdr_misi[:, index_isnr])
        plt.xlabel('Iterations'), plt.ylabel('SDR (dB)')
        plt.title('MISI')

    plt.figure(1)
    for ia in range(3):
        for index_isnr in range(3):
            plt.subplot(3, 3, ia*3+index_isnr+1)
            plt.plot(sdr_val[:, index_isnr, ia, :])
            if index_isnr==0: plt.ylabel(params['algos_list'][ia]),
            if ia==0: plt.title('iSNR= ' + str(params['input_SNR_list'][index_isnr]) + ' dB')
    plt.legend(params['cons_weight_list'])

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
              'max_iter': 20,
              'n_mix': 50,
              'input_SNR_list': [10, 0, -10],
              'cons_weight_list': np.insert(np.logspace(-3, 3, 7), 0, 0),
              'algos_list': ['Mix+Incons', 'Mix+Incons_hardMag', 'Mag+Incons_hardMix']
              }

    # Run the validation
    validation(params)
    plot_val(params['grad_step_range'])

# EOF
