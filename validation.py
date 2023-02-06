#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from helpers.data_io import load_src
from helpers.stft import my_stft
from open_unmx.estim_spectro import estim_spectro_from_mix
from helpers.plotter import plot_val_results
np.random.seed(0)  # Set random seed for reproducibility

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
    n_grad, n_beta = params['grad_step_range'].shape[0], params['beta_range'].shape[0]

    # Initialize the SDR array
    sdr_val = np.zeros((params['max_iter'] + 1, n_grad, n_beta, 2, 2, n_isnr, params['n_mix']))

    # Loop over iSNRs, mixtures and parameters
    for index_isnr, isnr in enumerate(params['input_SNR_list']):
        for index_mix in range(params['n_mix']):

            # Load time-domain signals and get the mixture's STFT
            audio_path = 'data/SNR_' + str(isnr) + '/' + str(index_mix) + '/'
            src_ref, mix = load_src(audio_path, params['sample_rate'])
            mix_stft = my_stft(mix, n_fft=params['n_fft'], hop_length=params['hop_length'],
                               win_length=params['win_length'], win_type=params['win_type'])[:, :, 0]

            # Estimate the magnitude spectrograms
            spectro_mag = estim_spectro_from_mix(mix)

            # Gradient descent
            for index_b, b in enumerate(params['beta_range']):
                for index_g, g in enumerate(params['grad_step_range']):
                    print('iSNR ' + str(index_isnr + 1) + ' / ' + str(n_isnr) +
                          ' -- Mix ' + str(index_mix + 1) + ' / ' + str(params['n_mix']) +
                          ' -- Beta ' + str(index_b + 1) + ' / ' + str(n_beta) +
                          ' -- Step size ' + str(index_g + 1) + ' / ' + str(n_grad))

                    # Run the gradient descent algorithm for d=1,2 and for the "right" and "left" problems
                    out = bregmisi_all(mix_stft, spectro_mag, src_ref=src_ref, win_length=params['win_length'],
                                       hop_length=params['hop_length'], win_type=params['win_type'], beta=b,
                                       grad_step=g * np.ones((2, 2)), max_iter=params['max_iter'])

                    # Store the SDR over iterations
                    sdr_val[:, index_g, index_b, 0, 0, index_isnr, index_mix] = out['sdr_1r']
                    sdr_val[:, index_g, index_b, 1, 0, index_isnr, index_mix] = out['sdr_2r']
                    sdr_val[:, index_g, index_b, 0, 1, index_isnr, index_mix] = out['sdr_1l']
                    sdr_val[:, index_g, index_b, 1, 1, index_isnr, index_mix] = out['sdr_2l']

    # Save results
    np.savez(val_sdr_path, sdr=sdr_val)

    return


def get_opt_gd_step(grad_step_range, val_sdr_path='outputs/val_sdr.npz'):
    """ Compute the optimal step size from the validation set SDR results
    Args:
        grad_step_range: numpy array - the step size grid
        val_sdr_path: string - the path where to load the validation SDR
    """

    # Load the validation SDR and average over mixtures
    sdr = np.load(val_sdr_path)['sdr']
    sdr_av = np.nanmean(sdr, axis=-1)

    # Remove Nans and values below 0 for better readability
    sdr_av[np.isnan(sdr_av)] = 0
    sdr_av[sdr_av < 0] = 0

    # Get the optimal step size
    gd_step_opt = grad_step_range[np.argmax(sdr_av[-1, :], axis=0)]
    np.savez('outputs/val_gd_step.npz', gd_step=gd_step_opt)

    return


if __name__ == '__main__':
    # Parameters
    params = {'sample_rate': 16000,
              'win_length': 1024,
              'hop_length': 256,
              'n_fft': 1024,
              'win_type': 'hann',
              'max_iter': 5,
              'n_mix': 50,
              'input_SNR_list': [10, 0, -10],
              'grad_step_range': np.logspace(-7, 1, 9),
              'beta_range': np.linspace(0, 2, 9)
              }

    # Run the validation
    validation(params)

    # Get the optimal step size
    get_opt_gd_step(params['grad_step_range'])

    # Plot the results
    plot_val_results(params['input_SNR_list'], index_left_right=0)

# EOF
