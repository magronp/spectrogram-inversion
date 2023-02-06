#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def plot_val_results(input_snr_list, index_left_right=0, val_sdr_path='outputs/val_sdr.npz'):
    """ Plot the results on the validation set
    Args:
        input_snr_list: list - the list of input SNRs
        index_left_right: int - corresponds to the direction of the problem: "right" (0) or "left" (1)
        val_sdr_path: string - the path where to load the validation SDR
    """

    # Load the validation SDR and average over mixtures
    sdr = np.load(val_sdr_path)['sdr']
    sdr_av = np.nanmean(sdr, axis=-1)

    # Get the improvement over initialization
    sdri = sdr_av[-1, :] - sdr_av[0, :]

    # Remove Nans and values below 0 for better visibility
    sdri[np.isnan(sdri)] = 0
    sdri[sdri < 0] = 0

    # Select the config to plot
    sdr_plot = sdri[:, :, :, index_left_right, :]

    # Plot results
    plt.figure(0)
    n_isnr = len(input_snr_list)
    my_extent = [0, 2, -7, 1]

    # First subplot for d=1
    for index_isnr in range(n_isnr):
        plt.subplot(2, n_isnr, index_isnr + 1)
        plt.imshow(sdr_plot[:, :, 0, index_isnr], aspect='auto', origin='lower', extent=my_extent)
        if index_isnr == 0:
            plt.ylabel('Step size (log)', fontsize=14)
        else:
            plt.yticks([])
        plt.xticks([])
        plt.colorbar()
        plt.title('iSNR = ' + str(input_snr_list[index_isnr]) + ' dB', fontsize=14)

    # Second subplot for d=2
    for index_isnr in range(n_isnr):
        plt.subplot(2, n_isnr, index_isnr + 1 + n_isnr)
        plt.imshow(sdr_plot[:, :, 1, index_isnr], aspect='auto', origin='lower', extent=my_extent)
        if index_isnr == 0:
            plt.ylabel('Step size (log)', fontsize=14)
        else:
            plt.yticks([])
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.colorbar()
    plt.show()
    plt.tight_layout()

    return


def plot_test_results(input_snr_list, beta_range, test_sdr_path='outputs/test_sdr.npz'):
    """ Plot the results on the test set
    Args:
        input_snr_list: list - the list of input SNRs
        beta_range: list - the range for the values of beta
        test_sdr_path: string - the path where to load the test SDR
    """
    # Load the data
    data = np.load(test_sdr_path)
    sdr_am, sdr_misi, sdr_gd = data['sdr_am'], data['sdr_misi'], data['sdr_gd']

    # Get the SDR improvement over amplitude mask
    sdr_misi -= sdr_am
    sdr_gd -= sdr_am

    # Mean SDR
    sdr_misi_av = np.nanmean(sdr_misi, axis=-1)
    sdr_gd_av = np.nanmean(sdr_gd, axis=-1)

    # Plot results
    n_isnr = len(input_snr_list)
    plt.figure(0)
    for index_isnr in range(n_isnr):
        plt.subplot(1, n_isnr, index_isnr+1)
        plt.plot(beta_range[1:], sdr_gd_av[1:, 0, 0, index_isnr], color='b', marker='x')
        plt.plot(beta_range[1:], sdr_gd_av[1:, 1, 0, index_isnr], color='b', marker='o')
        plt.plot(beta_range[1:], sdr_gd_av[1:, 0, 1, index_isnr], color='r', marker='x')
        plt.plot(beta_range[1:], sdr_gd_av[1:, 1, 1, index_isnr], color='r', marker='o')
        plt.plot([0, 2], [sdr_misi_av[index_isnr], sdr_misi_av[index_isnr]], linestyle='--', color='k')
        plt.xlabel(r'$\beta$', fontsize=14)
        plt.ylim(0.25, 1.25)
        if index_isnr == 0:
            plt.ylabel('SDRi (dB)', fontsize=14)
        plt.title('iSNR = ' + str(input_snr_list[index_isnr]) + ' dB', fontsize=14)
    plt.legend(['right, d=1', 'right, d=2', 'left, d=1', 'left, d=2', 'MISI'])
    plt.tight_layout()
    plt.show()

    return


def plot_test_results_pernoise(input_snr_list, beta_range, test_sdr_path='outputs/test_sdr.npz'):
    """ Plot the results on the test set for each noise type
    Args:
        input_snr_list: list - the list of input SNRs
        beta_range: list - the range for the values of beta
        test_sdr_path: string - the path where to load the test SDR
    """

    # Load the data
    data = np.load(test_sdr_path)
    sdr_am, sdr_misi, sdr_gd = data['sdr_am'], data['sdr_misi'], data['sdr_gd']

    # Get the improvement over Amplitude mask and keep the good GD setting
    sdr_misi -= sdr_am
    sdr_gd -= sdr_am

    for noise_type in range(3):
        # Load the noise type list of indices
        ind_noise = np.load('data/noise_ind.npz')['ind_noise_' + str(noise_type+1)]

        # Keep only mixtures corresponding to this noise
        sdr_misi = sdr_misi[:, ind_noise]
        sdr_gd = sdr_gd[:, :, :, :, ind_noise]

        # Mean and standard deviation of the SDR
        sdr_misi_av = np.nanmean(sdr_misi, axis=-1)
        sdr_gd_av = np.nanmean(sdr_gd, axis=-1)

        # Plot results
        n_isnr = len(input_snr_list)
        plt.figure(noise_type)
        for index_isnr in range(n_isnr):
            plt.subplot(1, n_isnr, index_isnr+1)
            plt.plot(beta_range[1:], sdr_gd_av[1:, 0, 0, index_isnr], color='b', marker='x')
            plt.plot(beta_range[1:], sdr_gd_av[1:, 1, 0, index_isnr], color='b', marker='o')
            plt.plot(beta_range[1:], sdr_gd_av[1:, 0, 1, index_isnr], color='r', marker='x')
            plt.plot(beta_range[1:], sdr_gd_av[1:, 1, 1, index_isnr], color='r', marker='o')
            plt.plot([0, 2], [sdr_misi_av[index_isnr], sdr_misi_av[index_isnr]], linestyle='--', color='k')
            plt.xlabel(r'$\beta$')
            if index_isnr == 0:
                plt.ylabel('SDRi (dB)')
            plt.title('iSNR = ' + str(input_snr_list[index_isnr]) + ' dB')
        plt.legend(['right, d=1', 'right, d=2', 'left, d=1', 'left, d=2', 'MISI'])

    return

# EOF
