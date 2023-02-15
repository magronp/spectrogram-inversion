#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from matplotlib import pyplot as plt


def plot_val_results(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'
    algos_list = params['algos_list']

    # Size Parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos = len(params['algos_list'])
    n_cons = params['cons_weight_list'].shape[0]

    # Load the validation SDR and average over mixtures
    loader = np.load(val_sdr_path)
    sdr_val, sdr_misi = np.nanmean(loader['sdr_val'], axis=2), np.nanmean(loader['sdr_misi'], axis=2)

    # MISI over iterations
    plt.figure(0)
    plt.plot(sdr_misi)
    for index_isnr in range(n_isnr):
        plt.subplot(1, n_isnr, index_isnr + 1)
        plt.plot(sdr_misi[:, index_isnr])
        if index_isnr == 0:
            plt.ylabel('SDR (dB)')
        plt.xlabel('Iterations')
        plt.title('iSNR= ' + str(params['input_SNR_list'][index_isnr]) + ' dB')
    plt.show()

    # Consistency-dependent algorithms over iterations
    plt.figure(1)
    for ia in range(n_algos):
        for index_isnr in range(n_isnr):
            plt.subplot(n_algos, n_isnr, ia*n_isnr+index_isnr+1)
            plt.plot(sdr_val[:, index_isnr, ia, :])
            if index_isnr == 0:
                plt.ylabel(algos_list[ia]),
            if ia == 0:
                plt.title('iSNR= ' + str(params['input_SNR_list'][index_isnr]) + ' dB')
    plt.legend(params['cons_weight_list'])
    plt.show()

    # Consistency-dependent algorithms over consistency weight
    #linestylelist = ['bo:', 'bo--', 'r+:', 'r+--', 'gx-']
    linestylelist = ['bo-', 'r+-', 'gx-']
    cons_weight_str = [r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"]
    sdr_val_opt_it = np.max(sdr_val, axis=0)
    plt.figure(2)
    for index_isnr in range(n_isnr):
        plt.subplot(1, n_isnr, index_isnr + 1)
        for ia in range(n_algos):
            plt.plot(sdr_val_opt_it[index_isnr, ia, :], linestylelist[ia])
        if index_isnr == 0:
            plt.ylabel('SDR (dB)', fontsize=16)
        plt.xlabel('Consistency weight', fontsize=16)
        plt.xticks(np.arange(0, n_cons, 1), cons_weight_str)
        plt.title('iSNR= ' + str(params['input_SNR_list'][index_isnr]) + ' dB', fontsize=16)
        plt.grid('on')
        if index_isnr == 1:
            plt.legend(algos_list, loc='center left')
    plt.show()
    plt.tight_layout()

    return


def plot_val_figures_article(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'
    algos_list = params['algos_list']

    # Size Parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos = len(params['algos_list'])
    n_cons = params['cons_weight_list'].shape[0]

    # Load the validation SDR and average over mixtures
    loader = np.load(val_sdr_path)
    sdr_val, sdr_misi = np.nanmean(loader['sdr_val'], axis=2), np.nanmean(loader['sdr_misi'], axis=2)

    # Consistency-dependent algorithms over consistency weight
    #linestylelist = ['bo:', 'bo--', 'r+:', 'r+--', 'gx-']
    linestylelist = ['bo-', 'r+-', 'gx-']
    cons_weight_str = [r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"]
    sdr_val_opt_it = np.max(sdr_val, axis=0)
    for index_isnr in range(n_isnr):
        #plt.subplot(1, n_isnr, index_isnr + 1)
        plt.figure(figsize=(4, 3))
        for ia in range(n_algos):
            plt.plot(sdr_val_opt_it[index_isnr, ia, :], linestylelist[ia])
        if index_isnr == 0:
            plt.ylabel('SDR (dB)', fontsize=16)
        plt.xlabel('Consistency weight', fontsize=16)
        plt.xticks(np.arange(0, n_cons, 1), cons_weight_str)
        plt.grid('on')
        plt.show()
        plt.tight_layout()

    # One plot over iterations
    plt.figure(figsize = (4, 3))
    sdr_val_opt_cons = np.max(sdr_val, axis=-1)
    index_isnr = 1
    plt.plot(sdr_misi[:, index_isnr], 'k-')
    for ia in range(n_algos):
        plt.plot(sdr_val_opt_cons[:, index_isnr, ia], linestylelist[ia])
    algos_list.insert(0, 'MISI')
    plt.legend(algos_list, fontsize=14, loc='lower center')
    plt.xlabel('Iterations', fontsize=16)
    plt.grid('on')
    plt.show()
    plt.tight_layout()

    return

# EOF
