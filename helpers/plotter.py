#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from matplotlib import pyplot as plt


def export_legend(legend, filename="legend.eps"):
    fig  = legend.figure
    fig.canvas.draw()
    bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    return


def plot_val_results(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'

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
                plt.ylabel(params['algos_list'][ia]),
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
            plt.legend(params['algos_list'], loc='center left')
    plt.show()
    plt.tight_layout()

    return


def plot_val_figures_article(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'

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
        plt.figure(figsize=(4, 3))
        for ia in range(n_algos):
            plt.plot(sdr_val_opt_it[index_isnr, ia, :], linestylelist[ia])
        if index_isnr == 0:
            plt.ylabel('SDR (dB)', fontsize=16)
        plt.xlabel('Consistency weight', fontsize=16)
        plt.xticks(np.arange(0, n_cons, 1), cons_weight_str)
        plt.grid('on')
        plt.tight_layout()
        plt.show()
        
    # One plot over iterations
    fig = plt.figure(figsize = (4, 3))
    sdr_val_opt_cons = np.max(sdr_val, axis=-1)
    index_isnr = 1
    for ia in range(n_algos):
        plt.plot(sdr_val_opt_cons[:, index_isnr, ia], linestylelist[ia], label=params['algos_list'][ia])
    plt.plot(sdr_misi[:, index_isnr], 'k-', label='MISI')
    plt.xlabel('Iterations', fontsize=16)
    plt.grid('on')
    plt.tight_layout()
    plt.show()

    leg = fig.legend(loc="lower left", ncol=4)
    export_legend(leg)

    return


def plot_val_figures_article_singleplot(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'

    # Size Parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos_cons = len(params['algos_list'])
    n_cons = params['cons_weight_list'].shape[0]

    # Load the validation SDR and average over mixtures
    loader = np.load(val_sdr_path)
    sdr_val, sdr_misi = np.nanmean(loader['sdr_val'], axis=2), np.nanmean(loader['sdr_misi'], axis=2)

    # Consistency-dependent algorithms over consistency weight
    #linestylelist = ['bo:', 'bo--', 'r+:', 'r+--', 'gx-']
    linestylelist = ['bo-', 'r+-', 'gx-']
    cons_weight_str = [r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^{0}$", r"$10^{1}$", r"$10^{2}$", r"$10^{3}$"]
    sdr_val_opt_it = np.max(sdr_val, axis=0)

    fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(3.5 * 4, 3.5))

    for index_isnr in range(n_isnr):
        for ia in range(n_algos_cons):
            ax[index_isnr].plot(sdr_val_opt_it[index_isnr, ia, :], linestylelist[ia])
        if index_isnr == 0:
            ax[index_isnr].set_ylabel('SDR (dB)', fontsize=16)
        ax[index_isnr].set_xlabel('Consistency weight', fontsize=16)
        ax[index_isnr].set_xticks(np.arange(0, n_cons, 1), cons_weight_str)
        ax[index_isnr].grid('on')

    # One plot over iterations (includes MISI)
    sdr_val_opt_cons = np.max(sdr_val, axis=-1)
    index_isnr = 1
    for ia in range(n_algos_cons):
        ax[n_isnr].plot(sdr_val_opt_cons[:, index_isnr, ia], linestylelist[ia], label=params['algos_list'][ia])
    ax[n_isnr].plot(sdr_misi[:, index_isnr], 'k-', label='MISI')
    ax[n_isnr].set_xlabel('Iterations', fontsize=16)
    ax[n_isnr].grid('on')

    bb = (fig.subplotpars.left, fig.subplotpars.top+0.005, fig.subplotpars.right-fig.subplotpars.left,.1)
    leg = fig.legend(bbox_to_anchor=bb, loc="lower left", ncol=4)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    export_legend(leg)

    return

# EOF
