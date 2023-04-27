#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.data_io import load_src
from librosa import stft
from helpers.algos import spectrogram_inversion
from helpers.openunmix import estim_spectro_from_mix
import pickle
from helpers.plotter import plot_val_figures_article, plot_val_results


def validation(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'

    # Size Parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos = len(params['algos_list'])
    n_cons = params['cons_weight_list'].shape[0]

    # Initialize the SDR/error arrays
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
            spectro_mag = estim_spectro_from_mix(mix)

            # MISI
            _, error, sdr = spectrogram_inversion(mix_stft, spectro_mag, algo='MISI', max_iter=params['max_iter'],
                                                  reference_sources=src_ref, win_length=params['win_length'],
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
                    
                    src_est, error, sdr =\
                        spectrogram_inversion(mix_stft, spectro_mag, algo=algo, consistency_weigth=consistency_weight,
                                              max_iter=params['max_iter'], reference_sources=src_ref,
                                              win_length=params['win_length'], hop_length=params['hop_length'],
                                              window=params['win_type'], compute_error=True)

                    sdr_val[:, index_isnr, index_mix, ia, ic] = sdr
                    error_val[:, index_isnr, index_mix, ia, ic] = error

    # Save results
    np.savez(val_sdr_path, sdr_val=sdr_val, sdr_misi=sdr_misi, error_val=error_val, error_misi=error_misi)

    return


def get_opt_val(params, out_dir='outputs/'):

    val_sdr_path = out_dir + 'val_sdr.npz'
    val_opt_path = out_dir + 'val_opt_'

    # Size Parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos = len(params['algos_list'])
    n_cons = params['cons_weight_list'].shape[0]

    # Load the validation SDR and average over mixtures
    loader = np.load(val_sdr_path)
    sdr_val, sdr_misi = np.nanmean(loader['sdr_val'], axis=2), np.nanmean(loader['sdr_misi'], axis=2)

    # Consistency-dependent algorithms - get optimal number of iterations and consistency weight
    dict_iter_opt = {}.fromkeys(params['algos_list'], np.zeros(n_isnr))
    dict_cons_opt = {}.fromkeys(params['algos_list'], np.zeros(n_isnr))

    my_shape = (params['max_iter']+1, n_cons)
    for ia in range(n_algos):
        for index_isnr in range(n_isnr):
            max_it, cons_opt_index = np.unravel_index(np.argmax(sdr_val[:, index_isnr, ia, :]), my_shape)
            dict_iter_opt[params['algos_list'][ia]][index_isnr] = max_it
            dict_cons_opt[params['algos_list'][ia]][index_isnr] = params['cons_weight_list'][cons_opt_index]

    # MISI - get optimal number of iterations
    dict_iter_opt['MISI'] = np.argmax(sdr_misi, axis=0)
    dict_cons_opt['MISI'] = np.zeros(n_isnr)

    # Add the info for the remaining algorithm for generality
    dict_iter_opt['Incons_hardMix'] = np.ones(n_isnr)
    dict_cons_opt['Incons_hardMix'] = np.zeros(n_isnr)

    # Save results
    with open(val_opt_path + 'iter.pkl', 'wb') as f:
        pickle.dump(dict_iter_opt, f)
    with open(val_opt_path + 'cons.pkl', 'wb') as f:
        pickle.dump(dict_cons_opt, f)

    return


if __name__ == '__main__':

    # Set random seed for reproducibility
    np.random.seed(1234)

    # Parameters
    out_dir = 'outputs/'
    params = {'sample_rate': 16000,
              'win_length': 1024,
              'hop_length': 256,
              'n_fft': 1024,
              'win_type': 'hann',
              'max_iter': 20,
              'n_mix': 50,
              'input_SNR_list': [10, 0, -10],
              'cons_weight_list': np.insert(np.logspace(-3, 3, 7), 0, 0),
              'algos_list': ['Mix+Incons', 'Mix+Incons_hardMag', 'Mag+Incons_hardMix'],
              }

    # Run the validation
    validation(params, out_dir)
    get_opt_val(params, out_dir)
    plot_val_results(params, out_dir)

    # A function to specifically produce the results from the paper
    #plot_val_figures_article(params, out_dir)

# EOF
