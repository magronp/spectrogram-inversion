#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
from helpers.algos import get_score, amplitude_mask, spectrogram_inversion
from helpers.data_io import load_src
from librosa import stft
from helpers.openunmix import estim_spectro_from_mix
import pickle


def testing(params, out_dir='outputs/'):

    # paths
    val_opt_path = out_dir + 'val_opt_'
    test_sdr_path = out_dir + 'test_sdr.npz'

    # Size Parameters
    n_isnr = len(params['input_SNR_list'])
    n_algos = len(params['algos_list'])

    # Initialize the SDR array
    sdr_test = np.zeros((n_algos+1, n_isnr, params['n_mix']))

    # Load the optimal consistency weight and numbers of iterations from validation
    with open(val_opt_path + 'iter.pkl', 'rb') as f:
        iter_opt = pickle.load(f)
    with open(val_opt_path + 'cons.pkl', 'rb') as f:
        cons_opt = pickle.load(f)

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

            # Amplitude mask
            src_est_am = amplitude_mask(spectro_mag, mix_stft, win_length=params['win_length'],
                                        hop_length=params['hop_length'], window=params['win_type'])
            sdr_test[0, index_isnr, index_mix] = get_score(src_ref, src_est_am)

            # Validation for all algos depending on a consistency weight
            for ia, algo in enumerate(params['algos_list']):
                print('iSNR ' + str(index_isnr + 1) + ' / ' + str(n_isnr) +
                      ' -- Mix ' + str(index_mix + 1) + ' / ' + str(params['n_mix']) +
                      ' -- Algo ' + str(ia + 1) + ' / ' + str(n_algos))

                # Get the optimal number of iterations and consistency weight for the current algorithm
                max_iter_opt, cons_weight = int(iter_opt[algo][index_isnr]), cons_opt[algo][index_isnr]

                # Apply the algorithm and collect the score
                src_est = spectrogram_inversion(mix_stft, spectro_mag, algo=algo, consistency_weigth=cons_weight,
                                                max_iter=max_iter_opt,  win_length=params['win_length'],
                                                hop_length=params['hop_length'], window=params['win_type'])[0]
                sdr_test[ia+1, index_isnr, index_mix] = get_score(src_ref, src_est)

    # Save results
    np.savez(test_sdr_path, sdr_test=sdr_test)

    return


def display_test_results(params, out_dir='outputs/'):

    algos_list = params['algos_list']
    algos_list.insert(0, 'AM')
    test_sdr_path = out_dir + 'test_sdr.npz'
    sdr_test = np.median(np.load(test_sdr_path)['sdr_test'], axis=-1)
    for index_isnr, isnr in enumerate(params['input_SNR_list']):
        print('------ iSNR =' + str(isnr) + ' dB')
        for ia, algo in enumerate(params['algos_list']):
            print(algo + ' --  {:.1f}'.format(sdr_test[ia, index_isnr]))

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
              'n_mix': 50,
              'input_SNR_list': [10, 0, -10],
              'algos_list': ['MISI', 'Mix+Incons', 'Mix+Incons_hardMag', 'Incons_hardMix', 'Mag+Incons_hardMix']
              }

    # Run the benchmark on the test set
    testing(params, out_dir)
    display_test_results(params, out_dir)

# EOF
