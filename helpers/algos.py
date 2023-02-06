#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from librosa import stft, istft

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def get_score(src_ref, src_est, eps=1e-8):
    """Calculate Signal-to-Distortion Ratio (SDR) in a speech enhancement framework
    Args:
        src_ref: numpy.ndarray (nsrc, nsamples) - ground truth sources
        src_est: numpy.ndarray (nsrc, nsamples) - estimated sources
    Returns:
        score: float
    """

    # Expected inputs of size (nsrc x nsamples)
    if src_ref.shape[0] > src_ref.shape[1]:
        src_ref = src_ref.T
        src_est = src_est.T

    # Get the SDR only for the first (=speech) source
    sdr = 10 * np.log10(np.sum(src_ref[0, :] ** 2) / (np.sum((src_ref[0, :] - src_est[0, :]) ** 2) + eps))

    return sdr


def amplitude_mask(spectro_mag, mix_stft, win_length=None, hop_length=None, window='hann'):
    """ Return the amplitude mask (assign the mixture's phase to each source)
    Args:
        spectro_mag: numpy.ndarray (nfreqs, nframes, nsrc) - magnitude spectrograms
        mix_stft: numpy.ndarray (nfreqs, nframes) - mixture's STFT
        hop_length: int - the hop size of the STFT
        win_length: int - the window length
        window: string - the STFT window type (e.g., Hann, Hamming, Blackman...)
    Returns:
        src_est: numpy.ndarray (nsamples, nsrc) - time-domain signals estimated with the amplitude mask
    """

    # Parameters
    n_src = spectro_mag.shape[0]
    if win_length is None: win_length = (spectro_mag.shape[1]-1)*2
    if hop_length is None: hop_length = win_length // 2

    # Masking
    stft_est = spectro_mag * np.repeat(np.exp(1j * np.angle(mix_stft))[np.newaxis, :, :], n_src, axis=0)

    # inverse STFT
    src_est = istft(stft_est, hop_length=hop_length, win_length=win_length, window=window)

    return src_est


def misi(mix_stft, spectro_mag, win_length=None, hop_length=None, src_ref=None, max_iter=20, window=None):
    """The multiple input spectrogram inversion algorithm for source separation.
    Args:
        mix_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectro_mag: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude spectrograms
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        max_iter: int - number of iterations
        window: string - window type
    Returns:
        estimated_sources: numpy.ndarray (nsamples, nrsc) - the time-domain estimated sources
        error: list (max_iter) - loss function (magnitude mismatch) over iterations
        sdr: list (max_iter) - score (SDR in dB) over iterations
    """

    # Parameters
    n_src = spectro_mag.shape[0]
    n_fft = (spectro_mag.shape[1] - 1) * 2
    if win_length is None: win_length = n_fft
    if hop_length is None: hop_length = win_length // 2
    if window is None: window = 'hann'

    # Pre allocate SDR and error
    compute_sdr = not (src_ref is None)
    error, sdr = [], []

    # Initialization with amplitude mask
    src_est = amplitude_mask(spectro_mag, mix_stft, win_length=win_length, hop_length=hop_length, window=window)

    if compute_sdr:
        sdr.append(get_score(src_ref, src_est))

    for iteration_number in range(max_iter):
        # STFT
        stft_est = stft(src_est, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        current_magnitude = np.abs(stft_est)
        # Normalize to the target amplitude
        stft_est = stft_est * spectro_mag / (np.abs(stft_est) + sys.float_info.epsilon)
        # Compute and distribute the mixing error
        mixing_error = mix_stft - np.sum(stft_est, axis=0)
        stft_est += np.repeat(mixing_error[np.newaxis, :, :], n_src, axis=0) / n_src
        # Inverse STFT
        src_est = istft(stft_est, win_length=win_length, hop_length=hop_length, window=window)
        # BSS score
        if compute_sdr:
            sdr.append(get_score(src_ref, src_est))
        # Error
        error.append(np.linalg.norm(current_magnitude - spectro_mag))

    return src_est, error, sdr



# The three projectors
def p_cons(sources_stft, win_length=None, hop_length=None, window=None):
    """Projector on the consistent matrices' subspace, which consists of an inverse STFT followed by an STFT
    """
    n_fft = (sources_stft.shape[1] - 1) * 2
    if win_length is None: win_length = n_fft
    if hop_length is None: hop_length = win_length // 2
    if window is None: window = 'hann'

    src_est = istft(sources_stft, win_length=win_length, hop_length=hop_length, window=window)
    update_sources_stft = stft(src_est, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    return update_sources_stft


def p_mix(sources_stft, mixture_stft, mixing_weights=None):
    """Projector on the conservative subspace, which consists in calculating the mixing error and distributing
    it over the components
    """
    nsrc = sources_stft.shape[0]
    if mixing_weights is None:
        mixing_weights = 1 / nsrc

    mixing_error = mixture_stft - np.sum(sources_stft, axis=0)
    update_sources_stft = sources_stft + np.repeat(mixing_error[np.newaxis, :, :], nsrc, axis=0) * mixing_weights

    return update_sources_stft


def p_mag(sources_stft, target_magnitudes):
    """Projector on the target magnitudes subspace: the magnitude of the sources is set at the target values
    """
    update_sources_stft = sources_stft * target_magnitudes / (np.abs(sources_stft) + sys.float_info.epsilon)
    return update_sources_stft


def spectrogram_inversion_update(sources_stft, mixture_stft, target_magnitudes, win_length, algo='MISI',
                                 consistency_weigth=1, mixing_weights=None, hop_length=None, window=None,
                                 compute_error=False):

    nsrc = sources_stft.shape[0]
    n_fft = (sources_stft.shape[1] - 1) * 2
    if win_length is None: win_length = n_fft
    if hop_length is None: hop_length = win_length // 2
    if window is None: window = 'hann'

    error = []
    if algo == 'MISI':
        aux_cons = p_cons(sources_stft, win_length, hop_length=hop_length, window=window)
        update_sources_stft = p_mag(aux_cons, target_magnitudes)
        mixing_weights = 1 / nsrc
        update_sources_stft = p_mix(update_sources_stft, mixture_stft, mixing_weights)
        if compute_error:
            error = np.linalg.norm(np.abs(aux_cons) - target_magnitudes)

    elif algo == 'Incons_hardMix':
        mixing_weights = 1 / nsrc
        update_sources_stft = p_cons(sources_stft, win_length, hop_length=hop_length, window=window)
        update_sources_stft = p_mix(update_sources_stft, mixture_stft, mixing_weights)

    elif algo == 'Mix+Incons_hardMag':
        mixing_weights = target_magnitudes / (
                np.repeat(np.sum(target_magnitudes, axis=0)[np.newaxis, :, :], nsrc, axis=0) + sys.float_info.epsilon)
        aux_cons = p_cons(sources_stft, win_length, hop_length=hop_length, window=window)
        aux_mix = p_mix(sources_stft, mixture_stft, mixing_weights)
        update_sources_stft = p_mag(aux_mix + consistency_weigth * mixing_weights * aux_cons, target_magnitudes)
        if compute_error:
            error = np.linalg.norm(mixture_stft - np.sum(sources_stft, axis=0)) +\
                    consistency_weigth * np.linalg.norm(aux_cons - sources_stft)

    elif algo == 'Mix+Incons':
        mixing_weights = target_magnitudes / (
                np.repeat(np.sum(target_magnitudes, axis=0)[np.newaxis, :, :], nsrc, axis=0) + sys.float_info.epsilon)
        aux_cons = p_cons(sources_stft, win_length, hop_length=hop_length, window=window)
        aux_mix = p_mix(sources_stft, mixture_stft, mixing_weights)
        update_sources_stft = (aux_mix + consistency_weigth * mixing_weights * aux_cons) /\
                              (1 + consistency_weigth * mixing_weights)
        if compute_error:
            error = np.linalg.norm(mixture_stft - np.sum(sources_stft, axis=0)) +\
                    consistency_weigth * np.linalg.norm(aux_cons - sources_stft)

    elif algo == 'Mag+Incons_hardMix':
        mixing_weights = 1 / nsrc
        aux_cons = p_cons(sources_stft, win_length, hop_length=hop_length, window=window)
        aux_mag = p_mag(sources_stft, target_magnitudes)
        update_sources_stft = p_mix((aux_mag + consistency_weigth * aux_cons) / (1 + consistency_weigth), mixture_stft)
        if compute_error:
            error = np.linalg.norm(np.abs(aux_cons) - target_magnitudes) +\
                    consistency_weigth * np.linalg.norm(aux_cons - sources_stft)

    else:
        raise ValueError('Unknown algorithm')

    return update_sources_stft, error


def spectrogram_inversion(mixture_stft, target_magnitudes, win_length, algo='MISI', consistency_weigth=None,
                          max_iter=5, reference_sources=None, hop_length=None, window='hann', compute_error=False):

    if hop_length is None:
        hop_length = win_length // 2

    compute_sdr = True
    if reference_sources is None:
        compute_sdr = False

    if algo=='Incons_hardMix':
        max_iter = 1

    # Parameters
    nsrc = target_magnitudes.shape[0]
    mixing_weights = target_magnitudes / (
                np.repeat(np.sum(target_magnitudes, axis=0)[np.newaxis, :, :], nsrc, axis=0) + sys.float_info.epsilon)

    # Initial STFT estimation (amplitude mask)
    update_sources_stft = target_magnitudes * np.exp(1j * np.repeat(np.angle(mixture_stft)[np.newaxis, :, :], nsrc, axis=0))

    # Pre allocate BSS score and Initial value if needed
    error, sdr = [], []
    if compute_sdr:
        estimated_sources = istft(update_sources_stft, win_length=win_length, hop_length=hop_length, window=window)
        sdr.append(get_score(reference_sources, estimated_sources))

    if not(algo == 'AM'):
        # Iterations
        for iteration_number in range(max_iter):

            # Update
            update_sources_stft, update_error = spectrogram_inversion_update(update_sources_stft, mixture_stft, target_magnitudes, win_length, algo=algo,
                                         mixing_weights=mixing_weights, consistency_weigth=consistency_weigth, hop_length=hop_length, window=window, compute_error=compute_error)
            error.append(update_error)

            # If the score needs to be computed
            if compute_sdr:
                estimated_sources = istft(update_sources_stft, win_length=win_length, hop_length=hop_length, window=window)
                sdr.append(get_score(reference_sources, estimated_sources))

    # After the iterative loop, inverse STFT if it hasn't been done
    if not(compute_sdr):
        estimated_sources = istft(update_sources_stft, win_length=win_length, hop_length=hop_length, window=window)

    return estimated_sources, error, sdr

# EOF
