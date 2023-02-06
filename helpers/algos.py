#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from helpers.stft import my_istft, my_stft

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


def amplitude_mask(spectro_mag, mix_stft, win_length=None, hop_length=None, win_type='hann'):
    """ Return the amplitude mask (assign the mixture's phase to each source)
    Args:
        spectro_mag: numpy.ndarray (nfreqs, nframes, nsrc) - magnitude spectrograms
        mix_stft: numpy.ndarray (nfreqs, nframes) - mixture's STFT
        hop_length: int - the hop size of the STFT
        win_length: int - the window length
        win_type: string - the STFT window type (e.g., Hann, Hamming, Blackman...)
    Returns:
        src_est: numpy.ndarray (nsamples, nsrc) - time-domain signals estimated with the amplitude mask
    """

    # Parameters
    n_src = spectro_mag.shape[2]
    if win_length is None: win_length = (spectro_mag.shape[0]-1)*2
    if hop_length is None: hop_length = win_length // 2

    # Masking
    stft_est = spectro_mag * np.repeat(np.exp(1j * np.angle(mix_stft))[:, :, np.newaxis], n_src, axis=2)

    # inverse STFT
    src_est = my_istft(stft_est, hop_length=hop_length, win_length=win_length, win_type=win_type)

    return src_est


def misi(mix_stft, spectro_mag, win_length=None, hop_length=None, src_ref=None, max_iter=20, win_type='hann'):
    """The multiple input spectrogram inversion algorithm for source separation.
    Args:
        mix_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectro_mag: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude spectrograms
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        max_iter: int - number of iterations
        win_type: string - window type
    Returns:
        estimated_sources: numpy.ndarray (nsamples, nrsc) - the time-domain estimated sources
        error: list (max_iter) - loss function (magnitude mismatch) over iterations
        sdr: list (max_iter) - score (SDR in dB) over iterations
    """

    # Parameters
    n_src = spectro_mag.shape[2]
    n_fft = (spectro_mag.shape[0]-1)*2
    if win_length is None: win_length = n_fft
    if hop_length is None: hop_length = win_length // 2

    # Pre allocate SDR and error
    compute_sdr = not (src_ref is None)
    error, sdr = [], []

    # Initialization with amplitude mask
    src_est = amplitude_mask(spectro_mag, mix_stft, win_length=win_length, hop_length=hop_length, win_type=win_type)

    if compute_sdr:
        sdr.append(get_score(src_ref, src_est))

    for iteration_number in range(max_iter):
        # STFT
        stft_est = my_stft(src_est, n_fft=n_fft, hop_length=hop_length, win_length=win_length, win_type=win_type)
        current_magnitude = np.abs(stft_est)
        # Normalize to the target amplitude
        stft_est = stft_est * spectro_mag / (np.abs(stft_est) + sys.float_info.epsilon)
        # Compute and distribute the mixing error
        mixing_error = mix_stft - np.sum(stft_est, axis=2)
        stft_est += np.repeat(mixing_error[:, :, np.newaxis], n_src, axis=2) / n_src
        # Inverse STFT
        src_est = my_istft(stft_est, win_length=win_length, hop_length=hop_length, win_type=win_type)
        # BSS score
        if compute_sdr:
            sdr.append(get_score(src_ref, src_est))
        # Error
        error.append(np.linalg.norm(current_magnitude - spectro_mag))

    return src_est, error, sdr


def grad_beta(Axd, spectro, beta=2., direc='right', eps=1e-8):

    if direc == 'right':
        G = (np.abs(Axd) + eps) ** (beta - 2) * (np.abs(Axd) - spectro)
    else:
        if beta == 0:
            G = 1 / (spectro + eps) - 1 / (np.abs(Axd) + eps)
        elif beta == 1:
            G = np.log((np.abs(Axd)) / (spectro + eps) + eps)
        else:
            G = ((np.abs(Axd) + eps) ** (beta - 1) - (spectro + eps) ** (beta - 1)) / (beta - 1)

    return G


def grad_beta_eps(stft_est, spectro, d=1, beta=2., direc='right', eps=1e-8):

    spectro_eps = (spectro ** (2/d) + eps) ** (d/2)
    abs_Axd_eps = (np.abs(stft_est) ** 2 + eps) ** (d/2)

    if direc == 'right':
        G = abs_Axd_eps ** (beta - 2) * (abs_Axd_eps - spectro_eps)
    else:
        if beta == 0:
            G = 1 / spectro_eps - 1 / abs_Axd_eps
        elif beta == 1:
            G = np.log(abs_Axd_eps / spectro_eps)
        else:
            G = (abs_Axd_eps ** (beta - 1) - spectro_eps ** (beta - 1)) / (beta - 1)

    return G


def bregmisi(mix_stft, spectro, win_length=None, hop_length=None, win_type='hann', src_ref=None, beta=2., d=1,
             grad_step=1e-3, direc='right', max_iter=20, eps=1e-8):
    """The Gradient Descent algorithm for phase recovery in audio source separation
    Args:
        mix_stft: numpy.ndarray (nfreqs, nframes) - input mixture STFT
        spectro: numpy.ndarray (nfreqs, nframes, nrsc) - the target sources' magnitude or power spectrograms
        win_length: int - the window length
        hop_length: int - the hop size of the STFT
        src_ref: numpy.ndarray (nsamples, nrsc) - reference sources for computing the SDR over iterations
        max_iter: int - number of iterations
        win_type: string - window type
        direc: string ('Right' or 'Left') - corresponds to the problem formulation
        d: int - magnitude (1) or power (2) measurements
        beta: float - parameter of the beta-divergence
        grad_step: float - step size for the gradient descent
        eps: float - small ridge added to the loss for avoiding numerical issues
    Returns:
        src_est: numpy.ndarray (nsamples, nrsc) - the time-domain estimated sources
        sdr: list (max_iter) - score (SDR in dB) over iterations
    """

    # Parameters
    n_src = spectro.shape[2]
    n_fft = (spectro.shape[0] - 1)*2
    if win_length is None: win_length = n_fft
    if hop_length is None: hop_length = win_length // 2

    # Pre allocate SDR and error
    compute_sdr = not (src_ref is None)
    sdr = []

    # Initialization with amplitude mask
    spectro_mag = np.power(spectro, 1/d)
    src_est = amplitude_mask(spectro_mag, mix_stft, win_length=win_length, hop_length=hop_length, win_type=win_type)
    if compute_sdr:
        sdr.append(get_score(src_ref, src_est))

    # Loop over iterations
    for iteration_number in range(max_iter):

        # Get the STFTs
        stft_est = my_stft(src_est, n_fft=n_fft, hop_length=hop_length, win_length=win_length, win_type=win_type)

        # Gradient descent in the TF domain
        #G = grad_beta(stft_est ** d, spectro, beta, direc)
        #breg_grad = d * (stft_est * (np.abs(stft_est) ** (d - 2)) * G)
        G = grad_beta_eps(stft_est, spectro, d, beta, direc, eps)
        breg_grad = d * (stft_est * ((np.abs(stft_est) ** 2 + eps) ** (d/2 - 1)) * G)
        stft_est -= grad_step * breg_grad

        # Compute and distribute the mixing error
        mixing_error = mix_stft - np.sum(stft_est, axis=2)
        corrected_stft = stft_est + np.repeat(mixing_error[:, :, np.newaxis], n_src, axis=2) / n_src

        # Back to time domain and score
        src_est = my_istft(corrected_stft, win_length=win_length, hop_length=hop_length, win_type=win_type)

        # BSS score
        if compute_sdr:
            sdr.append(get_score(src_ref, src_est))

    return src_est, sdr


def bregmisi_all(mix_stft, spectro_mag, win_length=None, hop_length=None, win_type='hann', src_ref=None, beta=2.,
                 grad_step=1e-3 * np.ones((2, 2)), max_iter=20):
    """The Gradient Descent algorithm for phase recovery in audio source separation, for d=1 and 2 and for the "right"
    and "left" problems
    Args:
        Same as the 'pr_breg_grad_ssep' function (minus 'd' and 'direc') and here 'spectro_mag' is the magnitude
    Returns:
        out: dictionary containing the estimated sources and corresponding SDRs over iterations
    """

    # Get the corresponding power spectrogram (for d=2)
    spectro_pow = np.power(spectro_mag, 2)

    est_1r, sdr_1r = bregmisi(mix_stft, spectro_mag, src_ref=src_ref, win_length=win_length, hop_length=hop_length,
                              win_type=win_type, beta=beta, d=1, grad_step=grad_step[0, 0], direc='right',
                              max_iter=max_iter)
    est_2r, sdr_2r = bregmisi(mix_stft, spectro_pow, src_ref=src_ref, win_length=win_length, hop_length=hop_length,
                              win_type=win_type, beta=beta, d=2, grad_step=grad_step[1, 0], direc='right',
                              max_iter=max_iter)
    est_1l, sdr_1l = bregmisi(mix_stft, spectro_mag, src_ref=src_ref, win_length=win_length, hop_length=hop_length,
                              win_type=win_type, beta=beta, d=1, grad_step=grad_step[0, 1], direc='left',
                              max_iter=max_iter)
    est_2l, sdr_2l = bregmisi(mix_stft, spectro_pow, src_ref=src_ref, win_length=win_length, hop_length=hop_length,
                              win_type=win_type, beta=beta, d=2, grad_step=grad_step[1, 1], direc='left',
                              max_iter=max_iter)

    # Store estimated sources and SDR over iterations (if any)
    out = {'src_est_1r': est_1r,
           'src_est_2r': est_2r,
           'src_est_1l': est_1l,
           'src_est_2l': est_2l,
           'sdr_1r': sdr_1r,
           'sdr_2r': sdr_2r,
           'sdr_1l': sdr_1l,
           'sdr_2l': sdr_2l
           }

    return out

# EOF
