#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import librosa

__author__ = 'Paul Magron -- IRIT, Université de Toulouse, CNRS, France'
__docformat__ = 'reStructuredText'


def load_src(audio_path, sample_rate):
    """ Load wav files as numpy arrays
    Args:
        audio_path: string - the path where to load the files
        sample_rate: int - the sampling frequency
    Returns:
        src_ref: numpy.ndarray (nsamples, 2) - time-domain original sources
        mix: numpy.ndarray (nsamples, 1) - time-domain mixture
    """

    # Load time-domain signals
    clean = librosa.core.load(audio_path + 'clean.wav', sr=sample_rate)[0]
    noise = librosa.core.load(audio_path + 'noise.wav', sr=sample_rate)[0]

    # Create array with both sources and compute the mix
    src_ref = np.concatenate((clean[:, np.newaxis], noise[:, np.newaxis]), axis=1)

    # Create the mixture
    mix = np.sum(src_ref, axis=1)[:, np.newaxis]

    return src_ref, mix


def record_src(audio_path, src, sample_rate, rec_mix=False):
    """ Record signals as wav files
    Args:
        audio_path: string - the path where to record the files
        src: numpy.ndarray (nsamples, 2) - speech and noise signals
        sample_rate: int - the sampling frequency
        rec_mix: bool - record the mixture (True) or not (False, default)
    """

    # Record the speech and noise sources
    librosa.output.write_wav(audio_path + 'clean.wav', np.asfortranarray(src[:, 0]), sr=sample_rate)
    librosa.output.write_wav(audio_path + 'noise.wav', np.asfortranarray(src[:, 1]), sr=sample_rate)

    # If original sources, also record the mixture
    if rec_mix:
        librosa.output.write_wav(audio_path + 'mix.wav', np.asfortranarray(np.sum(src, axis=1)), sr=sample_rate)

    return


# EOF
