#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Paul Magron -- INRIA Nancy - Grand Est, France'
__docformat__ = 'reStructuredText'

import numpy as np
import torch
import json
from pathlib import Path
from open_unmx import model
import tqdm
import sys
eps = sys.float_info.epsilon

'''
This script is largely adapted from the original Open Unmix code,
which is available at: https://github.com/sigsep/open-unmix-pytorch
If you use it, please acknowledge it by citing the corresponding paper:

F.-R. Stoter, S. Uhlich, A. Liutkus and Y. Mitsufuji,
"Open-Unmix - A Reference Implementation for Music Source Separation",
Journal of Open Source Software, 2019.

'''


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )
    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def load_model(target, model_name='open_unmx/', device='cpu'):
    """ Load the Open Unmix model for estimating the target spectrogram
    Args:
        target: string - 'speech' or 'noise'
        model_name: string - the path where the model is stored
        device: string - 'cpu' or 'cuda'
    """
    model_path = Path(model_name).expanduser()

    # load model from disk
    with open(Path(model_path, target + '.json'), 'r') as stream:
        results = json.load(stream)

    target_model_path = next(Path(model_path).glob("%s*.pth" % target))
    state = torch.load(
        target_model_path,
        map_location=device
    )

    max_bin = bandwidth_to_max_bin(
        state['sample_rate'],
        results['args']['nfft'],
        results['args']['bandwidth']
    )

    unmix = model.OpenUnmix(
        n_fft=results['args']['nfft'],
        n_hop=results['args']['nhop'],
        nb_channels=results['args']['nb_channels'],
        hidden_size=results['args']['hidden_size'],
        max_bin=max_bin
    )

    unmix.load_state_dict(state)
    unmix.stft.center = True
    unmix.eval()
    unmix.to(device)

    return unmix


def estim_spectro_from_mix(audio, device='cpu'):
    """
    Adapted from Open Unmix
    Performing the separation on audio input

    Parameters
    ----------
    audio: np.ndarray [shape=(nb_samples, nb_channels, nb_timesteps)]
        mixture audio

    targets: list of str
        a list of the separation targets.
        Note that for each target a separate model is expected
        to be loaded.

    model_name: str
        name of torchhub model or path to model folder, defaults to `umxhq`

    niter: int
         Number of EM steps for refining initial estimates in a
         post-processing stage, defaults to 1.

    softmask: boolean
        if activated, then the initial estimates for the sources will
        be obtained through a ratio mask of the mixture STFT, and not
        by using the default behavior of reconstructing waveforms
        by using the mixture phase, defaults to False

    alpha: float
        changes the exponent to use for building ratio masks, defaults to 1.0

    residual_model: boolean
        computes a residual target, for custom separation scenarios
        when not all targets are available, defaults to False

    device: str
        set torch device. Defaults to `cpu`.

    Returns
    -------
    estimates: `dict` [`str`, `np.ndarray`]
        dictionary of all restimates as performed by the separation model.

    """
    # Create a 2-channel audio for feeding open unmix
    audio = np.repeat(audio, 2, axis=1)
    # convert numpy audio to torch
    audio_torch = torch.tensor(audio.T[None, ...]).float().to(device)

    source_names = []
    V = []
    targets = ['speech', 'noise']
    model_name = 'open_unmx/'

    for j, target in enumerate(tqdm.tqdm(targets)):
        unmix_target = load_model(
            target=target,
            model_name=model_name,
            device=device
        )
        Vj = unmix_target(audio_torch).cpu().detach().numpy()
        # output is nb_frames, nb_samples, nb_channels, nb_bins
        V.append(Vj[:, 0, ...])  # remove sample dim
        source_names += [target]

    V = np.transpose(np.array(V), (1, 3, 2, 0))

    # Back to monochannel and remove extra 0 frames
    V = np.transpose(V[:, :, 0, :], (2, 1, 0))

    return V

