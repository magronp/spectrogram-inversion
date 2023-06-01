#  Spectrogram inversion algorithms for audio source separation

This repository contains the code for reproducing the experiments in our paper entitled [Spectrogram Inversion for Audio Source Separation via Consistency, Mixing, and Magnitude Constraints](https://arxiv.org/abs/2303.01864), published at EUSIPCO 2023.

We also provide some [sound examples](#sound-examples).

## Setup

First, clone or download this repository. 

### Packages

The needed packages are listed in `requirements.txt`.
For convenience, you can create a virtual environment and automatically install them as follows:

    python3 −m venv env
    source env/bin/activate
    pip3 install -r requirements.txt


### Getting the data

You will need to get the speech and noise data to reproduce the results.

* The speech data is obtained from the __VoiceBank__ dataset available [here](https://datashare.is.ed.ac.uk/handle/10283/2791). You should download the `clean_testset_wav.zip` file, and unzip it in the `data/VoiceBank/` folder.

* The noise data is obtained from the __DEMAND__ dataset available [here](https://zenodo.org/record/1227121#.X4hjZXZfg5k). You should download the `DLIVING_16k.zip`, `SPSQUARE_16k.zip` and `TBUS_16k.zip` files, and unzip them in the `data/DEMAND/` folder.

Note that you can change the folder structures, as long as you change the speech and noise directory paths accordingly in the code.

Then, simply execute the `prepare_data.py` script to create the noisy mixtures.

### Getting the pre-trained model

Since the experiments involve estimating the sources' magnitude spectrograms, you need to download the pytorch implementation of the Open Unmix pre-trained model available [here](https://zenodo.org/record/3786908#.X4hkeHZfg5k).
You should place the  `.json` and `.pth` files in the `data/open_unmx/` folder.
Note that you don't need to install the [Open Unmix](https://github.com/sigsep/open-unmix-pytorch) package (a simplified and adapted code is provided in this repository), but make sure to check it out ;)

## Reproducing the experiments

Now that you're all set, you can run the following scripts:

- `test_simple.py` runs all algorithms on a single noisy mixture and displays the SDR over iterations.

- `validation.py` performs a grid search over the consistency weights on the validation subset. Then, it determines the optimal consistency weight and number of iterations for all algorithms. This script also plots the validation results, and reproduces Fig. 1 (a)-(d) from the paper.

- `testing.py` runs the algorithms on the test subset and displays the results corresponding to Table. 2 in the paper.


## Sound examples

Clean speech  AM <audio src="example/clean.wav" controls preload></audio>

Noisy mix (-10 dB) AM <audio src="example/mix.wav" controls preload></audio>

The estimated speech signals are below. Note that the difference is quite subtle, although one can hear slightly less voice distortion with Mix+Incons.

|     |  |
| -------- | ------- |
| AM  | <audio src="example/clean_AM.wav" controls preload></audio>   |
| MISI | <audio src="example/clean_MISI.wav" controls preload></audio>   |
| Mix+Incons    | <audio src="example/clean_Mix+Incons.wav" controls preload></audio>   |
| Incons_hardMix    | <audio src="example/clean_Incons_hardMix.wav" controls preload></audio>   |
| Mix+Incons_hardMag    | <audio src="example/clean_Mix+Incons_hardMag.wav" controls preload></audio>   |
| Mag+Incons_hardMix    | <audio src="example/clean_Mag+Incons_hardMix.wav" controls preload></audio>   |


## Reference

If you use any of this code for your research, please cite our paper:
  
```latex
@inproceedings{Magron2023specinv,  
  author={Magron, Paul and Virtanen, Tuomas},  
  title={Spectrogram Inversion for Audio Source Separation via Consistency, Mixing, and Magnitude Constraints},  
  booktitle={Proc. European Signal Processing Conference (EUSIPCO)},  
  year={2023},
  month={September}
}
```
