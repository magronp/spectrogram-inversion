#  Phase recovery with Bregman divergences for audio source separation

This repository contains the code for reproducing the experiments in our paper entitled [Phase recovery with the Bregman divergence for audio source separation](https://arxiv.org/abs/2010.10255), published at the IEEE International Conference on Audio, Speech and Signal Processing (ICASSP) 2021.

### Getting the data

After cloning or downloading this repository, you will need to get the speech and noise data to reproduce the results.

* The speech data is obtained from the __VoiceBank__ dataset available [here](https://datashare.is.ed.ac.uk/handle/10283/2791). You should download the `clean_testset_wav.zip` file, and unzip it in the `data/VoiceBank/` folder.
Note that you can change the folder structure, as long as you change the path accordingly in the code.

* The noise data is obtained from the __DEMAND__ dataset available [here](https://zenodo.org/record/1227121#.X4hjZXZfg5k). You should download the `DLIVING_16k.zip`, `SPSQUARE_16k.zip` and `TBUS_16k.zip` files, and unzip them in the `data/DEMAND/` folder.

Note that you can change the folder structures, as long as you change the speech and noise directory paths accordingly in the code.

Then, simply execute the `prepare_data.py` script to create the noisy mixtures.

### Getting the pre-trained model

To run the experiments, you will need to first estimate the spectrograms of the sources, which is done using the pytorch implementation of the [Open Unmix](https://github.com/sigsep/open-unmix-pytorch) model trained for a speech enhancement task. 

The pre-trained model for estimating the speech and noise spectrograms is available [here](https://zenodo.org/record/3786908#.X4hkeHZfg5k).
You should place the  `.json` and `.pth` files in the `open_unmx/` folder. Note that you should also rename the `.pth` files simply as `speech.pth` and `noise.pth`.

### Reproducing the experiments

Now that you're all set, simply run the following scripts:

- `validation.py` will perform a grid search over the gradient step size on the validation subset to determine its optimal value for every setting.
It will also reproduce Fig. 1 from the paper.

- `testing.py` will run the algorithms (proposed gradient descent and MISI) on the test subset and plot the results corresponding to Fig. 2 in the paper.


### Reference

<details><summary>If you use any of this code for your research, please cite our paper:</summary>
  
```latex
@inproceedings{Magron2021,  
  author={P. Magron and P.-H. Vial and T. Oberlin and C. F{\'e}votte},  
  title={Phase recovery with Bregman divergences for audio source separation},  
  booktitle={Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},  
  year={2021},
  month={June}
}
```

</p>
</details>
# spec_inv
