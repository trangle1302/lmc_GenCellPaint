# Generative model for insilico painting
*This is the algorithm that placed first in ISBI 2024 [Light My Cell](https://lightmycells.grand-challenge.org/evaluation/phase-2/leaderboard/) competition.*
  
This repo is modified from [Stable-diffusion](https://github.com/CompVis/stable-diffusion) code base, thanks the authors who made their work public!

## Requirements
A suitable [conda](https://conda.io/) environment named `ldmbf` can be created
and activated with:

```
conda env create -f environmentclean.yaml
conda activate ldmbf
```

## Dataset
The models were trained on ~56700 previously unpublished images from 30 data acquisition sites. This dataset is very heterogeneous, including 3 imaging modality (Bright Field, Phase Contrast, Differential Inference Contrast microscopy), 40x - 100x objective, multiple different cell lines etc. Most importantly is the class imbalance (i.e. there's a few order of magnitude difference in number of training data for Actin vs Nucleus), and vaying combination of channels per image (1-3 organelles for each input). There's no field of view with all organelles.
More about the training dataset [here](https://lightmycells.grand-challenge.org/database/).

Final evaluation was performed on a hidden ~300 FOVs, which a submitted docker container will be evaluated for.

## Model & Weights

1 simplified VQGAN model was trained for each organnelle, with varying performance. These models can be used separately for your organelle of interest.

You can access all organelle checkpoints from here: checkpoint timestamped on April 19, 2024. 

```
wget https://ell-vault.stanford.edu/dav/trangle/www/ISBI2024_lmc_checkpoints.zip
unzip ISBI2024_lmc_checkpoints.zip
```

This should contains:
- `BF_to_Nucleus.ckpt`: 722MB.
- `BF_to_Mitochondria.ckpt`: 722MB.
- `BF_to_Actin.ckpt`: 722MB.
- `BF_to_Tubulin.ckpt`: 722MB.

> **_NOTE:_** I stopped trainning to submit before the deadline of April 20. However, it hasn't converged! This means that to have better and more generalizable models for the community to use later, we can re-split the whole dataset (when test set become public), and train further for better models.

### Docker container:
Building the docker image for submission (the checkpoints should be downloaded first): 
```
bash test_run.sh
docker save algo0:latest | gzip -c > algo0_latest.tar.gz
```

You can test the winning algorithm on grand challenge platform [here](https://grand-challenge.org/algorithms/lmc_control/).
The code to build and run docker container is also provided in this repo. This docker container takes transmitted light tiff as input, and output 4 same size predicted organelle tiff as outputs.

You can also download the docker container image from [here](https://ell-vault.stanford.edu/dav/trangle/www/ISBI2024_lmc_algo0_latest.tar.gz) to test locally. The docker image of course contains the checkpoints described above. 

### Example results

TODO: attached some image results here.


### Training & Inference
This command train an vqgan to predict organelle from transmitted light inputs
```
python main.py -t -b configs/autoencoder/lmc_BF_<organelle>.yaml --gpus=0,
```

Each model configuration:
```
  | Name            | Type                     | Params
-------------------------------------------------------------
0 | encoder         | Encoder                  | 22.3 M
1 | decoder         | Decoder                  | 33.0 M
2 | loss            | VQLPIPSWithDiscriminator | 17.5 M
3 | quantize        | VectorQuantizer2         | 24.6 K
4 | quant_conv      | Conv2d                   | 12    
5 | post_quant_conv | Conv2d                   | 12    
-------------------------------------------------------------
58.1 M    Trainable params
14.7 M    Non-trainable params
72.8 M    Total params
291.218   Total estimated model params size (MB)
```


### Other attempt (in March 2024)
As some might have guessed by the repo, I did try latent diffusion for this problem, since modelling the joint distribution and conditioning of multiple different channel combination sound promising. However, my limited attempt on this dataset showed that the performance of 1 diffusion model is subpar to individual organelle models. This approach did work better on a different dataset, where each FOV contains all channels of interest.



### BibTeX
This is a place holder, have not had time to finish the manuscript yet.
```
@article {Le2024.05.31.596710,
        author = {Le, Trang and Lundberg, Emma},
        title = {High-Resolution In Silico Painting with Generative Models},
        elocation-id = {2024.05.31.596710},
        year = {2024},
        doi = {10.1101/2024.05.31.596710},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2024/06/03/2024.05.31.596710},
        eprint = {https://www.biorxiv.org/content/early/2024/06/03/2024.05.31.596710.full.pdf},
        journal = {bioRxiv}
}
```
