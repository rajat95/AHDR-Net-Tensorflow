# AHDR-Net-Tensorflow

This repository is the tensorflow implementation for the HDR deghosting, AHDR-Net, proposed in the paper - [**Attention-guided Network for Ghost-free High Dynamic Range Imaging**](https://arxiv.org/pdf/1904.10293.pdf).


# Requirements
system requirements:
Python:3.5.2\
numpy==1.16.4\
tensorboard==1.13.1\
tensorflow-estimator==1.13.0\
tensorflow-gpu==1.13.1\
opencv-python==4.1.0.25


## Training
The training script for this model is fusion_train_256_ahdr.py.The model was trained on UCSD dataset for 50 epochs with batch size of 8.

## Testing
`python ahdr_test.py

# Test Outputs
Test outputs consist of predicted hdr images and tonemapped its version.

## Contact
For further queries, please mail at `durgesh080793 <at> gmail <dot> com`.

