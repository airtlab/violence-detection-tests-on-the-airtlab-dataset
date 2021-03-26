# Violence Detection Tests on the AIRTLab Dataset

This repository contains the source code of the experiments presented in the paper

>P. Sernani, N. Falcionelli, S. Tomassini, P. Contardo, and A.F. Dragoni, *Deep learning for automatic violence detection: tests on the AIRTLab dataset*.

The paper is currently under review for the publication in the [IEEE Access](https://ieeeaccess.ieee.org/) journal.

Specifically, the source code is contained in a Jupyter notebook, which is available in the “notebook” directory of this repository.

The experiments are accuracy tests of three deep learning based models on the classification of the videos of the AIRTLab dataset, to identify sequences of frames containing violent scenes.

The experiments were run on Google Colab, using the GPU runtime and Keras 2.4.3, with the TensorFlow 2.4.1 backend, and scikit-learn 0.22.2.post1.

You can directly check the notebook in this repository, or open it in Google Colab by clicking on the following badge.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/airtlab/violence-detection-tests-on-the-airtlab-dataset/blob/master/notebook/Violence_Detection_on_the_AIRTLAB_Dataset.ipynb)

## Data Description

The experiments are based on the videos of the AIRTLab dataset, available in the following GitHub repository

><https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos>

The dataset contains 350 video clips which are MP4 video files (30 fps, 1920 x 1080, 30 fps, average lenght 5.63 seconds). 120 clips contain non-violent behaviours, while 230 clips contain violent behaviours. The full specification of the dataset is available in the repository and in a dedicated open-access data paper, which can be found at the following link

><https://www.sciencedirect.com/science/article/pii/S2352340920314682>.

## Model architectures

Three different models were implemented and tested on the dataset:
1. a combination of [C3D](https://arxiv.org/abs/1412.0767), a 3D CNN pre-trained on the [Sport-1M dataset](https://cs.stanford.edu/people/karpathy/deepvideo/) used as a feature extractor, and an SVM a classifier, in a transfer learning fashion. Specifically, the C3D original weights were used without retraining, while the SVM is trained from scratch on the AIRTLab dataset.
2. a combination of C3D as a feature extractor and two fully connected layers to obtain an end-to-end network for classification. Also in this model the C3D original weights were used without training again, applying transfer learning. Only the two final fully connected layer are trained from scratch on the AIRTLab dataset.
3. a combination of the [ConvLSTM architecture](https://arxiv.org/abs/1506.04214), and two fully connected layers, getting an end-to-end network for classification. The entire network is trained from scratch on the AIRTLab dataset.

In the first two models C3D was used until the first fully connected layer ("fc6"). The SVM has a linear kernel
and C = 1. The following tables list the layers of the two end-to-end models, based on C3D and ConvLSTM

### C3D (until "fc6") + Fully Connected Layers

| Layer Type                                     | Output Shape             | Parameter # |
|:-----------------------------------------------|:-------------------------|------------:|
| Conv3D, *3x3x3*, *stride=1*                    | (None, 16, 112, 112, 64) |        5248 |
| MaxPooling3D, *1x2x2*                          | (None, 16, 56, 56, 64)   |           0 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 16, 56, 56, 128)  |      221312 |
| MaxPooling3D, *2x2x2*                          | (None, 8, 28, 28, 128)   |           0 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 8, 28, 28, 256)   |      884992 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 8, 28, 28, 256)   |     1769728 |
| MaxPooling3D, *2x2x2*                          | (None, 4, 14, 14, 256)   |           0 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 4, 14, 14, 512)   |     3539456 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 4, 14, 14, 512)   |     7078400 |
| MaxPooling3D, *2x2x2*                          | (None, 2, 7, 7, 512)     |           0 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 2, 7, 7, 512)     |     7078400 |
| Conv3D, *3x3x3*, *stride=1*                    | (None, 2, 7, 7, 512)     |     7078400 |
| ZeroPadding3D                                  | (None, 2, 8, 8, 512)     |           0 |
| MaxPooling3D, *2x2x2*                          | (None, 1, 4, 4, 512)     |           0 |
| Flatten                                        | (None, 8192)             |           0 |
| Dense, *4096 units*, *ReLU activation*         | (None, 4096)             |    33558528 |
| Dropout, *0.5*                                 | (None, 4096)             |           0 |
| Dense, *512 units*, *ReLU activation*          | (None, 512)              |     2097664 |
| Dropout, *0.5*                                 | (None, 512)              |           0 |
| Dense,  *1 unit*, *Sigmoid activation*         | (None, 1)                |         513 |


### End-to-End ConvLSTM

| Layer Type                                     | Output Shape         | Parameter # |
|:-----------------------------------------------|:---------------------|------------:|
| ConvLSTM2D, *64 3x3 filters*                   | (None, 110, 110, 64) |      154624 |
| Dropout, *0.5*                                 | (None, 110, 110, 64) |           0 |
| Flatten                                        | (None, 774400)       |           0 |
| Dense, *256 units*, *ReLU activation*          | (None, 256)          |   198246656 |
| Dropout, *0.5*                                 | (None, 256)          |           0 |
| Dense,  *1 unit*, *Sigmoid activation*         | (None, 1)            |         267 |

## Experiments

The notebook contains three experiments (one for each model) based on the Stratified Shuffle Split strategy to split the available data in 80% for training and 20% for testing. With the two end-to-end models the 12.5% of the training data (i.e. 10% of the entire dataset) was use for validation. For each split the confusion matrix and a classification report are printed as output. Moreover, at the end of each experiment, the average value of accuracy, sensitivity, specificity, and F1-score are reported, as well as the ROC computed in each split.