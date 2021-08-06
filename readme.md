# Violence Detection Tests on the AIRTLab Dataset

This repository contains the source code of the experiments presented in the paper

>P. Sernani, N. Falcionelli, S. Tomassini, P. Contardo, and A.F. Dragoni, *Deep learning for automatic violence detection: tests on the AIRTLab dataset*.

The paper is currently under review for the publication in the [IEEE Access](https://ieeeaccess.ieee.org/) journal.

The experiments are accuracy tests of three deep learning based models on the classification of the videos of the AIRTLab dataset, the Hockey Fight dataset, and the Crowd Violence dataset to identify sequences of frames containing violent scenes.

Specifically, the source code implementing the three models and the experiments on the datasets is contained in two Jupyter notebooks, which are available in the “notebook” directory of this repository:

- the [Violence_Detection_on_the_AIRTLAB_Dataset.ipynb](notebook/Violence_Detection_on_the_AIRTLAB_Dataset.ipynb) notebook includes the experiments of the three models on the AIRTLab dataset;
- the [Violence_Detection_on_the_Hockey_Fight_and_Crowd_Violence_Dataset.ipynb](notebook/Violence_Detection_on_the_Hockey_Fight_and_Crowd_Violence_Dataset.ipynb) notebook includes the experiments of the three models on the Hockey Fight and Crowd Violence dataset

Moreover, there are four additional notebooks in the "notebook/transfer-learning" folder. Such notebook include accuracy tests on the AIRTLab, Hockey Fight, and Crowd Violence datasets with models composed of 2D CNN and a recurrent layer (ConvLSTM or Bidirectional-LSTM). Specifically:
- the [Transfer_Learning_Violence_Detection_On_AIRTLab_Dataset_(2D_CNNs_+_Bi-LSTM).ipynb](notebook/transfer-learning/Transfer_Learning_Violence_Detection_On_AIRTLab_Dataset_(2D_CNNs_%2B_Bi-LSTM).ipynb) notebook includes the accuracy tests of five 2D CNNs trained on ImageNet combined with a Bidirectional-LSTM, on the AIRTLab dataset;
- the [Transfer_Learning_Violence_Detection_On_AIRTLab_Dataset_(2D_CNNs_+_ConvLSTM).ipynb](notebook/transfer-learning/Transfer_Learning_Violence_Detection_On_AIRTLab_Dataset_(2D_CNNs_%2B_ConvLSTM).ipynb) notebook includes the accuracy tests of five 2D CNNs trained on ImageNet combined with a ConvLSTM, on the AIRTLab dataset;
- the [Transfer_Learning_Violence_Detection_on_Hockey_Fight_And_Crowd_Violence_Datasets_(2D_CNNs_+_Bi-LSTM).ipynb](notebook/transfer-learning/Transfer_Learning_Violence_Detection_on_Hockey_Fight_And_Crowd_Violence_Datasets_(2D_CNNs_%2B_Bi-LSTM).ipynb) notebook includes the accuracy tests of five 2D CNNs trained on ImageNet combined with a Bidirectional-LSTM, on the Hockey Fight and Crowd Violence datasets;
-  the[Transfer_Learning_Violence_Detection_on_Hockey_Fight_And_Crowd_Violence_Datasets_(2D_CNNs_+_ConvLSTM).ipynb](notebook/transfer-learning/Transfer_Learning_Violence_Detection_on_Hockey_Fight_And_Crowd_Violence_Datasets_(2D_CNNs_%2B_ConvLSTM).ipynb) notebook includes the accuracy tests of five 2D CNNs trained on ImageNet combined with a ConvLSTM, on the Hockey Fight and Crowd Violence datasets;

All the experiments were run on Google Colab, using the GPU runtime and Keras 2.4.3, with the TensorFlow 2.4.1 backend, and scikit-learn 0.22.2.post1.

## Data Description

The experiments are based on the videos of the AIRTLab dataset, the Hockey Fight dataset and the Crowd Violence dataset.

The AIRTLab datase is available in the following GitHub repository

><https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos>

The dataset contains 350 video clips which are MP4 video files (30 fps, 1920 x 1080, 30 fps, average lenght 5.63 seconds). 120 clips contain non-violent behaviours, while 230 clips contain violent behaviours. The full specification of the dataset is available in the repository and in a dedicated open-access data paper, which can be found at the following link

><https://www.sciencedirect.com/science/article/pii/S2352340920314682>.

The Hockey Fight dataset is described in
> E. Bermejo Nievas, O. Deniz Suarez, G. Bueno García, and R. Sukthankar, *Violence detection in video using computer vision techniques*, in Computer Analysis of Images and Patterns, P. Real, D. Diaz-Pernil, H. Molina-Abril, A. Berciano, and W. Kropatsch, Eds. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011, pp. 332–339.

The Crowd Violence dataset is described in
> T. Hassner, Y. Itcher, and O. Kliper-Gross, *Violent flows: Real-time detection of violent crowd behavior*, in 2012 IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, 2012, pp. 1–6.

## Model architectures

Three different models were implemented and tested on the datasets:
1. a combination of [C3D](https://arxiv.org/abs/1412.0767), a 3D CNN pre-trained on the [Sport-1M dataset](https://cs.stanford.edu/people/karpathy/deepvideo/) used as a feature extractor, and an SVM a classifier, in a transfer learning fashion. Specifically, the C3D original weights were used without retraining, while the SVM is trained from scratch on the clips included in the datasets.
2. a combination of C3D as a feature extractor and two fully connected layers to obtain an end-to-end network for classification. Also in this model the C3D original weights were used without training again, applying transfer learning. Only the two final fully connected layer are trained from scratch on the clips of the datasets.
3. a combination of the [ConvLSTM architecture](https://arxiv.org/abs/1506.04214), and two fully connected layers, getting an end-to-end network for classification. The entire network is trained from scratch on the clips of the datasets.

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
| Dense,  *1 unit*, *Sigmoid activation*         | (None, 1)            |         257 |

## Experiments

The experimental evaluation of the described models is based on the Stratified Shuffle Split strategy to split the available data in 80% for training and 20% for testing. With the two end-to-end models the 12.5% of the training data (i.e. 10% of the entire dataset) was use for validation. For each split the confusion matrix and a classification report are printed as output. Moreover, at the end of each test, the average value of accuracy, sensitivity, specificity, and F1-score are reported, as well as the ROC computed in each split.

To prove the effectiveness of the proposed models, we did also transfer learning experiments using five 2D CNNs pretrained on ImageNet:

- VGG16 (https://keras.io/api/applications/vgg/#vgg16-function)
- VGG19 (https://keras.io/api/applications/vgg/#vgg19-function)
- ResNet50V2 (https://keras.io/api/applications/resnet/#resnet50v2-function)
- Xception (https://keras.io/api/applications/xception/)
- NASNet Mobile (https://keras.io/api/applications/nasnet/#nasnetmobile-function)

These five 2D CNN were combined with a recurrent layer to work on videos, and with fully connected layers to perform the final classification. A Bidirectional-LSTM (Bi-LSTM) and a ConvLSTM were used as recurrent layer. Therefore, a total of 10 models was tested, according to the following tables. Note that the number of parameter of the recurrent layer (Bi-LSTM or ConvLSTM) depends on the previous 2D CNN.

### 2D CNNs and Bi-LSTM

| Layer Type                                     | Output Shape         | Parameter # |
|:-----------------------------------------------|:---------------------|------------:|
| Time Distributed 2D CNN                        | -                    |           - |
| Time Distributed Flatten                       | -                    |           0 |
| Bi-LSTM, *128 units*                           | (None, 256)          |           - |
| Dropout, *0.5*                                 | (None, 256)          |           0 |
| Dense,  *128 units*, *ReLU activation*         | (None, 128)          |         32896 |
| Dropout, *0.5*                                 | (None, 128)          |           0 |
| Dense,  *1 unit*, *Sigmoid activation*         | (None, 1)            |         129 |

### 2D CNNs and ConvLSTM

| Layer Type                                     | Output Shape         | Parameter # |
|:-----------------------------------------------|:---------------------|------------:|
| Time Distributed 2D CNN                        | -                    |           - |
| ConvLSTM2D, *64 3x3 filters*                   | (None, 5, 5, 64)     |           - |
| Flatten                                        | (None, 1600)         |           0 |
| Dense, *256 units*, *ReLU activation*          | (None, 256)          |   198246656 |
| Dropout, *0.5*                                 | (None, 256)          |           0 |
| Dense,  *1 unit*, *Sigmoid activation*         | (None, 1)            |         257 |
