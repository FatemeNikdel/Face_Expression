# Facial Expression Recognition with Convolutional Neural Networks (CNN)
This code implements a Convolutional Neural Network (CNN) for Facial Expression Recognition using the CK+ dataset. The network architecture is defined in the `Defining Network` section and uses the MTCNN library for detecting and extracting facial features.

## Prerequisites
- Python 3
- Tensorflow 2.0 or higher
- MTCNN
- OpenCV
- Scikit-learn

## Dataset
The dataset used in this code is the CK+ dataset, which contains images of human faces with seven different emotions: anger, contempt, disgust, fear, happy, sadness, and surprise. The dataset can be downloaded from [here]( http://www.consortium.ri.cmu.edu/ckagree/).


## Code Structure
The code is divided into the following sections:

### Loading Data & Preprocess
In this section, the CK+ dataset is loaded and preprocessed. The images are resized to 64x64 and normalized to a range of [0, 1]. The labels are one-hot encoded using the `LabelBinarizer` class from the scikit-learn library.

### Data Augmentation
In this section, data augmentation is applied using the `ImageDataGenerator` class from the Tensorflow library. The following augmentations are used: rotation, width shift, height shift, shear, zoom, and horizontal flip.

### Defining Network
In this section, the CNN architecture is defined using the `Sequential` class from the Tensorflow library. The network contains two convolutional layers with batch normalization and max pooling, followed by a flatten layer, two dense layers with batch normalization, and a softmax output layer.

### Training
In this section, the network is trained using the `fit` method from the Tensorflow library. The augmented data is fed to the network in batches of 64, with a total of 30 epochs. The training process is timed using the `time` library.

## Usage
To use this code, download the `CK+` dataset and place it in a folder named CK+. Then, run the code in a Python environment with the required libraries installed.

## References
Zhang, Z., & Sabuncu, M. (2020). Facial Expression Recognition with Convolutional Neural Networks: A Survey. arXiv preprint arXiv:2005.05466.
Zhang, L., Zhang, L., & Feng, Q. (2018). Facial expression recognition using convolutional neural networks: State of the art. Neurocomputing, 318, 1-10.
