# Basic-CV
This repo contains the minimal implementations of algorithms used in computer vision. <br />
The target audience is students and beginners starting with computer vision/machine learning. <br />
To run the algorithm, go to the root directory and run the file with the name of the algorithm.  <br />

### k-NN and k-means
The 'basics' directory contains a simple implementation of k-NN and k-means algorithms. <br />
Both of them are tested on the Iris dataset and can be run from their respective files. <br />
See the docstring for details on each of the functions.

### SIFT and RANSAC

SIFT and RANSAC are implemented using inbuilt OpenCV functions. <br />
As SIFT is patented, it is free to use for research and non-commercial applications. <br />
It is not included as a default module in the new versions of OpenCV, and hence it is convenient to use an older version to run SIFT. <br />

Conda users can install via
```
conda install -c menpo opencv
```
For pip installation use
```
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
```

The example images to use SIFT are taken from the [Oxford5K](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) dataset.

### MNIST digit classification using Convolutional Neural Networks
A minimal CNN example of classifying MNIST digits using PyTorch is available in the directory 'pytorch'. <br />
To train or test, use the classification.py file. <br />
Train using the 'train_model' function. After training for 10 epochs with a batch size of 64, the test accuracy reaches ~98.5%. <br />
You will something like
```
Epoch completed 9
Model saved at epoch 9
Accuracy: 98.55
```
To test the image, use 'test_random_image' function. It will write the tested image in 
```
pytorch/mnist_model/results/random_image.png 
```
It will produce an output 
```
Label: 6, Prediction: 6
```