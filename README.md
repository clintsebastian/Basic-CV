# Basic-CV
This repo contains a few basic algorithms used in computer vision. <br />

### k-NN and k-means
The 'basics' directory contains a simple implementation of k-NN and k-means algorithms. <br />
Both of them are tested on the Iris dataset and can be run from their respective files. <br />
See the docstring for details on each of the functions.

### SIFT and RANSAC
The images are taken from the [Oxford5K](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/) dataset.
SIFT and RANSAC are implemented using inbuilt OpenCV functions. 
SIFT is patented, and it is available in older versions of OpenCV. 
Conda users can install via
```
conda install -c menpo opencv
```
For pip installation use
```
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16
```