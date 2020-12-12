#Facial Landmark Detection in Dlib
Landmark detection is a two step process.

## 1. Face Detection
In the first step, you need to detect the face in an image. For best results we should use the same face detector used in training the landmark detector. The output of a face detector is a rectangle (x, y, w, h) that contains the face.

We have seen that OpenCV’s face detector is based on HAAR cascades. Dlib’s face detector is based on Histogram of Oriented Gradients features and Support Vector Machines (SVM).

## 2. Landmark detection
The landmark detector finds the landmark points inside the face rectangle.
