#include <vector>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

#include "faceBlendCommon.hpp"

using namespace cv;
using namespace dlib;

int main() {
    // Get the face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    // The landmark detector is implemented in the shape_predictor class
    shape_predictor landmarkDetector;

    // Load the landmark model
    deserialize("../data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;

    //Read image
    Mat im = imread("../data/images/face1.png");

    // Detect landmarks
    std::vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, im);

    // Convert image to floating point in the range 0 to 1
    im.convertTo(im, CV_32FC3, 1/255.0);

    // Dimensions of output image
    Size size(600,600);

    // Variables for storing normalized image
    Mat imNorm;

    // Normalize image to output coordinates.
    normalizeImagesAndLandmarks(size, im, imNorm, points, points);

    imNorm.convertTo(imNorm, CV_8UC3, 255);

    imshow("Original Face", im);
    imshow("Aligned Face", imNorm);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
