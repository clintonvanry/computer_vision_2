#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include "faceBlendCommon.hpp"

using namespace cv;
using namespace std;


int main() {

    // load landmark detector
    // Landmark model location
    string PREDICTOR_PATH =  "../data/models/shape_predictor_68_face_landmarks.dat";

    // Get the face detector
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
    // The landmark detector is implemented in the shape_predictor class
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize(PREDICTOR_PATH) >> landmarkDetector;

    Mat img = imread("../data/images/girl-no-makeup.jpg");
    imshow("original", img);
    waitKey(0);

    destroyAllWindows();
    return 0;
}
