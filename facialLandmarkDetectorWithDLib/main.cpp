#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "renderFace.hpp"

using namespace cv;
using namespace dlib;

// Write landmarks to file
void writeLandmarksToFile(full_object_detection &landmarks, const std::string &filename)
{
    // Open file
    std::ofstream ofs;
    ofs.open(filename);

    // Loop over all landmark points
    for (int i = 0; i < landmarks.num_parts(); i++)
    {
        // Print x and y coordinates to file
        ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << std::endl;

    }
    // Close file
    ofs.close();
}


int main() {

    // Get the face detector
    frontal_face_detector faceDetector = get_frontal_face_detector();

    // The landmark detector is implemented in the shape_predictor class
    shape_predictor landmarkDetector;

    // Load the landmark model
    deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    // Read Image
    std::string imageFilename("../data/images/family.jpg");

    Mat im = imread(imageFilename);

    // landmarks will be stored in results/famil_0.txt
    std::string landmarksBasename("results/family");

    // Convert OpenCV image format to Dlib's image format
    cv_image<bgr_pixel> dlibIm(im);

    // Detect faces in the image
    std::vector<dlib::rectangle> faceRects = faceDetector(dlibIm);
    std::cout << "Number of faces detected: " << faceRects.size() << std::endl;

    // Vector to store landmarks of all detected faces
    std::vector<full_object_detection> landmarksAll;

    // Loop over all detected face rectangles
    for (int i = 0; i < faceRects.size(); i++)
    {
        // For every face rectangle, run landmarkDetector
        full_object_detection landmarks = landmarkDetector(dlibIm, faceRects[i]);

        // Print number of landmarks
        if (i == 0) {
            std::cout << "Number of landmarks : " << landmarks.num_parts() << std::endl;
        }

        // Store landmarks for current face
        landmarksAll.push_back(landmarks);

        // Draw landmarks on face
        renderFace(im, landmarks);

        // Write landmarks to disk
        std::stringstream landmarksFilename;
        landmarksFilename << landmarksBasename <<  "_"  << i << ".txt";
        std::cout << "Saving landmarks to " << landmarksFilename.str() << std::endl;
        writeLandmarksToFile(landmarks, landmarksFilename.str());

    }

    // Save image
    std::string outputFilename("results/familyLandmarks.jpg");
    std::cout << "Saving output image to " << outputFilename << std::endl;
    cv::imwrite(outputFilename, im);

    // Display image
    cv::imshow("Facial Landmark Detector", im);
    cv::waitKey(0);

    return 0;
}
