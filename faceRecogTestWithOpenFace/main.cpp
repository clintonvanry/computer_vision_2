#include <iostream>
#include <cmath>
#include <map>
#include <filesystem>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "faceBlendCommon.hpp"

using namespace cv;
using namespace dlib;

#define recThreshold 0.8

// read names and labels mapping from file
static void readLabelNameMap(const string& filename, std::vector<string>& names, std::vector<int>& labels,std::map<int, string>& labelNameMap, char separator = ';')
{
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file)
    {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    std::string line;
    std::string name, labelStr;
    // read lines from file one by one
    while (getline(file, line)) {
        stringstream liness(line);
        // read first word which is person name
        getline(liness, name, separator);
        // read second word which is integer label
        getline(liness, labelStr);
        if(!name.empty() && !labelStr.empty()) {
            names.push_back(name);
            // convert label from string format to integer
            int label = atoi(labelStr.c_str());
            labels.push_back(label);
            // add (integer label, person name) pair to map
            labelNameMap[label] = name;
        }
    }
}

static void readDescriptors(const string& filename, std::vector<int>& faceLabels, std::vector<Mat>& faceDescriptors, char separator = ';')
{
    std::ifstream file(filename.c_str(), ifstream::in);

    if (!file)
    {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    // each line has:
    // 1st element = face label
    // rest 128 elements = descriptor elements
    std::string line;
    std::string faceLabel;
    // valueStr = one element of descriptor in string format
    // value = one element of descriptor in float
    std::string valueStr;
    float value;
    std::vector<float> faceDescriptorVec;
    // read lines from file one by one
    while (getline(file, line))
    {
        stringstream liness(line);
        // read face label
        // read first word on a line till separator
        getline(liness, faceLabel, separator);
        if(!faceLabel.empty())
        {
            faceLabels.push_back(std::atoi(faceLabel.c_str()));
        }

        faceDescriptorVec.clear();
        // read rest of the words one by one using separator
        while (getline(liness, valueStr, separator))
        {
            if (!valueStr.empty())
            {
                // convert descriptor element from string to float
                faceDescriptorVec.push_back(atof(valueStr.c_str()));
            }
        }

        // convert face descriptor from vector of float to Dlib's matrix format
        Mat faceDescriptor(faceDescriptorVec);
        faceDescriptors.push_back(faceDescriptor.clone());
    }
}

// find nearest face descriptor from vector of enrolled faceDescriptor
// to a query face descriptor
void nearestNeighbor(Mat& faceDescriptorQuery,
                     std::vector<Mat>& faceDescriptors,
                     std::vector<int>& faceLabels, int& label, float& minDistance) {
    int minDistIndex = 0;
    minDistance = 1.0;
    label = -1;
    // Calculate Euclidean distances between face descriptor calculated on face dectected
    // in current frame with all the face descriptors we calculated while enrolling faces
    // Calculate minimum distance and index of this face
    for (int i = 0; i < faceDescriptors.size(); i++) {
        double distance = cv::norm(faceDescriptors[i].t() - faceDescriptorQuery);
        if (distance < minDistance) {
            minDistance = distance;
            minDistIndex = i;
        }
    }
    // if minimum distance is greater than a threshold
    // assign integer label -1 i.e. unknown face
    if (minDistance > recThreshold)
    {
        label = -1;
    }
    else
    {
        label = faceLabels[minDistIndex];
    }
}

int main() {
    // Initialize face detector, facial landmarks detector and face recognizer
    const std::string recModelPath = "../data/models/openface.nn4.small2.v1.t7";
    frontal_face_detector faceDetector = get_frontal_face_detector();
    dnn::Net recModel = dnn::readNetFromTorch(recModelPath);
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize("../data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;

    // read names, labels and labels-name-mapping from file
    std::map<int, string> labelNameMap;
    std::vector<string> names;
    std::vector<int> labels;
    const string labelNameFile = "../data/models/label_name_openface.txt";
    readLabelNameMap(labelNameFile, names, labels, labelNameMap);

    // read descriptors of enrolled faces from file
    const string faceDescriptorFile = "../data/models/descriptors_openface.csv";
    std::vector<int> faceLabels;
    std::vector<Mat> faceDescriptors;
    readDescriptors(faceDescriptorFile, faceLabels, faceDescriptors);

    std::string imagePath = "../data/images/faces/satya_demo.jpg";
    Mat im = cv::imread(imagePath);

    if (im.empty()) {
        exit(0);
    }
    double t = cv::getTickCount();
    cv_image<bgr_pixel> imDlib(im);

    // detect faces in image
    std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
    std::string name;

    // Now process each face we found
    for (int i = 0; i < faceRects.size(); i++)
    {
        std::cout << faceRects.size() << " Face(s) Found" << std::endl;

        Mat alignedFace;
        alignFace(im, alignedFace, faceRects[i], landmarkDetector, cv::Size(96, 96));
        cv::Mat blob = dnn::blobFromImage(alignedFace, 1.0 / 255, cv::Size(96, 96), Scalar(0, 0, 0), false, false);
        recModel.setInput(blob);
        Mat faceDescriptorQuery = recModel.forward();

        // Find closest face enrolled to face found in frame
        int label;
        float minDistance;
        nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);
        // Name of recognized person from map
        name = labelNameMap[label];

        cout << "Time taken = " << ((double) cv::getTickCount() - t) / cv::getTickFrequency() << endl;

        // Draw a rectangle for detected face
        Point2d p1 = Point2d(faceRects[i].left(), faceRects[i].top());
        Point2d p2 = Point2d(faceRects[i].right(), faceRects[i].bottom());
        cv::rectangle(im, p1, p2, Scalar(0, 0, 255), 1, LINE_8);

        // Draw circle for face recognition
        Point2d center = Point((faceRects[i].left() + faceRects[i].right()) / 2.0,
                               (faceRects[i].top() + faceRects[i].bottom()) / 2.0);
        int radius = static_cast<int> ((faceRects[i].bottom() - faceRects[i].top()) / 2.0);
        cv::circle(im, center, radius, Scalar(0, 255, 0), 1, LINE_8);

        // Write text on image specifying identified person and minimum distance
        stringstream stream;
        stream << name << " ";
        stream << fixed << setprecision(4) << minDistance;
        string text = stream.str(); // name + " " + std::to_string(minDistance);
        cv::putText(im, text, p1, FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 0, 0), 2);
    }

    // Show result
    cv::imshow("photo", im);
    cv::imwrite(cv::format("output-openface-%s.jpg",name.c_str()),im);

    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}
