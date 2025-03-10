#include <iostream>
#include <cmath>
#include <map>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "faceBlendCommon.hpp"

using namespace cv;
using namespace dlib;

namespace fs = std::filesystem;

void listDirectory(string dirName, std::vector<string>& folderNames, std::vector<string>& fileNames, std::vector<string>& symlinkNames)
{
    for(const auto& entry : fs::directory_iterator(fs::path(dirName)))
    {
        std::string fileName = entry.path().filename().string();
        if(entry.is_directory()){
            std::cout << "directory:" <<  fileName << std::endl;
            folderNames.emplace_back(entry.path().string());
        }
        if(entry.is_regular_file()){
            std::cout << "filename:" <<  fileName << std::endl;
            fileNames.emplace_back(fileName);
        }
        if(entry.is_symlink()){
            std::cout << "symlink:" <<  fileName << std::endl;
        }
    }

}

// filter files having extension ext i.e. jpg
void filterFiles(std::string dirPath, std::vector<string>& fileNames, std::vector<string>& filteredFilePaths, const std::string& ext, std::vector<int>& imageLabels, int index){
    for(const auto& fname : fileNames) {
        if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos)
        {
            std::string fnamePath = dirPath + "/" + fname;
            filteredFilePaths.push_back(fnamePath);
            imageLabels.push_back(index);
        }
    }
}

int main() {
    // Initialize face detector and face recognize
    const std::string recModelPath = "../data/models/openface.nn4.small2.v1.t7";
    frontal_face_detector faceDetector = get_frontal_face_detector();
    dnn::Net recModel = dnn::readNetFromTorch(recModelPath);
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize("../data/models/shape_predictor_5_face_landmarks.dat") >> landmarkDetector;

    // Now let's prepare our training data
    // data is organized assuming following structure
    // faces folder has subfolders.
    // each subfolder has images of a person
    string faceDatasetFolder = "../data/images/faces";
    std::vector<string> subfolders, fileNames, symlinkNames;

    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    listDirectory(faceDatasetFolder, subfolders, fileNames, symlinkNames);
    //listdir(faceDatasetFolder, subfolders, fileNames, symlinkNames);

    // names: vector containing names of subfolders i.e. persons
    // labels: integer labels assigned to persons
    // labelNameMap: dict containing (integer label, person name) pairs
    std::vector<string> names;
    std::vector<int> labels;
    std::map<int, string> labelNameMap;
    // add -1 integer label for un-enrolled persons
    names.emplace_back("unknown");
    labels.push_back(-1);

    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<string> imagePaths;
    std::vector<int> imageLabels;

    // variable to hold any subfolders within person subFolders
    std::vector<string> folderNames;

    // iterate over all subFolders within faces folder
    for (int i = 0; i < subfolders.size(); i++) {
        std::string personFolderName = subfolders[i];
        // remove / or \\ from end of subFolder
        std::size_t found = personFolderName.find_last_of("/\\");
        std::string name = personFolderName.substr(found + 1);
        // assign integer label to person subFolder
        int label = i;
        // add person name and label to vectors
        names.push_back(name);
        labels.push_back(label);
        // add (integer label, person name) pair to map
        labelNameMap[label] = name;

        // read imagePaths from each person subFolder
        // clear vectors
        folderNames.clear();
        fileNames.clear();
        symlinkNames.clear();
        // folderNames and symlinkNames are useless here
        // as we are only looking for files here
        // read all files present in subFolder
        listDirectory(subfolders[i], folderNames, fileNames, symlinkNames);
        //listdir(subfolders[i], folderNames, fileNames, symlinkNames);
        // filter only jpg files
        filterFiles(subfolders[i], fileNames, imagePaths, "jpg", imageLabels, i);
    }

    // process training data
    // We will store face descriptors in vector faceDescriptors
    // and their corresponding labels in vector faceLabels
    std::vector<Mat> faceDescriptors;
    std::vector<int> faceLabels;
    Mat faceDescriptor;

    // iterate over images
    for (int i = 0; i < imagePaths.size(); i++) {
        string imagePath = imagePaths[i];
        int imageLabel = imageLabels[i];

        std::cout << "processing: " << imagePath << std::endl;

        // read image using OpenCV
        Mat im = cv::imread(imagePath);

        cv_image<bgr_pixel> imDlib(im);
        std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
        std::cout << faceRects.size() << " Face(s) Found" << std::endl;

        // Now process each face we found
        for (auto & faceRect : faceRects) {
            Mat alignedFace;
            alignFace(im, alignedFace, faceRect, landmarkDetector, cv::Size(96, 96));

            cv::Mat blob = dnn::blobFromImage(alignedFace, 1.0/255, cv::Size(96, 96), Scalar(0,0,0), false, false);
            recModel.setInput(blob);
            faceDescriptor = recModel.forward();

            // add face descriptor and label for this face to
            // vectors faceDescriptors and faceLabels
            faceDescriptors.push_back(faceDescriptor.clone());

            // add label for this face to vector containing labels corresponding to
            // vector containing face descriptors
            faceLabels.push_back(imageLabel);

        }
    }

    std::cout << "number of face descriptors " << faceDescriptors.size() << std::endl;
    std::cout << "number of face labels " << faceLabels.size() << std::endl;

    // write label name map to disk
    const string labelNameFile = "label_name_openface.txt";
    ofstream of;
    of.open (labelNameFile);
    for (int m = 0; m < names.size(); m++)
    {
        of << names[m];
        of << ";";
        of << labels[m];
        of << "\n";
    }
    of.close();

    // write face labels and descriptor to disk
    // each row of file descriptors_openface.csv has:
    // 1st element as face label and
    // rest 128 as descriptor values
    const string descriptorsPath = "descriptors_openface.csv";
    ofstream ofs;
    ofs.open(descriptorsPath);
    // write descriptors
    for (int m = 0; m < faceDescriptors.size(); m++) {
        Mat faceDescriptorVec = faceDescriptors[m];
        ofs << faceLabels[m];
        ofs << ";";
        for (int n = 0; n < faceDescriptorVec.cols; n++) {
            ofs << std::fixed << std::setprecision(8) << faceDescriptorVec.at<float>(n);
            // cout << n << " " << faceDescriptorVec.at<float>(n) << endl;
            if ( n == (faceDescriptorVec.cols - 1)) {
                ofs << "\n";  // add ; if not the last element of descriptor
            } else {
                ofs << ";";  // add newline character if last element of descriptor
            }
        }
    }
    ofs.close();
    return 1;
}
