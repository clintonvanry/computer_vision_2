#include <iostream>
#include <cmath>
#include <map>
#include <filesystem>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

#include "labelData.h"

using namespace cv;
using namespace dlib;

namespace fs = std::filesystem;

#define THRESHOLD 0.52

// ----------------------------------------------------------------------------------------
// The next bit of code defines a ResNet network. It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
        alevel0<
                alevel1<
                        alevel2<
                                alevel3<
                                        alevel4<
                                                max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                                                        input_rgb_image_sized<150>
                                                >>>>>>>>>>>>;
// ----------------------------------------------------------------------------------------

void listDirectory(std::string dirName, std::vector<std::string>& folderNames, std::vector<std::string>& fileNames, std::vector<std::string>& symlinkNames)
{
    for(const auto& entry : fs::directory_iterator(fs::path(dirName)))
    {
        std::string fileName = entry.path().filename().string();
        if(entry.is_directory()){
            std::cout << "reading directory:" <<  fileName << std::endl;
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
void filterFiles(std::string dirPath, std::vector<std::string>& fileNames, std::vector<std::string>& filteredFilePaths, const std::string& ext, std::vector<int>& imageLabels, int index){
    for(const auto& fname : fileNames) {
        if (fname.find(ext, (fname.length() - ext.length())) != std::string::npos)
        {
            std::string fnamePath = dirPath + "/" + fname;
            filteredFilePaths.push_back(fnamePath);
            imageLabels.push_back(index);
        }
    }
}

bool mapContainsKey(const std::map<int,std::string>& map, int key)
{
    if (map.find(key) == map.end())
    {
        return false;
    }
    return true;
}

// find nearest face descriptor from vector of enrolled faceDescriptor
// to a query face descriptor
void nearestNeighbor(dlib::matrix<float, 0, 1>& faceDescriptorQuery,
                     std::vector<dlib::matrix<float, 0, 1>>& faceDescriptors,
                     std::vector<int>& faceLabels, int& label, float& minDistance) {
    int minDistIndex = 0;
    minDistance = 1.0;
    label = -1;
    // Calculate Euclidean distances between face descriptor calculated on face dectected
    // in current frame with all the face descriptors we calculated while enrolling faces
    // Calculate minimum distance and index of this face
    for (int i = 0; i < faceDescriptors.size(); i++)
    {
        double distance = length(faceDescriptors[i] - faceDescriptorQuery);
        if (distance < minDistance)
        {
            minDistance = distance;
            minDistIndex = i;
        }
    }
    // Dlib specifies that in general, if two face descriptor vectors have a Euclidean
    // distance between them less than 0.6 then they are from the same
    // person, otherwise they are from different people.

    // This threshold will vary depending upon number of images enrolled
    // and various variations (illuminaton, camera quality) between
    // enrolled images and query image
    // We are using a threshold of 0.52
    // if minimum distance is greater than a threshold
    // assign integer label -1 i.e. unknown face
    if (minDistance > THRESHOLD)
    {
        label = -1;
    } else{
        label = faceLabels[minDistIndex];
    }
}

void writeDescriptorsToDisk(const std::vector<matrix<float,0,1>>& faceDescriptors, const std::vector<int>& faceLabels)
{
    // write face labels and descriptor to disk
    // each row of file descriptors.csv has:
    // 1st element as face label and
    // rest 128 as descriptor values
    const std::string descriptorsPath = "../descriptors.csv";
    std::ofstream ofs;
    ofs.open(descriptorsPath);
    // write descriptors
    for (int m = 0; m < faceDescriptors.size(); m++) {
        matrix<float,0,1> faceDescriptor = faceDescriptors[m];
        std::vector<float> faceDescriptorVec(faceDescriptor.begin(), faceDescriptor.end());
        // cout << "Label " << faceLabels[m] << endl;
        ofs << faceLabels[m];
        ofs << ";";
        for (int n = 0; n < faceDescriptorVec.size(); n++) {
            ofs << std::fixed << std::setprecision(8) << faceDescriptorVec[n];
            // cout << n << " " << faceDescriptorVec[n] << endl;
            if ( n == (faceDescriptorVec.size() - 1)) {
                ofs << "\n";  // add ; if not the last element of descriptor
            } else {
                ofs << ";";  // add newline character if last element of descriptor
            }
        }
    }
    ofs.close();
}

void getFolderAndFiles(std::vector<std::string>& names,std::vector<int>& labels, std::vector<std::string>& imagePaths, std::map<int, std::string>& labelNameMap, std::vector<int>& imageLabels)
{
    // Now let's prepare our training data
    // data is organized assuming following structure
    // faces folder has subfolders.
    // each subfolder has images of a person/celeb
    std::string faceDatasetFolder = "../resource/asnlib/publicdata/celeb_mini";
    std::vector<std::string> subfolders, fileNames, symlinkNames;

    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    listDirectory(faceDatasetFolder, subfolders, fileNames, symlinkNames);

    // variable to hold any subfolders within person subFolders
    std::vector<std::string> folderNames;

    // iterate over all subFolders within celeb folder
    for (int i = 0; i < subfolders.size(); i++)
    {
        std::string personFolderName = subfolders[i];
        // remove / or \\ from end of subFolder
        std::size_t found = personFolderName.find_last_of("/\\");
        std::string name = personFolderName.substr(found+1);

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
        // filter only jpg files
        filterFiles(subfolders[i], fileNames, imagePaths, "JPEG", imageLabels, i);
    }

}

void enrollment(std::map<int, std::string>& labelNameMap, std::vector<matrix<float,0,1>>& faceDescriptors, std::vector<int>& faceLabels,
                frontal_face_detector& faceDetector, const shape_predictor& landmarkDetector, anet_type& net, std::map<int,std::string>& folderImageMap)
{

    std::vector<std::string> names;
    std::vector<int> labels;

    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<std::string> imagePaths;
    std::vector<int> imageLabels;
    // variable to hold any subfolders within person subFolders
    std::vector<std::string> folderNames;
    getFolderAndFiles(names,labels,imagePaths,labelNameMap,imageLabels);

    // process training data
    // We will store face descriptors in vector faceDescriptors
    // and their corresponding labels in vector faceLabels
    //std::vector<matrix<float,0,1>> faceDescriptors;
    //std::vector<int> faceLabels;

    // iterate over images
    for (int i = 0; i < imagePaths.size(); i++)
    {
        std::string imagePath = imagePaths[i];
        int imageLabel = imageLabels[i];

        std::cout << "processing: " << imagePath << std::endl;

        if(!mapContainsKey(folderImageMap,imageLabel))
        {
            folderImageMap[imageLabel] = imagePath;
        }

        // read image using OpenCV
        Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);

        // convert image from BGR to RGB
        // because Dlib used RGB format
        Mat imRGB;
        cvtColor(im, imRGB, COLOR_BGR2RGB);

        // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
        // Dlib's dnn module doesn't accept Dlib's cv_image template
        dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

        // detect faces in image
        std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
        // Now process each face we found
        for (int j = 0; j < faceRects.size(); j++) {

            // Find facial landmarks for each detected face
            full_object_detection landmarks = landmarkDetector(imDlib, faceRects[j]);

            // object to hold preProcessed face rectangle cropped from image
            matrix<rgb_pixel> face_chip;

            // original face rectangle is warped to 150x150 patch.
            // Same pre-processing was also performed during training.
            extract_image_chip(imDlib, get_face_chip_details(landmarks, 150, 0.25), face_chip);

            // Compute face descriptor using neural network defined in Dlib.
            // It is a 128D vector that describes the face in img identified by shape.
            matrix<float,0,1> faceDescriptor = net(face_chip);

            // add face descriptor and label for this face to
            // vectors faceDescriptors and faceLabels
            faceDescriptors.push_back(faceDescriptor);
            // add label for this face to vector containing labels corresponding to
            // vector containing face descriptors
            faceLabels.push_back(imageLabel);
        }
    }

    std::cout << "number of face descriptors " << faceDescriptors.size() << std::endl;
    std::cout << "number of face labels " << faceLabels.size() << std::endl;

}
// read descriptors saved on disk
void readDescriptors(const std::string& filename, std::vector<int>& faceLabels, std::vector<matrix<float,0,1>>& faceDescriptors, char separator = ';')
{
    std::ifstream file(filename.c_str(), std::ifstream::in);
    if (!file)
    {
        std::string error_message = "No valid input file was given, please check the given filename.";
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
        std::stringstream liness(line);
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
        dlib::matrix<float, 0, 1> faceDescriptor = dlib::mat(faceDescriptorVec);
        faceDescriptors.push_back(faceDescriptor);
    }
}
void loadTrainedData(const std::string& descriptorsFilePath, std::vector<int>&  faceLabels, std::vector<matrix<float,0,1>>& faceDescriptors,std::map<int, std::string>& labelNameMap,std::map<int,std::string>& folderImageMap )
{
    readDescriptors(descriptorsFilePath, faceLabels, faceDescriptors);

    std::vector<std::string> names;
    std::vector<int> labels;
    std::vector<std::string> imagePaths;
    std::vector<int> imageLabels;
    std::vector<std::string> folderNames;

    getFolderAndFiles(names,labels,imagePaths,labelNameMap,imageLabels);

    for (int i = 0; i < imagePaths.size(); i++) {
        std::string imagePath = imagePaths[i];
        int imageLabel = imageLabels[i];
        if (!mapContainsKey(folderImageMap, imageLabel)) {
            folderImageMap[imageLabel] = imagePath;
        }
    }
}

void test(std::vector<matrix<float,0,1>>& faceDescriptors, std::vector<int> faceLabels, std::map<int, std::string> labelNameMap,
          frontal_face_detector& faceDetector, const shape_predictor& landmarkDetector, anet_type& net, std::map<int,std::string>& folderImageMap)
{

    std::string faceTestFolder = "../resource/asnlib/publicdata/test-images";
    std::vector<std::string> subfolders, fileNames, symlinkNames;
    std::vector<int> tmpImageLabels;
    std::vector<std::string> imagePaths;

    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    listDirectory(faceTestFolder, subfolders, fileNames, symlinkNames);
    filterFiles(faceTestFolder, fileNames, imagePaths, "jpg", tmpImageLabels, 0);
    Dict celebs = generateLabelMap();

    int fileNameIndex = 0;
    // foreach filename lets find the closest celeb match
    for( const auto& imagePath : imagePaths)
    {
        std::cout << "testing image: " << imagePath << std::endl;
        Mat im = cv::imread(imagePath, cv::IMREAD_COLOR);
        // convert image from BGR to RGB
        // because Dlib used RGB format
        Mat imRGB = im.clone();
        cv::cvtColor(im, imRGB, cv::COLOR_BGR2RGB);

        // convert OpenCV image to Dlib's cv_image object, then to Dlib's matrix object
        // Dlib's dnn module doesn't accept Dlib's cv_image template
        dlib::matrix<dlib::rgb_pixel> imDlib(dlib::mat(dlib::cv_image<dlib::rgb_pixel>(imRGB)));

        // detect faces in image
        std::vector<dlib::rectangle> faceRects = faceDetector(imDlib);
        std::cout << faceRects.size() << " Faces Detected " << std::endl;
        std::string name;
        std::string celebName;

        // Now process each face we found
        for (int i = 0; i < faceRects.size(); i++)
        {
            // Find facial landmarks for each detected face
            full_object_detection landmarks = landmarkDetector(imDlib, faceRects[i]);

            // object to hold preProcessed face rectangle cropped from image
            matrix<rgb_pixel> face_chip;

            // original face rectangle is warped to 150x150 patch.
            // Same pre-processing was also performed during training.
            extract_image_chip(imDlib, get_face_chip_details(landmarks,150,0.25), face_chip);

            // Compute face descriptor using neural network defined in Dlib.
            // It is a 128D vector that describes the face in img identified by shape.
            matrix<float,0,1> faceDescriptorQuery = net(face_chip);

            // Find closest face enrolled to face found in frame
            int label;
            float minDistance;
            nearestNeighbor(faceDescriptorQuery, faceDescriptors, faceLabels, label, minDistance);

            // Name of recognized person/celeb from map
            if(label > -1) // if we have found a label it will greater than 0
            {
                name = labelNameMap[label];
                celebName = celebs[name];
                std::string imageCelebPath = folderImageMap[label];
                Mat imCeleb = cv::imread(imageCelebPath, cv::IMREAD_COLOR);

                imshow(fileNames[fileNameIndex], im);
                imshow(celebName, imCeleb);

                std::cout << "foldername: " << name << std::endl;
                std::cout << "celeb name:" << celebName << std::endl;
            }



        }

        fileNameIndex++;
    }



}

int main() {

    // Initialize face detector, facial landmarks detector and face recognizer
    std::string predictorPath, faceRecognitionModelPath;
    predictorPath = "../resource/lib/publicdata/models/shape_predictor_68_face_landmarks.dat";
    faceRecognitionModelPath = "../resource/lib/publicdata/models/dlib_face_recognition_resnet_model_v1.dat";
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(predictorPath) >> landmarkDetector;
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;

    std::map<int,std::string> folderImageMap;
    std::map<int, std::string> labelNameMap;
    std::vector<matrix<float,0,1>> faceDescriptors;
    std::vector<int> faceLabels;

    std::string descriptorsFilePath = "../descriptors.csv";
    if(fs::exists(descriptorsFilePath))
    {
        loadTrainedData(descriptorsFilePath,faceLabels,faceDescriptors, labelNameMap,folderImageMap);
    }
    else
    {
        // verify if we can use the trained data or perform the enrollment.
        enrollment(labelNameMap, faceDescriptors, faceLabels, faceDetector, landmarkDetector, net, folderImageMap);
        writeDescriptorsToDisk(faceDescriptors,faceLabels);
    }


    test(faceDescriptors,faceLabels,labelNameMap, faceDetector, landmarkDetector,net, folderImageMap);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
