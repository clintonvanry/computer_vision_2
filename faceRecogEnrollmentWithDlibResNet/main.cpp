#include <iostream>
#include <cmath>
#include <map>
#include <filesystem>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace dlib;

namespace fs = std::filesystem;

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

template<typename T>
void printVector(std::vector<T>& vec)
{
    for (int i = 0; i < vec.size(); i++) {
        std::cout << i << " " << vec[i] << "; ";
    }
    std::cout << std::endl;
}

void listDirectory(std::string dirName, std::vector<std::string>& folderNames, std::vector<std::string>& fileNames, std::vector<std::string>& symlinkNames)
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

int main() {
    // Initialize face detector, facial landmarks detector and face recognizer
    std::string predictorPath, faceRecognitionModelPath;
    predictorPath = "../data/models/shape_predictor_68_face_landmarks.dat";
    faceRecognitionModelPath = "../data/models/dlib_face_recognition_resnet_model_v1.dat";
    frontal_face_detector faceDetector = get_frontal_face_detector();
    shape_predictor landmarkDetector;
    deserialize(predictorPath) >> landmarkDetector;
    anet_type net;
    deserialize(faceRecognitionModelPath) >> net;

    // Now let's prepare our training data
    // data is organized assuming following structure
    // faces folder has subfolders.
    // each subfolder has images of a person
    std::string faceDatasetFolder = "../data/images/faces";
    std::vector<std::string> subfolders, fileNames, symlinkNames;
    // fileNames and symlinkNames are useless here
    // as we are looking for sub-directories only
    listDirectory(faceDatasetFolder, subfolders, fileNames, symlinkNames);

    // names: vector containing names of subfolders i.e. persons
    // labels: integer labels assigned to persons
    // labelNameMap: dict containing (integer label, person name) pairs
    std::vector<std::string> names;
    std::vector<int> labels;
    std::map<int, std::string> labelNameMap;
    // add -1 integer label for un-enrolled persons
    names.push_back("unknown");
    labels.push_back(-1);

    // imagePaths: vector containing imagePaths
    // imageLabels: vector containing integer labels corresponding to imagePaths
    std::vector<std::string> imagePaths;
    std::vector<int> imageLabels;


    // variable to hold any subfolders within person subFolders
    std::vector<std::string> folderNames;

    // iterate over all subFolders within faces folder
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
        filterFiles(subfolders[i], fileNames, imagePaths, "jpg", imageLabels, i);
    }

    // process training data
    // We will store face descriptors in vector faceDescriptors
    // and their corresponding labels in vector faceLabels
    std::vector<matrix<float,0,1>> faceDescriptors;
    // std::vector<cv_image<bgr_pixel> > imagesFaceTrain;
    std::vector<int> faceLabels;

    // iterate over images
    for (int i = 0; i < imagePaths.size(); i++)
    {
        std::string imagePath = imagePaths[i];
        int imageLabel = imageLabels[i];

        std::cout << "processing: " << imagePath << std::endl;

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
        std::cout << faceRects.size() << " Face(s) Found" << std::endl;
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

    // write label name map to disk
    const std::string labelNameFile = "label_name.txt";
    std::ofstream of;
    of.open (labelNameFile);
    for (int m = 0; m < names.size(); m++) {
        of << names[m];
        of << ";";
        of << labels[m];
        of << "\n";
    }
    of.close();

    // write face labels and descriptor to disk
    // each row of file descriptors.csv has:
    // 1st element as face label and
    // rest 128 as descriptor values
    const std::string descriptorsPath = "descriptors.csv";
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

    return 0;
}
