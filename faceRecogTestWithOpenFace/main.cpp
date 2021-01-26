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


int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
