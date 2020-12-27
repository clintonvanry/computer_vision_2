#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "faceBlendCommon.hpp"

using namespace cv;
using namespace dlib;

void readFileNames(const std::string& dirName, std::vector<std::string>& imageFnames)
{
    std::string imgExt = ".jpg";

    const std::filesystem::path directory(dirName);
    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if(entry.path().extension() == imgExt){
            imageFnames.emplace_back(entry.path().string());
        }
    }
}

int main() {

    // Get the face detector
    frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

    // The landmark detector is implemented in the shape_predictor class
    shape_predictor landmarkDetector;

    // Load the landmark model
    std::cout << "Load the landmark model" << std::endl;
    deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

    // Directory containing images.
    std::string dirName = "../data/images/presidents/";

    // Read images in the directory
    std::cout << "Read images in the directory" << std::endl;
    std::vector<std::string> imageNames, ptsNames;
    readFileNames(dirName, imageNames);

    // Vector of vector of points for all image landmarks.
    std::vector<std::vector<Point2f> > allPoints;

    // Read images and perform landmark detection.
    std::vector<Mat> images;
    for(auto & imageName : imageNames)
    {
        Mat img = imread(imageName);
        std::cout << img.channels() << std::endl;
        if(!img.data)
        {
            std::cout << "image " << imageName << " not read properly" << std::endl;
        }
        else
        {
            std::cout << "Landmark detection for image: " << imageName << std::endl;
            std::vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, img,2);
            if (!points.empty())
            {
                allPoints.push_back(points);
                img.convertTo(img, CV_32FC3, 1/255.0);
                images.push_back(img);
            }
        }
    }

    int numImages = images.size();

    // Space for normalized images and points.
    std::vector<Mat> imagesNorm;
    std::vector<std::vector<Point2f> > pointsNorm;

    // Space for average landmark points
    std::vector<Point2f> pointsAvg(allPoints[0].size());

    // Dimensions of output image
    Size size(600,600);

    // 8 Boundary points for Delaunay Triangulation
    std::cout << "8 Boundary points for Delaunay Triangulation" << std::endl;
    std::vector <Point2f> boundaryPts;
    getEightBoundaryPoints(size, boundaryPts);

    // Warp images and transform landmarks to output coordinate system,
    // and find average of transformed landmarks.
    std::cout << "Warp images and transform landmarks to output coordinate system and find average of transformed landmarks." << std::endl;
    for(size_t i = 0; i < images.size(); i++)
    {
        std::vector <Point2f> points = allPoints[i];

        Mat img;
        normalizeImagesAndLandmarks(size,images[i],img, points, points);

        // Calculate average landmark locations
        for ( size_t j = 0; j < points.size(); j++)
        {
            pointsAvg[j] += points[j] * ( 1.0 / numImages);
        }

        // Append boundary points. Will be used in Delaunay Triangulation
        for (const auto & boundaryPt : boundaryPts)
        {
            points.push_back(boundaryPt);
        }

        pointsNorm.push_back(points);
        imagesNorm.push_back(img);

    }

    // Append boundary points to average points.
    for (const auto & boundaryPt : boundaryPts)
    {
        pointsAvg.push_back(boundaryPt);
    }

    // Calculate Delaunay triangles
    std::cout << "Calculate Delaunay triangles" << std::endl;
    Rect rect(0, 0, size.width, size.height);
    std::vector<std::vector<int> > dt;
    calculateDelaunayTriangles(rect, pointsAvg, dt);

    // Space for output image
    Mat output = Mat::zeros(size, CV_32FC3);

    // Warp input images to average image landmarks
    std::cout << "Warp input images to average image landmarks" << std::endl;
    for(size_t i = 0; i < numImages; i++)
    {
        Mat img;
        std::cout << "Warp input images to average image landmarks for image number: " << i << std::endl;
        warpImage(imagesNorm[i],img, pointsNorm[i], pointsAvg, dt);
        // Add image intensities for averaging
        output = output + img;

    }

    // Divide by numImages to get average
    std::cout << "Divide by numImages to get average" << std::endl;
    output = output / (double)numImages;

    // Display result
    std::cout << "Display result" << std::endl;
    imshow("image", output);
    waitKey(0);

    destroyAllWindows();

    return 0;

}
