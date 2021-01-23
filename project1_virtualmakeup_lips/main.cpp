#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include "faceBlendCommon.hpp"

using namespace cv;
using namespace std;

// data structures
struct range{
    int start;
    int end;
};
enum class facialFeatureType { jawLine, leftEyebrow, rightEyebrow, noseBridge, lowerNose, leftEye, rightEye, upperLip, lowerLip, leftCheek, rightCheek };
struct dlibFacialFeature{
    facialFeatureType facialFeature;
    vector<range> points;
    Scalar color;
};

vector<dlibFacialFeature> facialFeatures;
void initialiseFacialFeatures();
void displayImageWithLandmarks(Mat& img, const vector<Point2f>& points);
void applyLipstick(const Mat& sourceImage,  const vector<Point2f>& points);
void applyBlush(const Mat& sourceImage,  const vector<Point2f>& points);
void Combined(Mat& img, const vector<Point2f>& points);

int main() {

    // load landmark detector
    // Landmark model location
    string PREDICTOR_PATH =  "../data/models/shape_predictor_68_face_landmarks.dat";

    // Get the face detector
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
    // The landmark detector is implemented in the shape_predictor class
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize(PREDICTOR_PATH) >> landmarkDetector;

    //Mat img = imread("../data/images/girl-no-makeup.jpg");
    Mat img = imread("../data/images/modelWithSmile.jpg");

    Mat imgCopy = img.clone();
    vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, img);

    initialiseFacialFeatures();
    displayImageWithLandmarks(imgCopy, points);

    //applyLipstick(img,points);
    //applyBlush(img,points);

    Combined(img, points);
    waitKey(0);

    destroyAllWindows();
    return 0;
}

void initialiseFacialFeatures()
{

    facialFeatures.push_back({ facialFeatureType::jawLine,  {{0,16}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::leftEyebrow,  {{17,21}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::rightEyebrow,  {{22,26}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::noseBridge,  {{27,30}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::lowerNose,  {{30,35}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::leftEye,  {{36,41}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::rightEye,  {{42,47}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::upperLip,  {{48,55}, {60,65}}, Scalar(255,255,0)});
    facialFeatures.push_back({ facialFeatureType::lowerLip,  {{48,49}, {54,61}, {64,68}}, Scalar(0,255,255)});
    facialFeatures.push_back({ facialFeatureType::leftCheek,  {{1,5}, {28,29}, {31,32}}, Scalar(255,0,0)});
    facialFeatures.push_back({ facialFeatureType::rightCheek,  {{12,16},{35,36}, {28,29}}, Scalar(0,255,0)});

}
void displayImageWithLandmarks(Mat& img, const vector<Point2f>& points)
{
    for(const auto& facialFeature : facialFeatures){
        if(facialFeature.facialFeature == facialFeatureType::rightCheek
            || facialFeature.facialFeature == facialFeatureType::leftCheek
            || facialFeature.facialFeature == facialFeatureType::upperLip
            || facialFeature.facialFeature == facialFeatureType::lowerLip )
        for(const auto& p : facialFeature.points){
            for(auto i = p.start; i < p.end; i++){

                std::cout << i << "( " << points[i].x << " , " << points[i].y << " )" << std::endl;
                cv::circle(img, points[i], 3, facialFeature.color, -1);
            }
        }
    }

    imshow("points of interest", img);
}
void applyBlush(const Mat& sourceImage,  const vector<Point2f>& points)
{
    Scalar cheekColor(208,208,242);
    vector<Point> leftCheekPoints;

    int leftCheek[7] = {1 , 2 , 3 ,4,5,  31,39};
    for(int i = 0; i < 7; i++){
        std::cout << i << "( " << points[leftCheek[i]].x << " , " << points[leftCheek[i]].y << " )" << std::endl;
        leftCheekPoints.emplace_back(static_cast<int>(points[leftCheek[i]].x), static_cast<int>(points[leftCheek[i]].y));
    }

    vector<Point> leftCheekPointsOrdered;
    convexHull(leftCheekPoints,leftCheekPointsOrdered);

    Mat maskLeftCheek = Mat::zeros(sourceImage.size(), sourceImage.type());
    fillConvexPoly(maskLeftCheek,leftCheekPointsOrdered,cheekColor);


    vector<Point> rightCheekPoints;
    int rightCheek[7] = { 12,13,14,15,16,35,42};
    for(int i = 0; i < 7; i++){
        std::cout << i << "( " << points[rightCheek[i]].x << " , " << points[rightCheek[i]].y << " )" << std::endl;
        rightCheekPoints.emplace_back(static_cast<int>(points[rightCheek[i]].x), static_cast<int>(points[rightCheek[i]].y));
    }

    vector<Point> rightCheekPointsOrdered;
    convexHull(rightCheekPoints,rightCheekPointsOrdered);

    Mat maskRightCheek = Mat::zeros(sourceImage.size(), sourceImage.type());
    fillConvexPoly(maskRightCheek,rightCheekPointsOrdered,cheekColor);

    Mat mask = Mat::zeros(sourceImage.size(), sourceImage.type());
    addWeighted(maskLeftCheek,1, maskRightCheek,1,0,mask);

    auto erodeKernalSize = Size(13,13);

    auto kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE,erodeKernalSize, Point(-1,-1));
    erode(mask,mask,kernel);

    auto blurKernalSize = Size(3,3);
    GaussianBlur(mask,mask,blurKernalSize,0,0);

    Mat result;
    auto alpha = 0.04;
    auto beta = 1.0 - alpha;

    addWeighted(mask,alpha,sourceImage,beta,0,result);

    Mat result2;
    hconcat(sourceImage,result,result2);

    imshow("leftCheek", maskLeftCheek);
    imshow("rightCheek", maskRightCheek);
    imshow("cheek mask combine", mask);
    imshow("model with blush", result);
    imshow("Blush before and after", result2);


}

void applyLipstick(const Mat& sourceImage,  const vector<Point2f>& points)
{
    Scalar lipColour(255,0,0);

    vector<Point> upperLipPoints;
    int upper[10] = {48,49,50,51,52,53,54,61,62,63};
    for(int i : upper){
        upperLipPoints.emplace_back(static_cast<int>(points[i].x), static_cast<int>(points[i].y));
    }

    vector<Point> upperLipPointsOrdered;
    convexHull(upperLipPoints,upperLipPointsOrdered);

    Mat maskUpper = Mat::zeros(sourceImage.size(), sourceImage.type());
    fillConvexPoly(maskUpper,upperLipPointsOrdered,lipColour);

    vector<Point> lowerLipPoints;
    int lower[12] = { 48,54, 55,56,57,58,59,60,64,65,66,67};
    for(int i : lower){
        lowerLipPoints.emplace_back(static_cast<int>(points[i].x), static_cast<int>(points[i].y));
    }

    vector<Point> lowerLipPointsOrdered;
    convexHull(lowerLipPoints,lowerLipPointsOrdered);

    Mat maskLower = Mat::zeros(sourceImage.size(), sourceImage.type());
    fillConvexPoly(maskLower,lowerLipPointsOrdered,lipColour);

    Mat mask = Mat::zeros(sourceImage.size(), sourceImage.type());
    addWeighted(maskUpper,1, maskLower,1,0,mask);

    auto lipKernalSize = Size(3,3);
    auto lipKernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE,lipKernalSize, Point(-1,-1));

    erode(mask,mask,lipKernel);
    GaussianBlur(mask,mask,lipKernalSize,0,0);


    Mat result;
    auto alpha = 0.1;
    auto beta = 1.0 - alpha;
    addWeighted(mask,alpha,sourceImage,beta,0,result);

    Mat result2;
    hconcat(sourceImage,result,result2);

    imshow("lowerlip", maskLower);
    imshow("upperlip", maskUpper);
    imshow("lip mask combine", mask);
    imshow("model with lipstick", result);
    imshow("lipstick before and after", result2);

}

void Combined(Mat& img, const vector<Point2f>& points)
{
    auto imgWithMakeup = img.clone();

    Scalar lipColour(255,0,0);

    // upper lip
    vector<Point> upperLipPoints;
    int upper[10] = {48,49,50,51,52,53,54,61,62,63};
    for(int i : upper){
        upperLipPoints.emplace_back(static_cast<int>(points[i].x), static_cast<int>(points[i].y));
    }

    vector<Point> upperLipPointsOrdered;
    convexHull(upperLipPoints,upperLipPointsOrdered);

    Mat maskUpper = Mat::zeros(imgWithMakeup.size(), imgWithMakeup.type());
    fillConvexPoly(maskUpper,upperLipPointsOrdered,lipColour);

    // lower lip
    vector<Point> lowerLipPoints;
    int lower[12] = { 48,54, 55,56,57,58,59,60,64,65,66,67};
    for(int i : lower){
        lowerLipPoints.emplace_back(static_cast<int>(points[i].x), static_cast<int>(points[i].y));
    }

    vector<Point> lowerLipPointsOrdered;
    convexHull(lowerLipPoints,lowerLipPointsOrdered);

    Mat maskLower = Mat::zeros(imgWithMakeup.size(), imgWithMakeup.type());
    fillConvexPoly(maskLower,lowerLipPointsOrdered,lipColour);

    // combine masks
    Mat mask = Mat::zeros(imgWithMakeup.size(), imgWithMakeup.type());
    addWeighted(maskUpper,1, maskLower,1,0,mask);

    // to make the mask fit with the image without hard edges we need
    // shrink it (erode) and blur the image to remove hard lines
    auto lipKernalSize = Size(3,3);
    auto lipKernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE,lipKernalSize, Point(-1,-1));

    erode(mask,mask,lipKernel);
    GaussianBlur(mask,mask,lipKernalSize,0,0);

    // blend the mask to the image
    auto alpha = 0.1;
    auto beta = 1.0 - alpha;
    addWeighted(mask,alpha,img,beta,0,imgWithMakeup);

    // combine the original image with the image with that has makeup
    Mat imageWithLipMakeUp;
    hconcat(img,imgWithMakeup,imageWithLipMakeUp);

    // display image
    imshow("image with lip makeup", imageWithLipMakeUp);


    // Feature 2: blush applied
    //Scalar cheekColor(208,208,242);
    Scalar cheekColor(0,0,255);
    vector<Point> leftCheekPoints;

    int leftCheek[7] = {1,2,3,4,5,31,39};
    for(int i : leftCheek){
        leftCheekPoints.emplace_back(static_cast<int>(points[i].x), static_cast<int>(points[i].y));
    }
    vector<Point> leftCheekPointsOrdered;
    convexHull(leftCheekPoints,leftCheekPointsOrdered);

    Mat maskLeftCheek = Mat::zeros(imgWithMakeup.size(), imgWithMakeup.type());
    fillConvexPoly(maskLeftCheek,leftCheekPointsOrdered,cheekColor);

    vector<Point> rightCheekPoints;
    int rightCheek[7] = {12,13,14,15,16,35,42};
    for(int i : rightCheek){
        rightCheekPoints.emplace_back(static_cast<int>(points[i].x), static_cast<int>(points[i].y));
    }

    vector<Point> rightCheekPointsOrdered;
    convexHull(rightCheekPoints,rightCheekPointsOrdered);

    Mat maskRightCheek = Mat::zeros(imgWithMakeup.size(), imgWithMakeup.type());
    fillConvexPoly(maskRightCheek,rightCheekPointsOrdered,cheekColor);

    // combine masks
    Mat cheekMask = Mat::zeros(imgWithMakeup.size(), imgWithMakeup.type());
    addWeighted(maskLeftCheek,1, maskRightCheek,1,0,cheekMask);

    // to make the mask fit with the image without hard edges we need
    // shrink it (erode) and blur the image to remove hard lines
    auto erodeKernalSize = Size(30,30);
    auto cheekKernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE,erodeKernalSize, Point(-1,-1));
    erode(cheekMask,cheekMask,cheekKernel);

    auto blurCheekKernalSize = Size(13,13);
    GaussianBlur(cheekMask,cheekMask,blurCheekKernalSize,0,0);

    imshow("erode mask", cheekMask);

    // blend the mask to the image
    auto cheekAlpha = 0.04;
    auto cheekBeta = 1.0 - cheekAlpha;
    addWeighted(cheekMask,cheekAlpha,imgWithMakeup,cheekBeta,0,imgWithMakeup);

    // combine the original image with the image with that has makeup
    Mat imageWithLipAndBlushMakeUp;
    hconcat(img,imgWithMakeup,imageWithLipAndBlushMakeUp);

    // display image
    imshow("combined makeup", imageWithLipAndBlushMakeUp);
    waitKey(0);

}