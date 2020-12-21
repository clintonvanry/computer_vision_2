#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

// Alpha blending using multiply and add functions
void blend(const Mat& alpha, const Mat& foreground,const  Mat& background, Mat& outImage)
{
    Mat fore, back;
    multiply(alpha, foreground, fore);
    multiply(Scalar::all(1.0)-alpha, background, back);
    add(fore, back, outImage);


}

// Alpha Blending using direct pointer access
void alphaBlendDirectAccess(const Mat& alpha, const Mat& foreground,const Mat& background, Mat& outImage)
{

    int numberOfPixels = foreground.rows * foreground.cols * foreground.channels();

    auto* fptr = reinterpret_cast<float*>(foreground.data);
    auto* bptr = reinterpret_cast<float*>(background.data);
    auto* aptr = reinterpret_cast<float*>(alpha.data);
    auto* outImagePtr = reinterpret_cast<float*>(outImage.data);

    int j;
    for ( j = 0; j < numberOfPixels; ++j, outImagePtr++, fptr++, aptr++, bptr++)
    {
        *outImagePtr = (*fptr)*(*aptr) + (*bptr)*(1 - *aptr);
    }

}

int main() {

    // Read in the png foreground asset file that contains both rgb and alpha information
    Mat foreGroundImage = imread("../data/images/foreGroundAssetLarge.png", -1);
    Mat bgra[4];
    split(foreGroundImage, bgra);//split png foreground

    // Save the foreground RGB content into a single Mat
    std::vector<Mat> foregroundChannels;
    foregroundChannels.push_back(bgra[0]);
    foregroundChannels.push_back(bgra[1]);
    foregroundChannels.push_back(bgra[2]);
    Mat foreground = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(foregroundChannels, foreground);
    imshow("foreground without alpha", foreground);

    // Save the alpha information into a single Mat
    std::vector<Mat> alphaChannels;
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    alphaChannels.push_back(bgra[3]);
    Mat alpha = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    merge(alphaChannels, alpha);
    imshow("alpha only", alpha);

    Mat copyWithMask = Mat::zeros(foreGroundImage.size(), CV_8UC3);
    foreground.copyTo(copyWithMask, bgra[3]);
    imshow("copyWithMask", copyWithMask);
    imshow("alpha",  bgra[3] > 0 );
    imshow("original foreground", foreGroundImage);


    // Read background image
    Mat background = imread("../data/images/backGroundLarge.jpg");
    imshow("original background", background);
    waitKey(0);

    // Convert Mat to float data type
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0/255); // keeps the alpha values between 0 and 1

    // Number of iterations to average the performance over
    int numOfIterations = 1; //1000;

    // Alpha blending using functions multiply and add
    Mat outImage= Mat::zeros(foreground.size(), foreground.type());
    auto t = (double)getTickCount();
    for (int i=0; i<numOfIterations; i++) {
        blend(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "Time for alpha blending using multiply & add function : " << t*1000/numOfIterations << " milliseconds" << std::endl;

    // Alpha blending using direct Mat access with for loop
    outImage = Mat::zeros(foreground.size(), foreground.type());
    t = (double)getTickCount();
    for (int i=0; i<numOfIterations; i++) {
       alphaBlendDirectAccess(alpha, foreground, background, outImage);
    }
    t = ((double)getTickCount() - t)/getTickFrequency();
    std::cout << "Time for alpha blending using alphaBlendDirectAccess : " << t*1000/numOfIterations << " milliseconds" << std::endl;

    imshow("alpha blended image", outImage/255);
    waitKey(0);

    return 0;
}
