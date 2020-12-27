#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace dlib;

#ifndef M_PI
#define M_PI 3.14159
#endif

Mat barrel(Mat &src, float k)
{
    int w = src.cols;
    int h = src.rows;

    // Meshgrid of destiation image
    Mat Xd = cv::Mat::zeros(src.size(), CV_32F);
    Mat Yd = cv::Mat::zeros(src.size(), CV_32F);

    float Xu, Yu;
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            // Normalize x and y
            Xu = ( (float) x / w )- 0.5;
            Yu = ( (float) y / h )- 0.5;

            // Radial distance from center
            float r = sqrt(Xu * Xu + Yu * Yu);

            // Implementing the following equation
            // dr = k * r * cos(pi*r)
            float dr = k * r * cos(M_PI * r);

            // Outside the maximum radius dr is set to 0
            if (r > 0.5) dr = 0;

            // Remember we need to provide inverse mapping to remap
            // Hence the negative sign before dr
            float rn = r - dr;

            // Applying the distortion on the grid
            // Back to un-normalized coordinates
            Xd.at<float>(y,x) =  w * (rn * Xu / r + 0.5);
            Yd.at<float>(y,x) = h * (rn * Yu / r + 0.5);

        }
    }

    // Interpolation of points
    Mat dst;
    cv::remap( src, dst, Xd, Yd, INTER_CUBIC, BORDER_CONSTANT, Scalar(0,0, 0) );
    return dst;
}

int main() {
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor pose_model;

    std::string modelPath = "../data/models/shape_predictor_68_face_landmarks.dat";

    //std::string filename = "../data/images/hillary-clinton.jpg";
    std::string filename = "../data/images/model.jpg";

    float bulge_amount = .5;

    int radius = 30;

    deserialize(modelPath) >> pose_model;

    // load a nice picture
    auto t = (double)cv::getTickCount();

    Mat src = imread(filename);

    cv_image<bgr_pixel> cimg(src);
    std::vector<dlib::rectangle> faces;

    faces = detector(cimg);

    // Find the pose of each face.
    full_object_detection landmarks;

    // Find the landmark points using DLIB Facial landmarks detector
    landmarks = pose_model(cimg, faces[0]);

    // Find the roi for left and right Eye
    Rect roiEyeRight ( (landmarks.part(43).x()-radius)
            , (landmarks.part(43).y()-radius)
            , ( landmarks.part(46).x() - landmarks.part(43).x() + 2*radius )
            , ( landmarks.part(47).y() - landmarks.part(43).y() + 2*radius ) );
    Rect roiEyeLeft ( (landmarks.part(37).x()-radius)
            , (landmarks.part(37).y()-radius)
            , ( landmarks.part(40).x() - landmarks.part(37).x() + 2*radius )
            , ( landmarks.part(41).y() - landmarks.part(37).y() + 2*radius ) );

    // Find the patch and apply the transformation
    Mat eyeRegion, output;
    output = src.clone();
    src(roiEyeRight).copyTo(eyeRegion);
    eyeRegion = barrel(eyeRegion, bulge_amount);
    eyeRegion.copyTo(output(roiEyeRight));

    src(roiEyeLeft).copyTo(eyeRegion);
    eyeRegion = barrel(eyeRegion, bulge_amount);
    eyeRegion.copyTo(output(roiEyeLeft));

    std::cout << "time taken " << ((double)cv::getTickCount() - t)/cv::getTickFrequency() << std::endl;

    // imshow("distorted",dst);
    imshow("final",output);
    waitKey(0);
    return 0;
}
