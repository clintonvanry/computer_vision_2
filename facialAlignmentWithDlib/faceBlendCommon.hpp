/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This program is distributed WITHOUT ANY WARRANTY to the
 Plus and Premium membership students of the online course
 titled "Computer Visionfor Faces" by Satya Mallick for
 personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com

 */

#ifndef BIGVISION_faceBlendCommon_HPP_
#define BIGVISION_faceBlendCommon_HPP_


#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace cv;
using namespace std;

#ifndef M_PI
  #define M_PI 3.14159
#endif


// Constrains points to be inside boundary
[[maybe_unused]] void constrainPoint(Point2f &p, const Size& sz)
{
  p.x = min(max( (double)p.x, 0.0), (double)(sz.width - 1));
  p.y = min(max( (double)p.y, 0.0), (double)(sz.height - 1));

}

// Converts Dlib landmarks into a vector for Point2f
void dlibLandmarksToPoints(dlib::full_object_detection &landmarks, vector<Point2f>& points)
{
  // Loop over all landmark points
  for (int i = 0; i < landmarks.num_parts(); i++)
  {
    Point2f pt(landmarks.part(i).x(), landmarks.part(i).y());
    points.push_back(pt);
  }
}

// Compute similarity transform given two pairs of corresponding points.
// OpenCV requires 3 points for calculating similarity matrix.
// We are hallucinating the third point.
void similarityTransform(std::vector<cv::Point2f>& inPoints, std::vector<cv::Point2f>& outPoints, cv::Mat &tform)
{

  double s60 = sin(60 * M_PI / 180.0);
  double c60 = cos(60 * M_PI / 180.0);

  vector <Point2f> inPts = inPoints;
  vector <Point2f> outPts = outPoints;

  // Placeholder for the third point.
  inPts.emplace_back(0,0);
  outPts.emplace_back(0,0);

  // The third point is calculated so that the three points make an equilateral triangle
  inPts[2].x =  c60 * (inPts[0].x - inPts[1].x) - s60 * (inPts[0].y - inPts[1].y) + inPts[1].x;
  inPts[2].y =  s60 * (inPts[0].x - inPts[1].x) + c60 * (inPts[0].y - inPts[1].y) + inPts[1].y;

  outPts[2].x =  c60 * (outPts[0].x - outPts[1].x) - s60 * (outPts[0].y - outPts[1].y) + outPts[1].x;
  outPts[2].y =  s60 * (outPts[0].x - outPts[1].x) + c60 * (outPts[0].y - outPts[1].y) + outPts[1].y;

  // Now we can use estimateRigidTransform for calculating the similarity transform.
  tform = cv::estimateAffinePartial2D(inPts, outPts);
}

// Normalizes a facial image to a standard size given by outSize.
// The normalization is done based on Dlib's landmark points passed as pointsIn
// After the normalization the left corner of the left eye is at (0.3 * w, h/3 )
// and the right corner of the right eye is at ( 0.7 * w, h / 3) where w and h
// are the width and height of outSize.
void normalizeImagesAndLandmarks(const Size& outSize, Mat &imgIn, Mat &imgOut, vector<Point2f>& pointsIn, vector<Point2f>& pointsOut)
{
  int h = outSize.height;
  int w = outSize.width;


  vector<Point2f> eyecornerSrc;
  if (pointsIn.size() == 68)
  {
    // Get the locations of the left corner of left eye
    eyecornerSrc.push_back(pointsIn[36]);
    // Get the locations of the right corner of right eye
    eyecornerSrc.push_back(pointsIn[45]);
  }
  else if(pointsIn.size() == 5)
  {
    // Get the locations of the left corner of left eye
    eyecornerSrc.push_back(pointsIn[2]);
    // Get the locations of the right corner of right eye
    eyecornerSrc.push_back(pointsIn[0]);
  }

  vector<Point2f> eyecornerDst;
  // Location of the left corner of left eye in normalized image.
  eyecornerDst.emplace_back( 0.3*w, h/3);
  // Location of the right corner of right eye in normalized image.
  eyecornerDst.emplace_back( 0.7*w, h/3);

  // Calculate similarity transform
  Mat tform;
  similarityTransform(eyecornerSrc, eyecornerDst, tform);

  // Apply similarity transform to input image
  imgOut = Mat::zeros(h, w, CV_32FC3);
  warpAffine(imgIn, imgOut, tform, imgOut.size());

  // Apply similarity transform to landmarks
  transform( pointsIn, pointsOut, tform);

}


// Compare dlib rectangle
bool rectAreaComparator(dlib::rectangle &r1, dlib::rectangle &r2)
{ return r1.area() < r2.area(); }


vector<Point2f> getLandmarks(dlib::frontal_face_detector &faceDetector, dlib::shape_predictor &landmarkDetector, Mat &img, float FACE_DOWNSAMPLE_RATIO = 1 )
{

  vector<Point2f> points;

  Mat imgSmall;
  cv::resize(img, imgSmall, cv::Size(), 1.0/FACE_DOWNSAMPLE_RATIO, 1.0/FACE_DOWNSAMPLE_RATIO);

  // Convert OpenCV image format to Dlib's image format
  dlib::cv_image<dlib::bgr_pixel> dlibIm(img);
  dlib::cv_image<dlib::bgr_pixel> dlibImSmall(imgSmall);


  // Detect faces in the image
  std::vector<dlib::rectangle> faceRects = faceDetector(dlibImSmall);

  if(!faceRects.empty())
  {
    // Pick the biggest face
      dlib::rectangle rect;
      rect = *std::max_element(faceRects.begin(), faceRects.end(), rectAreaComparator);

    dlib::rectangle scaledRect(
                    static_cast<long>((rect.left() * FACE_DOWNSAMPLE_RATIO)),
                    static_cast<long>((rect.top() * FACE_DOWNSAMPLE_RATIO)),
                    static_cast<long>((rect.right() * FACE_DOWNSAMPLE_RATIO)),
                    static_cast<long>((rect.bottom() * FACE_DOWNSAMPLE_RATIO))
                    );

    dlib::full_object_detection landmarks = landmarkDetector(dlibIm, scaledRect);
    dlibLandmarksToPoints(landmarks, points);
  }

  return points;

}

#endif // BIGVISION_faceBlendCommon_HPP_
