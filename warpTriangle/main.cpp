#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;

// Warps and alpha blends triangular regions from img1 and img2 to img
void warpTriangle(Mat &img1, Mat &img2, const std::vector<Point2f>& tri1, const std::vector<Point2f>& tri2)
{
    // Find bounding rectangle for each triangle
    Rect r1 = boundingRect(tri1);
    Rect r2 = boundingRect(tri2);

    // Crop the input image to the bounding box of input triangle
    Mat img1Cropped;
    img1(r1).copyTo(img1Cropped);

    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> tri1Cropped, tri2Cropped;
    std::vector<Point> tri2CroppedInt;
    for(int i = 0; i < 3; i++)
    {
        tri1Cropped.emplace_back( tri1[i].x - r1.x, tri1[i].y -  r1.y );
        tri2Cropped.emplace_back( tri2[i].x - r2.x, tri2[i].y - r2.y );

        // fillConvexPoly needs a vector of Point and not Point2f
        tri2CroppedInt.emplace_back((int)(tri2[i].x - r2.x), (int)(tri2[i].y - r2.y) );

    }

    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( tri1Cropped, tri2Cropped );

    // Apply the Affine Transform just found to the src image
    Mat img2Cropped = Mat::zeros(r2.height, r2.width, img1Cropped.type());
    warpAffine( img1Cropped, img2Cropped, warpMat, img2Cropped.size(), INTER_LINEAR, BORDER_REFLECT_101);

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
    fillConvexPoly(mask, tri2CroppedInt, Scalar(1.0, 1.0, 1.0), 16, 0);

    // Copy triangular region of the rectangular patch to the output image
    multiply(img2Cropped,mask, img2Cropped);
    multiply(img2(r2), Scalar(1.0,1.0,1.0) - mask, img2(r2));
    add(img2(r2),img2Cropped, img2(r2));


}

int main() {
    // Read input image and convert to float
    Mat imgIn = imread("../data/images/kingfisher.jpg");

    // Convert to floating point image in the range 0 to 1.
    imgIn.convertTo(imgIn, CV_32FC3, 1/255.0);

    // Create white output image the same size and type of input image
    Mat imgOut = Mat::ones(imgIn.size(), imgIn.type());
    imgOut = Scalar(1.0,1.0,1.0);

    // Input triangle
    std::vector <Point2f> triIn;
    triIn.emplace_back(Point2f(360,50));
    triIn.emplace_back(Point2f(60,100));
    triIn.emplace_back(Point2f(300,400));

    // Output triangle
    std::vector <Point2f> triOut;
    triOut.emplace_back(Point2f(400,200));
    triOut.emplace_back(Point2f(160,270));
    triOut.emplace_back(Point2f(400,400));

    // Warp all pixels inside input triangle to output triangle
    warpTriangle(imgIn, imgOut, triIn, triOut);

    // Draw triangle on the input and output image.

    // Convert back to uint because OpenCV antialiasing
    // does not work on image of type CV_32FC3

    imgIn.convertTo(imgIn, CV_8UC3, 255.0);
    imgOut.convertTo(imgOut, CV_8UC3, 255.0);

    // Draw triangle using this color
    Scalar color = Scalar(255, 150, 0);

    // cv::poly lines needs vector of type Point and not Point2f
    std::vector <Point> triInInt, triOutInt;
    for(int i=0; i < 3; i++)
    {
        triInInt.emplace_back(Point(triIn[i].x,triIn[i].y));
        triOutInt.emplace_back(Point(triOut[i].x,triOut[i].y));
    }

    // Draw triangles in input and output images
    int lineWidth = 2;
    polylines(imgIn, triInInt, true, color, lineWidth, LINE_AA);
    polylines(imgOut, triOutInt, true, color, lineWidth, LINE_AA);

    // Display and save input and output images.
    namedWindow("Input");
    imshow("Input", imgIn);
    imwrite("results/kingfisherInputTriangle.jpg", imgIn);

    namedWindow("Output");
    imshow("Output", imgOut);
    imwrite("results/kingfisherOutputTriangle.jpg", imgOut);

    waitKey(0);
    return 0;
}
