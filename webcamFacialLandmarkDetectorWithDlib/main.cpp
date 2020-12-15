#include <iostream>
#include <string>
#include <vector>

#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include "renderFace.hpp"

using namespace dlib;


#define RESIZE_HEIGHT 320
#define SKIP_FRAMES 2
#define OPENCV_FACE_RENDER

void EnumerateCameras()
{
    std::string winName;
    cv::Mat frame;
    std::cout << "Searching for cameras IDs...";

    int camCount = 0;
    for(int idx=0; idx<10; idx++)
    {
        cv::VideoCapture cap(idx);       // open the camera
        if(cap.isOpened())           // check if we succeeded
        {
            std::cout << idx << "OK ";
            cap >> frame;
            winName = "A frame from camID: " + std::to_string(idx);
            imshow(winName,frame);  // display a frame from the current camera
            camCount++;
        }
    }

    std::cout << std::endl << camCount << " cam(s) available";
    std::cout << "Press a key...";
    cv::waitKey(0);

}


int main() {

    std::string winName("Fast Facial Landmark Detector");
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    //EnumerateCameras();

    // Create a VideoCapture object
    cv::VideoCapture cap(1);
    // Check if OpenCV is able to read feed from camera
    if (!cap.isOpened())
    {
        std::cerr << "Unable to connect to camera" << std::endl;
        return 1;
    }

    // Just a place holder. Actual value calculated after 100 frames.
    double fps = 30.0;

    // Get first frame and allocate memory.
    cv::Mat im;
    cap >> im;

    // We will use a fixed height image as input to face detector
    cv::Mat imSmall, imDisplay;
    float height = im.rows;
    // calculate resize scale
    float RESIZE_SCALE = height/RESIZE_HEIGHT;
    cv::Size size = im.size();

    // Load face detection and pose estimation models
    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor predictor;
    deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> predictor;

    // initiate the tickCounter
    double t = (double)cv::getTickCount();
    int count = 0;

    std::vector<rectangle> faces;
    // Grab and process frames until the main window is closed by the user.
    while(1)
    {
        if ( count == 0 ) {
            t = cv::getTickCount();
        }

        // Grab a frame
        cap >> im;
        // create imSmall by resizing image by resize scale
        cv::resize(im, imSmall, cv::Size(), 1.0/RESIZE_SCALE, 1.0/RESIZE_SCALE);
        // Change to dlib's image format. No memory is copied
        cv_image<bgr_pixel> cimgSmall(imSmall);
        cv_image<bgr_pixel> cimg(im);

        // Process frames at an interval of SKIP_FRAMES.
        // This value should be set depending on your system hardware
        // and camera fps.
        // To reduce computations, this value should be increased
        if ( count % SKIP_FRAMES == 0 )
        {
            // Detect faces
            faces = detector(cimgSmall);
        }

        // Find facial landmarks for each face.
        std::vector<full_object_detection> shapes;
        // Iterate over faces
        for (unsigned long i = 0; i < faces.size(); ++i)
        {
            // Since we ran face detection on a resized image,
            // we will scale up coordinates of face rectangle
            rectangle r(
                    (long)(faces[i].left() * RESIZE_SCALE),
                    (long)(faces[i].top() * RESIZE_SCALE),
                    (long)(faces[i].right() * RESIZE_SCALE),
                    (long)(faces[i].bottom() * RESIZE_SCALE)
            );
            // Find face landmarks by providing reactangle for each face
            full_object_detection shape = predictor(cimg, r);
            shapes.push_back(shape);
            // Draw facial landmarks
            renderFace(im, shape);
        }

        // Put fps at which we are processinf camera feed on frame
        cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);

        // Display it all on the screen
        cv::imshow(winName, im);
        // Wait for keypress
        char key = cv::waitKey(1);
        if (key == 27) // ESC
        {
            // If ESC is pressed, exit.
           break;
        }

        // increment frame counter
        count++;
        // calculate fps after each 100 frames are processed
        if(count == 100)
        {
            t = ((double)cv::getTickCount() - t)/cv::getTickFrequency();
            fps = 100.0/t;
            count = 0;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
