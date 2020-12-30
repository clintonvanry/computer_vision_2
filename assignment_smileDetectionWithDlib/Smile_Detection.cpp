#include <iostream>
#include <string>
#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define RESIZE_HEIGHT 240
#define FACE_DOWNSAMPLE_RATIO_DLIB 1

// forward declares

bool smile_detector(const dlib::cv_image<dlib::bgr_pixel> &cimg, const dlib::rectangle &face, const dlib::full_object_detection &landmarks);


/*
 * Task
You will have to complete the smile_detector function.
In the function, you will receive an image and you have to write the logic for smile detection and return a boolean variable indicating
whether the frame has a smiling person or not.

We have provided the supporting code for keeping track of the smiling frames and we will use it to compare with the ground truth.
If the percentage of overlap is high enough, you will obtain full marks.

Finally, we also write the output video in a avi file which you can
download and see from the jupyter dashboard whether the smile is getting detected correctly.

You can see the output video to check whether the code detects smile or not and help to debug your code.
You will have a total of 5 attempts. This assignment carries 30 marks.

Some hints:
Find the lip and/or jaw coordinates using the facial landmarks.
For a person to be smiling, the ratio of width of the lip and the jaw width should be high.
Return True if a smile is detected, else return False.

TODO : Complete the smile detector function
You need to apply the face detector and shape_predictor to the input image to get the landmarks.
Then, Use the landmarks and come up with a logic so that the smile is detected for each frame.
Finally, override the variable isSmiling to True if smile is detected.
You can explain your logic in the next cell.

Solution Logic
Write your logic here for detecting if the face has a smile or not. The below function accepts an image along with face rectangle and landmarks.
HINT: Try to do it on a single image instead of the i

example output
Processed 0 frames
Smile detected in 0 number of frames

Processed 50 frames
Smile detected in 39 number of frames
 */


bool smile_detector(const dlib::cv_image<dlib::bgr_pixel> &cimg, \
					const dlib::rectangle &face, \
					const dlib::full_object_detection &landmarks)
{
    // Return true if a smile is detected, else return false
    bool isSmiling = false;

    ///
    /// YOUR CODE HERE
    ///
    cv::Point2f leftMouthCorner = cv::Point2f (landmarks.part(48).x(), landmarks.part(48).y());
    cv::Point2f rightMouthCorner = cv::Point2f(landmarks.part(54).x(), landmarks.part(54).y());
    cv::Point2f leftJawCorner = cv::Point2f(landmarks.part(3).x(), landmarks.part(3).y());
    cv::Point2f rightJawCorner = cv::Point2f(landmarks.part(13).x(), landmarks.part(13).y());

    auto widthOfMouth = std::abs( leftMouthCorner.x - rightMouthCorner.x);
    auto widthOfJaw = std::abs(leftJawCorner.x - rightJawCorner.x);
    auto ratio = (widthOfMouth /widthOfJaw);

    if( ratio > 0.447){
        isSmiling = true;
    }

    std::cout << " ratio: " << ratio << std::endl;

    return isSmiling;
}

int main(){

    // initialize dlib's face detector (HOG-based) and then create
    // the facial landmark predictor
    std::cout << "[INFO] loading facial landmark predictor..." << std::endl;
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor shape_predictor;

    // Load model
    dlib::deserialize("../data/models/shape_predictor_68_face_landmarks.dat") >> shape_predictor;

    // Initializing video capture object
    cv::VideoCapture capture("../data/videos/smile.mp4");

    if (!capture.isOpened()) {
        std::cerr << "[ERROR] Unable to connect to camera" << std::endl;
    }

    // Create a VideoWriter object
    cv::VideoWriter smileDetectionOut("smileDetectionOutput.avi",
                                      cv::VideoWriter::fourcc('M','J','P','G'),
                                      15,
                                      cv::Size((int) capture.get(cv::CAP_PROP_FRAME_WIDTH),
                                               (int) capture.get(cv::CAP_PROP_FRAME_HEIGHT)));

    int frame_number = 0;
    std::vector<int> smile_frames;

    cv::Mat frame, frame_small;
    float IMAGE_RESIZE;

    while(capture.read(frame)){

        if (frame.empty()) {
            std::cout << "[ERROR] Unable to capture frame" << std::endl;
            break;
        }

        std::cout << "Processing frame: " << frame_number << std::endl;

        IMAGE_RESIZE = (float)frame.rows / RESIZE_HEIGHT;
        cv::resize(frame, frame, cv::Size(), 1.0 / IMAGE_RESIZE, 1.0 / IMAGE_RESIZE);
        cv::resize(frame, frame_small, cv::Size(), 1.0 / FACE_DOWNSAMPLE_RATIO_DLIB, 1.0 / FACE_DOWNSAMPLE_RATIO_DLIB);

        // Turn OpenCV's Mat into something dlib can deal with. Note that this just
        // wraps the Mat object, it doesn't copy anything. So cimg is only valid as
        // long as frame is valid.  Also don't do anything to frame that would cause it
        // to reallocate the memory which stores the image as that will make cimg
        // contain dangling pointers.  This basically means you shouldn't modify frame
        // while using cimg.
        dlib::cv_image<dlib::bgr_pixel> cimg(frame);
        dlib::cv_image<dlib::bgr_pixel> cimg_small(frame_small);

        // Detect faces
        std::vector<dlib::rectangle> faces = detector(cimg_small);

        // if # faces detected is zero
        if (0 == faces.size()) {
            putText(frame, "Unable to detect face, Please check proper lighting", cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            putText(frame, "Or Decrease FACE_DOWNSAMPLE_RATIO", cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX, 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        }
        else {
            dlib::rectangle face(
                    (long)(faces[0].left() * FACE_DOWNSAMPLE_RATIO_DLIB),
                    (long)(faces[0].top() * FACE_DOWNSAMPLE_RATIO_DLIB),
                    (long)(faces[0].right() * FACE_DOWNSAMPLE_RATIO_DLIB),
                    (long)(faces[0].bottom() * FACE_DOWNSAMPLE_RATIO_DLIB)
            );

            dlib::full_object_detection landmarks = shape_predictor(cimg, face);

            //auto mar = MouthAspectRatio(landmarks);
            //auto jar = JawAspectRatio(landmarks);

            if (smile_detector(cimg, face, landmarks)) {
                cv::putText(frame, cv::format("Smiling :) "), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                std::string imgFilename = "results/smile/img_" + std::to_string( frame_number)  + ".jpg";
                cv::imwrite(imgFilename,frame);
                smile_frames.push_back(frame_number);
            }
            else{
                cv::putText(frame, cv::format("Not Smiling :) "), cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                std::string imgFilename = "results/nosmile/img_" + std::to_string( frame_number)  + ".jpg";
                cv::imwrite(imgFilename,frame);
            }
        }

        if (frame_number%50 == 0){
            std::cout << "\nProcessed " << frame_number << " frames" << std::endl;
            std::cout << "Smile detected in " << smile_frames.size() << " number of frames" << std::endl;
        }
        // Write to VideoWriter
        cv::resize(frame, frame, cv::Size(), IMAGE_RESIZE, IMAGE_RESIZE);
        smileDetectionOut.write(frame);
        frame_number++;
    }

    std::cout << "Total smile detected is " << smile_frames.size() << " for " << frame_number << " number of frames" << std::endl;
    capture.release();
    smileDetectionOut.release();

    return 0;
}