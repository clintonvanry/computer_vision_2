#include <iostream>

#include <leptonica/allheaders.h> // leptonica main header for image io
#include <tesseract/baseapi.h> // tesseract main header

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

int main() {
    std::cout << "Hello, World!" << std::endl;
    auto img = imread("tesseract-snapshot.png", IMREAD_COLOR);

    auto ocr = new tesseract::TessBaseAPI();
    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    ocr->SetPageSegMode(tesseract::PSM_AUTO);
    ocr->SetImage(img.data, img.cols,img.rows,3,img.step);

    std::string outputText = std::string(ocr->GetUTF8Text());
    std::cout << outputText << std::endl;

    ocr->End();

    return 0;
}
