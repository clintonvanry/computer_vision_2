#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <tesseract/renderer.h>

using namespace cv;

int main() {
    auto doc_img = imread("invoice.jpg", IMREAD_COLOR);



    std::string tessdata = "D:\\Program Files\\Tesseract-OCR\\tessdata";
    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    ocr->Init(tessdata.c_str(), "eng");
    ocr->SetImage(doc_img.data, doc_img.cols, doc_img.rows, 3, doc_img.step);

    //We need to use the following code, if not used, there wont be any localisation.
    ocr->Recognize(0);

    std::vector<std::string> billing_amount;
    std::vector<std::string> email_ids;
    // initialize the iterator
    tesseract::ResultIterator* iter = ocr->GetIterator();
    // decide which level do we want.
    tesseract::PageIteratorLevel level = tesseract::RIL_WORD;
    if (iter != 0) {
        do {

            int x1, y1, x2, y2;
            // get the bounding box (ie x,y,w,h)
            iter->BoundingBox(level, &x1, &y1, &x2, &y2);
            //draw it on the image for visualization purpose
            rectangle(doc_img, Rect(x1, y1, x2-x1, y2-y1),Scalar(255,0,0),2);
            //get the text inside the localised box
            const char* word = iter->GetUTF8Text(level);

            std::string wordString(word);
            if(wordString.find("$") != std::string::npos ){
                billing_amount.push_back(wordString);
            }
            if(wordString.find("@") != std::string::npos ){
                email_ids.push_back(wordString);
            }

            //get the confidence of te localised box.
            float conf = iter->Confidence(level);

            std::cout << word << std::endl;

            delete[] word;
        } while (iter->Next(level));
    }

    delete iter;
    ocr->End();

    std::cout << "Extracted Billing Amount:" << std::endl;
    for(auto billAmount : billing_amount){
        std::cout << billAmount << std::endl;
    }

    std::cout << "Extracted Email IDs:" << std::endl;
    for(auto email : email_ids){
        std::cout << email << std::endl;
    }

    imshow("doc_img",doc_img);
    waitKey(0);

    destroyAllWindows();
    return 0;
}
