#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main(int, char**) {
    Mat src=imread("5.jpg");
    Mat dst;
    cvtColor(src,dst,COLOR_BGR2GRAY);
    resize(dst,dst,Size(28,28));
    imwrite("test.png",dst);
    std::cout << "Hello, world!\n";
}

