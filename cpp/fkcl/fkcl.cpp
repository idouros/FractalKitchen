// fkcl.cpp : Defines the entry point for the application.
//
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "fkcl.h"

using namespace std;
using namespace cv;

int main()
{
	listDevices();

	int n_rows = 200;
	int n_cols = 200;

	cv::Mat fractalImage(n_rows, n_cols, CV_8UC3); 
	fractalImage.setTo(cv::Scalar(0, 0, 0));



	namedWindow("Display window", WINDOW_AUTOSIZE);
	imshow("Display window", fractalImage);
	waitKey(0);


	return 0;
}
