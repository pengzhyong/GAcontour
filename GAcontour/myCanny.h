#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
namespace myspace
{
	void Thinning(const Mat& srcImg, Mat& dstImg, const Mat& orienMat);
	void Hysteresis(Mat& srcImg, double tlow, double thigh);
	void inhibition(Mat& srcImg, double beta = 1, int steps = 1, int kersz = 3, double sigama = 1);

	Mat Mycanny(Mat& srcImg, double tlow, double thigh, bool isInhibition, bool isExcitation, Size kersz = Size(3,3), double sigma = 1.0);
}
