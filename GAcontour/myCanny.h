#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
namespace myspace
{
	void Thinning(const Mat& srcImg, Mat& dstImg, const Mat& orienMat);
	void Hysteresis(Mat& srcImg, float tlow, float thigh);
	Mat Mycanny(Mat& srcImg, float tlow, float thigh, Size kersz = Size(3,3), float sigma = 1.0);
}
