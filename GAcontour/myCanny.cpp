#include <stack>
#include <opencv2/opencv.hpp>
#include "myCanny.h"
using namespace std;
namespace myspace
{
	void Thinning(const Mat& srcImg, Mat& dstImg, const Mat& orienMat)
	{
		dstImg = srcImg.clone();

		int nh = srcImg.rows;
		int nw = srcImg.cols;
		for (int r = 1; r < nh - 1; r++)
		{
			for (int c = 1; c < nw - 1; c++)
			{
				float angle = orienMat.at<float>(r, c);
				float v0 = srcImg.at<float>(r, c);
				if (angle >= 3.0 / 4.0 * CV_PI)//-----注意，图像坐标系的方向！！！
				{
					float fraction = (angle - 3.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					float v1 = srcImg.at<float>(r, c - 1) * fraction + srcImg.at<float>(r + 1, c - 1) * (1 - fraction);
					float v2 = srcImg.at<float>(r, c + 1) * fraction + srcImg.at<float>(r - 1, c + 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<float>(r, c) = 0.0;
				}
				else if (angle >= 1.0 / 2.0 * CV_PI)
				{
					float fraction = (angle - 1.0 / 2.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					float v1 = srcImg.at<float>(r + 1, c - 1) * fraction + srcImg.at<float>(r + 1, c) * (1 - fraction);
					float v2 = srcImg.at<float>(r - 1, c + 1) * fraction + srcImg.at<float>(r - 1, c) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<float>(r, c) = 0.0;
				}
				else if (angle >= 1.0 / 4.0 * CV_PI)
				{
					float fraction = (angle - 1.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					float v1 = srcImg.at<float>(r + 1, c) * fraction + srcImg.at<float>(r + 1, c + 1) * (1 - fraction);
					float v2 = srcImg.at<float>(r - 1, c) * fraction + srcImg.at<float>(r - 1, c - 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<float>(r, c) = 0.0;
				}
				else if (angle >= 0.0)
				{
					float fraction = angle / (1.0 / 4.0 * CV_PI);
					float v0 = srcImg.at<float>(r, c);
					float v1 = srcImg.at<float>(r + 1, c + 1) * fraction + srcImg.at<float>(r, c + 1) * (1 - fraction);
					float v2 = srcImg.at<float>(r - 1, c - 1) * fraction + srcImg.at<float>(r, c - 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<float>(r, c) = 0.0;
				}
			}
		}
	}

	void Hysteresis(Mat& srcImg, float tlow, float thigh)
	{
		double maxVal, minVal;
		minMaxLoc(srcImg, &minVal, &maxVal);
		if (maxVal != minVal)
			srcImg.forEach<float>([minVal, maxVal](float &p, const int * position)->void {p = (p - minVal) / (maxVal - minVal); });//forEach is parralel, can increase performance
		
		srcImg(Range(0, 1), Range(0, srcImg.cols)) = 0.0;//图像周边置0
		srcImg(Range(srcImg.rows - 1, srcImg.rows), Range(0, srcImg.cols)) = 0.0;
		srcImg(Range(0, srcImg.rows), Range(0, 1)) = 0.0;
		srcImg(Range(0, srcImg.rows), Range(srcImg.cols - 1, srcImg.cols)) = 0.0;
		stack<pair<int, int>> edgePoints;
		for (int r = 1; r < srcImg.rows - 1; r++)//push all the edge points into stack
		{
			float* ptr = srcImg.ptr<float>(r);
			for (int c = 1; c < srcImg.cols - 1; c++)
			{
				if (ptr[c] > thigh)
				{
					edgePoints.push(pair<int, int>(r, c));
					ptr[c] = -1;//mean that this point has been in stack, avoid push repeat
				}
			}
		}
		while (!edgePoints.empty())
		{
			pair<int, int> posIndex = edgePoints.top();
			int posx = posIndex.first;
			int posy = posIndex.second;
			edgePoints.pop();
			pair<int, int> neigbor[8] = { pair<int, int>(posx - 1, posy - 1),
				pair<int, int>(posx, posy - 1),
				pair<int, int>(posx + 1, posy - 1),
				pair<int, int>(posx - 1, posy),
				pair<int, int>(posx + 1, posy),
				pair<int, int>(posx - 1, posy + 1),
				pair<int, int>(posx, posy + 1),
				pair<int, int>(posx + 1, posy + 1) };
			for (int i = 0; i < 8; i++)
			{
				if (srcImg.at<float>(neigbor[i].first, neigbor[i].second) > tlow)
				{
					edgePoints.push(neigbor[i]);
					srcImg.at<float>(neigbor[i].first, neigbor[i].second) = -1;
				}
			}

		}
		srcImg.forEach<float>([](float &p, const int * position)->void {if (abs(p + 1) < 0.0001) { p = 1; } else p = 0; });// restore p==-1 to p=1
	}

	Mat Mycanny(Mat& srcImg, float tlow, float thigh, Size kersz, float sigma)
	{
		if (srcImg.channels() != 1) cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
		if (srcImg.type() != CV_32F) srcImg.convertTo(srcImg, CV_32F, 1.0 / 255.0);
		GaussianBlur(srcImg, srcImg, kersz, sigma);
		//--------get orientation mat and magnitude mat
		Mat imgx, imgy;
		Sobel(srcImg, imgx, srcImg.depth(), 1, 0);
		Sobel(srcImg, imgy, srcImg.depth(), 0, 1);
		
		Mat orientMat(srcImg.size(), CV_32F, Scalar(0));
		Mat magMat(srcImg.size(), CV_32F);

		for (int r = 0; r < srcImg.rows; r++)
		{
			for (int c = 0; c < srcImg.cols; c++)
			{
				float angle = 0.5 * CV_PI;
				float eps = 0.01;
				if(abs(imgx.at<float>(r, c)) > eps )
					angle = atan(imgy.at<float>(r, c) / imgx.at<float>(r, c));
				if (angle < 0) angle += CV_PI;
				orientMat.at<float>(r, c) = angle;
				magMat.at<float>(r, c) = sqrt(imgx.at<float>(r, c)*imgx.at<float>(r, c) + imgy.at<float>(r, c)*imgy.at<float>(r, c));
			}
		}
		Mat dstImg;
		Thinning(magMat, dstImg, orientMat);
		Hysteresis(dstImg, tlow, thigh);
		return dstImg;
	}
}
