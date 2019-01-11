#pragma once
#include <opencv2/opencv.hpp>
#include "NCRF.h"
using namespace cv;
void foo()
{
	/*Mat m(512, 512, CV_64F, Scalar(0));
	Point2f center(m.rows / 2, m.cols / 2);
	double len = 100;
	line(m, Point2f(center.x, center.y + len), Point2f(center.x, center.y - len), Scalar(1));
	line(m, Point2f(center.x - 30, center.y + len), Point2f(center.x - 30, center.y - len), Scalar(1));
	line(m, Point2f(center.x + 30, center.y + len), Point2f(center.x + 30, center.y - len), Scalar(1));*/
	double sigma = 2;
	double lambda = 10;
	double theta = 45.0 / 180 * 2 * CV_PI;
	double phi = 0;
	double gamma = 0.3;
	double bandwidth = 1;
	int n = int(ceilf(2.5 * sigma / gamma));
	int kersz = 2 * n + 1;
	double radiu = 40;
	Point2f center(n, n);//坐上角为原点，x轴向下为正，y轴向右为正
	Point2f circleCenter(n, n + radiu);//圆心位置
	Mat kernel = GaborKernel2d(sigma, lambda, 0.0, phi, gamma, bandwidth);
	//namedWindow("kernel", 0);
	//normalize(kernel, kernel, 0, 1, NORM_MINMAX);
	//imshow("kernel", kernel);
	//waitKey(0);
	Mat xhh(kernel.size(), CV_64F, Scalar(0));//小灰灰
	for (int r = 0; r < kernel.rows; r++)
	{
		for (int c = 0; c < kernel.cols; c++)
		{
			//double radiu1 = radiu - (c - center.y);
			//double len = r - center.y;
			//double theta = 0.5 * (CV_PI - len / radiu1);
			//double chord = radiu1 * cos(theta) * 2;
			//int y1 = center.y + (c - center.y) + chord * cos(theta);
			//int x1 = center.x + chord * sin(theta);
			//if (x1 < 0 || x1 >= kernel.rows || y1 < 0 || y1 >= kernel.cols)
			//	continue;
			////if (kernel.at<double>(r, c) > 0)
			//xhh.at<double>(x1, y1) = kernel.at<double>(r, c);

			double dx = r - circleCenter.x;
			double dy = c - circleCenter.y;
			double radiu1 = sqrt(dx*dx + dy*dy);
			if (abs(dx - 0.0) < 0.0001 && abs(dy + radiu) < 0.0001)
				int var = 0;
			double theta = atan(dx/dy);
			double len = radiu1 * theta;
			int x0 = circleCenter.x - len + 0.5;//关键的0.5！ 四舍五入
			int y0 = circleCenter.y - radiu1 + 0.5;
			if(x0 < 0 || x0 >= kernel.rows || y0 < 0 || y0 > kernel.cols)
				continue;
			xhh.at<double>(r, c) = kernel.at<double>(x0, y0);
			if (r != x0 || c != y0)
				int bar = 0;
		}
	}
	Mat roteMat = getRotationMatrix2D(Point2f(center), 0, 1.0);
	Mat result;
	warpAffine(xhh, result, roteMat, xhh.size());
	normalize(result, result, 0, 1, NORM_MINMAX);
	namedWindow("affine kernel", 0);
	imshow("affine kernel", result);
	waitKey(0);
}
