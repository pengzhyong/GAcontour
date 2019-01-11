#pragma once
#include <opencv2/opencv.hpp>
#include "NCRF.h"
using namespace cv;

Mat CircleDetection(Mat srcImg, double radiu)
{
	vector<double> lamdaVec = { 3,5,7,9,11,13,15,17,19,21,23 };// , 25, 27, 29, 31, 33, 35};
	vector<double> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);
	double gamma = 0.1;//aspect ratio
	double bandwidth = 1;
	int ntheta = 24;
	int nphi = 2;
	double arrphi[2] = { 0, 0.5*CV_PI };
	vector<double> phi = { arrphi, arrphi + 2 };
	vector<double> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	bool halfwave = 0;

	vector<Mat> thetas;
	for (int i = 0; i < ntheta; i++)
	{
		vector<Mat> phiVec;
		for (int j = 0; j < nphi; j++)
		{
			Mat tmpResult(srcImg.size(), srcImg.type(), Scalar(0));
			for (int s = 0; s < lamdaVec.size(); s++)
			{
				Mat kernel = GaborKernel2d(sigmaVec.at(s), lamdaVec.at(s), theta[i], phi[j], gamma, bandwidth, radiu);
				Mat tmpScale;
				filter2D(srcImg, tmpScale, srcImg.depth(), kernel);
				tmpResult.forEach<double>([tmpScale](double& p, const int* pos)->void { p = (p + tmpScale.at<double>(pos[0], pos[1])); });
			}
			int sz = lamdaVec.size();
			tmpResult.forEach<double>([sz](double& p, const int* pos)->void {p = p * 1.0 / sz; });

			if (halfwave)//°ë²¨ÕûÁ÷
			{
				for (int r = 0; r < tmpResult.rows; r++)
				{
					double* ptr = tmpResult.ptr<double>(r);
					for (int c = 0; c < tmpResult.cols; c++)
					{
						if (ptr[c] < 0)
							ptr[c] = 0;
					}
				}
			}
			phiVec.push_back(tmpResult);
		}
		Mat thetaMat(phiVec.at(0).size(), phiVec.at(0).type());
		thetaMat.forEach<double>([phiVec](double& p, const int* pos)
			->void { p = sqrt(phiVec.at(0).at<double>(pos[0], pos[1]) * phiVec.at(0).at<double>(pos[0], pos[1])
				+ 0 * phiVec.at(1).at<double>(pos[0], pos[1]) * phiVec.at(0).at<double>(pos[0], pos[1])); });
		thetas.push_back(thetaMat);
	}
	Mat dstImg(srcImg.size(), srcImg.type(), Scalar(-1000.0));
	for (int i = 0; i < ntheta; i++)
	{
		for (int r = 0; r < srcImg.rows; r++)
		{
			for (int c = 0; c < srcImg.cols; c++)
			{
				if (dstImg.at<double>(r, c) < thetas[i].at<double>(r, c))
					dstImg.at<double>(r, c) = thetas[i].at<double>(r, c);
			}
		}
	}

	normalize(dstImg, dstImg, 0, 1, NORM_MINMAX);
	imshow("dstImg", dstImg);
	waitKey(0);
	return dstImg;
}
