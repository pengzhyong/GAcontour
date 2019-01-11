#include <stack>
#include <string>
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
				double angle = orienMat.at<double>(r, c);
				double v0 = srcImg.at<double>(r, c);
				if (angle >= 3.0 / 4.0 * CV_PI)//-----注意，图像坐标系的方向！！！
				{
					double fraction = (angle - 3.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					double v1 = srcImg.at<double>(r, c - 1) * fraction + srcImg.at<double>(r + 1, c - 1) * (1 - fraction);
					double v2 = srcImg.at<double>(r, c + 1) * fraction + srcImg.at<double>(r - 1, c + 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;
				}
				else if (angle >= 1.0 / 2.0 * CV_PI)
				{
					double fraction = (angle - 1.0 / 2.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					double v1 = srcImg.at<double>(r + 1, c - 1) * fraction + srcImg.at<double>(r + 1, c) * (1 - fraction);
					double v2 = srcImg.at<double>(r - 1, c + 1) * fraction + srcImg.at<double>(r - 1, c) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;
				}
				else if (angle >= 1.0 / 4.0 * CV_PI)
				{
					double fraction = (angle - 1.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					double v1 = srcImg.at<double>(r + 1, c) * fraction + srcImg.at<double>(r + 1, c + 1) * (1 - fraction);
					double v2 = srcImg.at<double>(r - 1, c) * fraction + srcImg.at<double>(r - 1, c - 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;
				}
				else if (angle >= 0.0)
				{
					double fraction = angle / (1.0 / 4.0 * CV_PI);
					double v0 = srcImg.at<double>(r, c);
					double v1 = srcImg.at<double>(r + 1, c + 1) * fraction + srcImg.at<double>(r, c + 1) * (1 - fraction);
					double v2 = srcImg.at<double>(r - 1, c - 1) * fraction + srcImg.at<double>(r, c - 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;
				}
			}
		}
	}

	void Hysteresis(Mat& srcImg, double tlow, double thigh)
	{
		double maxVal, minVal;
		minMaxLoc(srcImg, &minVal, &maxVal);
		
		srcImg(Range(0, 1), Range(0, srcImg.cols)) = 0.0;//图像周边置0
		srcImg(Range(srcImg.rows - 1, srcImg.rows), Range(0, srcImg.cols)) = 0.0;
		srcImg(Range(0, srcImg.rows), Range(0, 1)) = 0.0;
		srcImg(Range(0, srcImg.rows), Range(srcImg.cols - 1, srcImg.cols)) = 0.0;
		minMaxLoc(srcImg, &minVal, &maxVal);


		if (maxVal != minVal)
			srcImg.forEach<double>([minVal, maxVal](double &p, const int * position)->void {p = (p - minVal) / (maxVal - minVal); });//forEach is parralel, can increase performance
		stack<pair<int, int>> edgePoints;
		for (int r = 1; r < srcImg.rows - 1; r++)//push all the edge points into stack
		{
			double* ptr = srcImg.ptr<double>(r);
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
				if (srcImg.at<double>(neigbor[i].first, neigbor[i].second) > tlow)
				{
					edgePoints.push(neigbor[i]);
					srcImg.at<double>(neigbor[i].first, neigbor[i].second) = -1;
				}
			}

		}
		srcImg.forEach<double>([](double &p, const int * position)->void {if (abs(p + 1) < 0.0001) { p = 1; } else p = 0; });// restore p==-1 to p=1
	}

	void inhibition(Mat& srcImg, double beta, int steps, int kersz, double sigama)
	{
		Mat ker(kersz, kersz, CV_64F);
		double kersum = 0;
		for (int i = 0; i < kersz; i++)
		{
			for (int j = 0; j < kersz; j++)
			{
				int m = kersz / 2;
				int x = i - m;
				int y = j - m;
				double tmp = exp((-x * x - y * y)*1.0 / (2 * sigama*sigama)) / sigama;
				ker.at<double>(i, j) = tmp;
				kersum += tmp;
			}
		}
		ker.forEach<double>([kersum](double& p, const int* pos)
			->void {p /= kersum; });
		Mat tmpImg(srcImg.size(), srcImg.type());
		while (steps--)
		{
			filter2D(srcImg, tmpImg, srcImg.depth(), ker);
			//srcImg.forEach<double>([tmpImg, beta](double& p, const int* pos)
			//	->void {p = p - beta * tmpImg.at<double>(pos[0], pos[1]); if (p < 0) p = 0; });
			for (int r = 0; r < srcImg.rows; r++)
			{
				double* psrc = srcImg.ptr<double>(r);
				double* ptmp = tmpImg.ptr<double>(r);
				for (int c = 0; c < srcImg.cols; c++)
				{
					psrc[c] -= beta * ptmp[c];
					if (psrc[c] < 0)psrc[c] = 0;
				}
			}
			
		}
	}

	void excitation(Mat& magImg, const Mat& orientMat, double gama)
	{
		Mat tmpMat = magImg.clone();
		int nh = magImg.rows;
		int nw = magImg.cols;
		for (int r = 1; r < nh - 1; r++)
		{
			for (int c = 1; c < nw - 1; c++)
			{
				double fraction;
				double v1 = 0;
				double v2 = 0;

				double angle = orientMat.at<double>(r, c);
				double v0 = magImg.at<double>(r, c);
				if (angle >= 3.0 / 4.0 * CV_PI)//-----注意，图像坐标系的方向！！！
				{
					/*double fraction = (angle - 3.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					double v1 = srcImg.at<double>(r, c - 1) * fraction + srcImg.at<double>(r + 1, c - 1) * (1 - fraction);
					double v2 = srcImg.at<double>(r, c + 1) * fraction + srcImg.at<double>(r - 1, c + 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;*/

					fraction = (angle - 3.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					v1 = tmpMat.at<double>(r + 1, c) * fraction + tmpMat.at<double>(r + 1, c + 1) * (1 - fraction);
					v2 = tmpMat.at<double>(r - 1, c) * fraction + tmpMat.at<double>(r - 1, c - 1) * (1 - fraction);
				}
				else if (angle >= 1.0 / 2.0 * CV_PI)
				{
					/*double fraction = (angle - 1.0 / 2.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					double v1 = srcImg.at<double>(r + 1, c - 1) * fraction + srcImg.at<double>(r + 1, c) * (1 - fraction);
					double v2 = srcImg.at<double>(r - 1, c + 1) * fraction + srcImg.at<double>(r - 1, c) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;*/

					fraction = (angle - 1.0 / 2.0 * CV_PI) / (1.0 / 4.0 * CV_PI);

					v1 = tmpMat.at<double>(r + 1, c + 1) * fraction + tmpMat.at<double>(r, c + 1) * (1 - fraction);
					v2 = tmpMat.at<double>(r - 1, c - 1) * fraction + tmpMat.at<double>(r, c - 1) * (1 - fraction);
					//magImg.at<double>(r, c) += abs(gama * (v1 + v2));


				}
				else if (angle >= 1.0 / 4.0 * CV_PI)
				{
					/*double fraction = (angle - 1.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
					double v1 = srcImg.at<double>(r + 1, c) * fraction + srcImg.at<double>(r + 1, c + 1) * (1 - fraction);
					double v2 = srcImg.at<double>(r - 1, c) * fraction + srcImg.at<double>(r - 1, c - 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;*/

					fraction = (angle - 1.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);

					v1 = tmpMat.at<double>(r, c - 1) * fraction + tmpMat.at<double>(r + 1, c - 1) * (1 - fraction);
					v2 = tmpMat.at<double>(r, c + 1) * fraction + tmpMat.at<double>(r - 1, c + 1) * (1 - fraction);
					//magImg.at<double>(r, c) += abs(gama * (v1 + v2));


				}
				else if (angle >= 0.0)
				{
					/*double fraction = angle / (1.0 / 4.0 * CV_PI);
					double v0 = srcImg.at<double>(r, c);
					double v1 = srcImg.at<double>(r + 1, c + 1) * fraction + srcImg.at<double>(r, c + 1) * (1 - fraction);
					double v2 = srcImg.at<double>(r - 1, c - 1) * fraction + srcImg.at<double>(r, c - 1) * (1 - fraction);
					if (v0 < v1 || v0 <= v2)
						dstImg.at<double>(r, c) = 0.0;*/

					fraction = angle / (1.0 / 4.0 * CV_PI);

					v1 = tmpMat.at<double>(r + 1, c - 1) * fraction + tmpMat.at<double>(r + 1, c) * (1 - fraction);
					v2 = tmpMat.at<double>(r - 1, c + 1) * fraction + tmpMat.at<double>(r - 1, c) * (1 - fraction);
					//magImg.at<double>(r, c) += abs(gama * (v1 + v2));
				}
				double eps = 0.0001;
				if (magImg.at<double>(r, c) > eps)
					magImg.at<double>(r, c) += abs(gama * (v1 + v2));
			}
		}
	}

	Mat Mycanny(Mat& srcImg, double tlow, double thigh, bool isInhibition, bool isExcitation, Size kersz, double sigma)
	{
		if (srcImg.channels() != 1) cvtColor(srcImg, srcImg, COLOR_BGR2GRAY);
		if (srcImg.type() != CV_64F) srcImg.convertTo(srcImg, CV_64F, 1.0 / 255.0);
		GaussianBlur(srcImg, srcImg, kersz, 3);
		//--------get orientation mat and magnitude mat
		Mat imgx, imgy;
		Sobel(srcImg, imgx, srcImg.depth(), 1, 0);
		Sobel(srcImg, imgy, srcImg.depth(), 0, 1);
		
		Mat orientMat(srcImg.size(), CV_64F, Scalar(0));
		Mat magMat(srcImg.size(), CV_64F);

		for (int r = 0; r < srcImg.rows; r++)
		{
			for (int c = 0; c < srcImg.cols; c++)
			{
				double angle = 0.5 * CV_PI;
				double eps = 0.01;
				if(abs(imgx.at<double>(r, c)) > eps )
					angle = atan(imgy.at<double>(r, c) / imgx.at<double>(r, c));
				if (angle < 0) angle += CV_PI;
				orientMat.at<double>(r, c) = angle;
				magMat.at<double>(r, c) = sqrt(imgx.at<double>(r, c)*imgx.at<double>(r, c) + imgy.at<double>(r, c)*imgy.at<double>(r, c));
			}
		}
		Mat dstImg;
		int steps = 1;
		while (steps--)
		{
			if (isInhibition == true)
				inhibition(magMat, 0.2, 1, 11, 5);
			//Thinning(magMat, dstImg, orientMat);
			//excitation(dstImg, orientMat, 0.5);
			int nex = 10;//iteration times
			while (nex--)
			{
				//if (isInhibition == true)
				//	inhibition(magMat, 0.1, 1, 3, 3);
				if (isExcitation == true)
					excitation(magMat, orientMat, 0.8);
			}
			
			//inhibition(magMat, 0.3, 3, 21);
			//cout << magMat << endl;
			Thinning(magMat, dstImg, orientMat);
			//normalize(dstImg, dstImg, 0, 1, NORM_MINMAX);
 			Hysteresis(dstImg, tlow, thigh);
			dstImg.forEach<double>([](double& p, const int* pos)
				->void {if (p < 0.001) p = 1; else p = 0; });

			//Hysteresis(dstImg, tlow, thigh);
			//string winname = "step: ";// +to_string(steps + 1);
			//normalize(dstImg, dstImg, 0, 1, NORM_MINMAX);
			//imshow(winname, dstImg); waitKey();
		}
		
		return dstImg;
	}
}
