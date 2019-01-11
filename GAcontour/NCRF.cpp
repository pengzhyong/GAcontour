#include "NCRF.h"
#include <stack>
#include <numeric>

int GaborFilter(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave, 
	double lamda, double sigma, vector<double> theta, vector<double> phi, double gamma, double bandwidth)
{
	int kersz;
	if (gamma <= 1 && gamma > 0)
		kersz = int(ceilf(2.5 * sigma / gamma));
	else
		kersz = int(ceilf(2.5 * sigma));
	kersz = 2 * kersz + 1;
	
	//Mat fftImg = createFftImage(srcImg, kersz, kersz);
	int ntheta = theta.size();
	int nphi = phi.size();
	Mat kernel;
	vector<Mat> phiVec;
	for (int i = 0; i < ntheta; i++)
	{
		for (int j = 0; j < nphi; j++)
		{
			kernel = GaborKernel2d(sigma, lamda, theta[i], phi[j], gamma, bandwidth);
			
			Mat tmpResult;
			filter2D(srcImg, tmpResult, srcImg.depth(), kernel);
			
			/*normalize(kernel, kernel, 0, 1, NORM_MINMAX);
			namedWindow("kernel", 0);
			imshow("kernel", kernel);
			normalize(tmpResult, tmpResult, 0, 1, NORM_MINMAX);
			imshow("tmpReslut", tmpResult);
			waitKey(0);*/
			if (halfwave)//半波整流
			{
				for (int r = 0; r < tmpResult.rows; r++)
				{ 
					double* ptr = tmpResult.ptr<double>(r);
					for (int c = 0; c < tmpResult.cols; c++)
					{
						if(ptr[c] < 0)
							ptr[c] = 0;
					}
				}
			}
			phiVec.push_back(tmpResult);
		}
		dstImg.push_back(phiVec);
		phiVec.clear();
	}
	return kersz;
}

int GaborFilterMulScale(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	vector<double> lamda, vector<double> sigma, vector<double> theta, vector<double> phi, double gamma, double bandwidth)
{
	//Mat fftImg = createFftImage(srcImg, kersz, kersz);
	int ntheta = theta.size();
	int nphi = phi.size();
	int kersz;

	for (int i = 0; i < ntheta; i++)
	{
		vector<Mat> phiVec;
		for (int j = 0; j < nphi; j++)
		{
			Mat tmpResult;
			for (int s = 0; s < lamda.size(); s++)
			{
				if (gamma <= 1 && gamma > 0)
					kersz = int(ceilf(2.5 * sigma[s] / gamma));
				else
					kersz = int(ceilf(2.5 * sigma[s]));
				kersz = 2 * kersz + 1;
				Mat kernel = GaborKernel2d(sigma.at(s), lamda.at(s), theta[i], phi[j], gamma, bandwidth);				
				Mat scaleMat;
				filter2D(srcImg, scaleMat, srcImg.depth(), kernel);
				if (s == 0)
					tmpResult = scaleMat.clone();
				else
				{
					tmpResult.forEach<double>([scaleMat](double& p, const int* pos)->void { p = p + scaleMat.at<double>(pos[0], pos[1]); });
				}
			}
			int sz = lamda.size();
			tmpResult.forEach<double>([sz](double& p, const int* pos)->void {p = p * 1.0 / sz; });

			if (halfwave)//半波整流
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
		dstImg.push_back(phiVec);
	}
	return kersz;
}

int GaborFilterMulScaleBend(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	vector<double> lamda, vector<double> sigma, vector<double> theta, vector<double> phi, double gamma, double bandwidth)
{
	//Mat fftImg = createFftImage(srcImg, kersz, kersz);
	int ntheta = theta.size();
	int nphi = phi.size();
	int kersz;

	for (int i = 0; i < ntheta; i++)
	{
		vector<Mat> phiVec;
		for (int j = 0; j < nphi; j++)
		{
			Mat tmpResult;
			for (int s = 0; s < lamda.size(); s++)
			{
				double sigmas = sigma[s];
				vector<double> radiusVec = { -3 * sigmas, -5 * sigmas, -7 * sigmas, -9 * sigmas, -11 * sigmas, -13 * sigmas, -20 * sigmas, -40 * sigmas, -100 * sigmas,
					3 * sigmas, 5 * sigmas, 7 * sigmas, 9 * sigmas, 11 * sigmas, 13 * sigmas, 20 * sigmas, 40 * sigmas, 100 * sigmas, };
				vector<double> logRadiusVec;
				for (auto i : radiusVec)
				{
					logRadiusVec.push_back(log(abs(i)));
				}
				double sumLogRadius = accumulate(logRadiusVec.begin(), logRadiusVec.end(), 0);
				vector<double> coefVec;
				for (auto i : logRadiusVec)
					coefVec.push_back(i / sumLogRadius);
				//vector<double> radiusVec = { 50000 };

				Mat scaleMat;
				for (int ri = 0; ri < radiusVec.size(); ri++)
				{
					if (gamma <= 1 && gamma > 0)
						kersz = int(ceilf(2.5 * sigma[s] / gamma));
					else
						kersz = int(ceilf(2.5 * sigma[s]));         
					kersz = 2 * kersz + 1;
					Mat kernel = GaborKernel2d(sigma.at(s), lamda.at(s), theta[i], phi[j], gamma, bandwidth, 0.2 * radiusVec.at(ri));
					Mat kernel2 = GaborKernel2d(sigma.at(s), lamda.at(s), theta[i], phi[j], gamma, bandwidth);// , 100000.0/* * radiusVec.at(ri)*/);
					/*namedWindow("kernel1", 0);
					namedWindow("kernel2", 0);
					normalize(kernel, kernel, 0, 1, NORM_MINMAX);
					normalize(kernel2, kernel2, 0, 1, NORM_MINMAX);

					imshow("kernel1", kernel);
					imshow("kernel2", kernel2);
					waitKey(0);*/
					//kernel = kernel2.clone();
					Mat tmpScale;
					filter2D(srcImg, tmpScale, srcImg.depth(), kernel);
					if (ri == 0)
						scaleMat = tmpScale.clone();
					else
						scaleMat.forEach<double>([tmpScale, coefVec, ri](double& p, const int* pos)->void { p = max(p, /*coefVec.at(ri) **/ tmpScale.at<double>(pos[0], pos[1])); });
				}
				if (s == 0)
					tmpResult = scaleMat.clone();
				else
				{
					tmpResult.forEach<double>([scaleMat](double& p, const int* pos)->void { p = (p + scaleMat.at<double>(pos[0], pos[1])); });
				}
				/*if (gamma <= 1 && gamma > 0)
					kersz = int(ceilf(2.5 * sigma[s] / gamma));
				else
					kersz = int(ceilf(2.5 * sigma[s]));
				kersz = 2 * kersz + 1;
				Mat kernel = GaborKernel2d(sigma.at(s), lamda.at(s), theta[i], phi[j], gamma, bandwidth);
				Mat scaleMat;
				filter2D(srcImg, scaleMat, srcImg.depth(), kernel);
				if (s == 0)
					tmpResult = scaleMat.clone();
				else
				{
					tmpResult.forEach<double>([scaleMat](double& p, const int* pos)->void { p = p + scaleMat.at<double>(pos[0], pos[1]); });
				}*/
			}
			int sz = lamda.size();
			tmpResult.forEach<double>([sz](double& p, const int* pos)->void {p = p * 1.0 / sz; });

			if (halfwave)//半波整流
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
		dstImg.push_back(phiVec);
	}
	return kersz;
}

void PhaseSuppos(vector<Mat>& dstImg, const vector<vector<Mat>>& srcImg, int ntheta, int nphi, int supMethod)
{
	if (supMethod == 0)
	{
		for (int i = 0; i < ntheta; i++)
		{
			dstImg.push_back(srcImg[i][0]);
		}
		return;
	}

	int nh = srcImg[0][0].rows;
	int nw = srcImg[0][0].cols;

	if (supMethod == 1)//1阶范数
	{
		for (int i = 0; i < ntheta; i++)
		{			
			Mat tmpImg(nh, nw, CV_64F);
			for (int r = 0; r < nh; r++)
			{
				double* ptr = tmpImg.ptr<double>(r);			
				for (int c = 0; c < nw; c++)
				{
					ptr[c] = 0;
					for (int j = 0; j < nphi; j++)
					{
						const double* phi1_ptr = srcImg[i][j].ptr<double>(r);
						ptr[c] += abs(phi1_ptr[c]);
					}
					
				}
			}
			
			dstImg.push_back(tmpImg);
		}
		return;
	}

	if (supMethod == 2)//2阶范数
	{
		for (int i = 0; i < ntheta; i++)
		{
			Mat tmpImg(nh, nw, CV_64F);
			for (int r = 0; r < nh; r++)
			{
				double* ptr = tmpImg.ptr<double>(r);
				for (int c = 0; c < nw; c++)
				{
					ptr[c] = 0;
					for (int j = 0; j < nphi; j++)
					{
						ptr[c] += srcImg[i][j].at<double>(r, c) * srcImg[i][j].at<double>(r, c);
					}
					ptr[c] = sqrtf(ptr[c]);
				}
			}

			dstImg.push_back(tmpImg);
		}
		return;
	}

	if (supMethod == 3)//无穷阶范数
	{
		for (int i = 0; i < ntheta; i++)
		{
			Mat tmpImg(nh, nw, CV_64F);
			for (int r = 0; r < nh; r++)
			{
				double* ptr = tmpImg.ptr<double>(r);
				for (int c = 0; c < nw; c++)
				{
					ptr[c] = 0;
					for (int j = 0; j < nphi; j++)
					{
						const double* phi1_ptr = srcImg[i][j].ptr<double>(r);
						if (phi1_ptr[c] > ptr[c])
							ptr[c] = phi1_ptr[c];
					}
				}
			}

			dstImg.push_back(tmpImg);
		}
		return;
	}
}

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, double sigma, double alpha, int k1, int k2)
{
	int nh = srcImg[0].rows;
	int nw = srcImg[0].cols;
	int ntheta = srcImg.size();
	double initValue = 0.0;
	if (supMethod == 3)//max sup, infinit norm
		initValue = -1000.0;//-INF

	Mat inhibitor(srcImg[0].size(), CV_64F, Scalar(initValue));//抑制区
	if (inhibMethod == 2)
	{
		for (int r = 0; r < nh; r++)
		{
			double* ptr = inhibitor.ptr<double>(r);
			for (int c = 0; c < nw; c++)
			{
				for (int i = 0; i < ntheta; i++)
				{
					if (supMethod == 1)
						ptr[c] += abs(srcImg[i].at<double>(r, c));
					if (supMethod == 2)
						ptr[c] += srcImg[i].at<double>(r, c) * srcImg[i].at<double>(r, c);
					if (supMethod == 3)
					{
						if (ptr[c] < srcImg[i].at<double>(r, c))
							ptr[c] = srcImg[i].at<double>(r, c);
					}
				}
				if (supMethod == 2)
					ptr[c] = sqrtf(ptr[c]);
			}
		}
		/*Mat showImg = inhibitor.clone();
		normalize(showImg, showImg, 0.0, 1.0, NORM_MINMAX);
		imshow("inhib", showImg);
		waitKey(0);*/
	}
	Mat inhibMat(nh, nw, CV_64F);
	int depth = srcImg[0].depth();
	Mat dogKernel = DogKernel2d(sigma, k1, k2);
	if (inhibMethod == 2)
		filter2D(inhibitor, inhibMat, depth, dogKernel);

	
	for (int i = 0; i< ntheta; i++)
	{
		if (inhibMethod == 2)// isotropic inhibition
		{
			/*Mat showImg = inhibitor.clone();
			normalize(showImg, showImg, 0.0, 1.0, NORM_MINMAX);
			imshow("inhib" + to_string(i), showImg);
			waitKey(0);*/

			Mat finalMat(nh, nw, CV_64F);//一个难找的bug, finalMat必须在循环体内部定义，因为finalMat用于push_back进输出参数，若定义在循环体外部，相当于push_back的是同一个对象，没有新的对象
										 //当函数体退出时，会销毁函数体内部定义的变量。引用、指针除外。此处说明push_back进输出参数的对象也不会被销毁。
			
			finalMat.forEach<double>([srcImg,i,alpha,inhibMat](double& p, const int* pos)->void {
				p = srcImg[i].at<double>(pos[0], pos[1]) - alpha * inhibMat.at<double>(pos[0], pos[1]);
				if (p < 0) p = 0; });//Set negtive value to zero!
			/*for (int r = 0; r < srcImg[i].rows; r++)
			{
				for (int c = 0; c < srcImg[i].cols; c++)
				{
					finalMat.at<double>(r, c) = srcImg[i].at<double>(r, c) - alpha * inhibMat.at<double>(r, c);
					if (finalMat.at<double>(r, c) < 0)
						finalMat.at<double>(r, c) = 0;
				}
			}*/
			dstImg.push_back(finalMat);	

		}
		if (inhibMethod == 3)// anisotropic inhibition
		{
			Mat finalMat(nh, nw, CV_64F);
			filter2D(srcImg[i], inhibMat, depth, dogKernel);
			finalMat.forEach<double>([srcImg, i, alpha, inhibMat](double& p, const int* pos)->void {
				p = srcImg[i].at<double>(pos[0], pos[1]) - alpha * inhibMat.at<double>(pos[0], pos[1]); 
				if (p < 0) p = 0; });
			dstImg.push_back(finalMat);
		}
		if (inhibMethod == 1)// no inhibition
		{
			dstImg.push_back(srcImg[i]);
		}

		//-----刺激响应
		Mat exatator(srcImg[0].size(), CV_64F, Scalar(0));//刺激区
		for (int r = 0; r < nh; r++)
		{
			double* ptr = exatator.ptr<double>(r);
			for (int c = 0; c < nw; c++)
			{
				for (int j = 0; j < ntheta; j++)
				{
					double dtheta = ((j - i + ntheta) % ntheta)*(2 * CV_PI) / ntheta;
					ptr[c] += srcImg[j].at<double>(r, c) * abs(cos(dtheta));
				}
			}
		}
		double beta = 1.0;
		Mat exataMat(nh, nw, CV_64F);
		int depth = srcImg[0].depth();
		Mat dogKernel = DogKernel2d(sigma, k1, k2);
		int cx = dogKernel.rows / 2;
		int cy = dogKernel.cols / 2;
		for (int r = 0; r < dogKernel.rows; r++)
		{
			for (int c = 0;  c < dogKernel.cols; c++)
			{
				double k_theta = atan2(-r+cx, c-cy);
				if (k_theta < 0) k_theta += 2 * CV_PI;
				//if (-r + cx < 0 && c - cy>0) k_theta += 2 * CV_PI;
				//if (-r + cx < 0 && c - cy < 0) k_theta += CV_PI;
				//if (-r + cx > 0 && c - cy > 0) k_theta += 0;
				//if (-r + cx < 0 && c - cy < 0) k_theta += CV_PI;;

				double theta0 = (i*2.0*CV_PI/ntheta +0.5*CV_PI);
				if (theta0 > 2 * CV_PI) theta0 -= 2 * CV_PI;
				//if (theta0 > CV_PI) theta0 -= CV_PI;
				//if (theta0 > 0.5 * CV_PI) theta0 -= 0.5 * CV_PI;
				if (abs(k_theta - theta0) > 5.0*CV_PI / 180.0 && abs(abs(k_theta - theta0) - CV_PI) > 5.0*CV_PI / 180.0)
					dogKernel.at<double>(r, c) = 0.0;
			}
		}
		/*normalize(dogKernel, dogKernel, 0, 1, NORM_MINMAX);
		string name = "dogkernel_" + to_string(i);
		namedWindow(name, 0);
		imshow(name, dogKernel); waitKey();*/
		filter2D(exatator, exataMat, depth, dogKernel);
		dstImg.back().forEach<double>([dstImg, i, beta, exataMat](double& p, const int* pos)->void {
			p = dstImg.back().at<double>(pos[0], pos[1]) + beta * exataMat.at<double>(pos[0], pos[1]);
			if (p < 0) p = 0; });
	}
}

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, double sigma, const vector<Mat>& kernel)
{
	int nh = srcImg[0].rows;
	int nw = srcImg[0].cols;
	int ntheta = srcImg.size();
	
	int depth = srcImg[0].depth();
	for (int i = 0; i < ntheta; i++)
	{
		Mat finalMat = srcImg.at(i).clone();
		for (int j = 0; j < ntheta; j++)
		{
			Mat tmpInhib;
			int kernelIndex = (j + i) % ntheta;
			filter2D(srcImg[i], tmpInhib, depth, kernel[kernelIndex]);
			finalMat.forEach<double>([tmpInhib](double& p, const int* pos)->void {p += tmpInhib.at<double>(pos[0], pos[1]); });
		}
		dstImg.push_back(finalMat);
	}
}

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, double sigma, const vector<Mat>& kernel, const vector<double>& coefs)
{
	int nh = srcImg[0].rows;
	int nw = srcImg[0].cols;
	int ntheta = srcImg.size();
	int kersize = kernel.size();
	int depth = srcImg[0].depth();
	Mat inhibitor(srcImg[0].size(), CV_64F, Scalar(0.0));
	for (int r = 0; r < nh; r++)
	{
		double* ptr = inhibitor.ptr<double>(r);
		for (int c = 0; c < nw; c++)
		{
			for (int i = 0; i < ntheta; i++)
			{
				if (supMethod == 1)
					ptr[c] += abs(srcImg[i].at<double>(r, c));
				if (supMethod == 2)
					ptr[c] += srcImg[i].at<double>(r, c) * srcImg[i].at<double>(r, c);
				if (supMethod == 3)
				{
					if (ptr[c] < srcImg[i].at<double>(r, c))
						ptr[c] = srcImg[i].at<double>(r, c);
				}
			}
			if (supMethod == 2)
				ptr[c] = sqrtf(ptr[c]);
		}
	}

	Mat inhibMat(nh, nw, CV_64F);
	//Mat dogKernel = DogKernel2d(sigma, 1, 4);
	Mat dogKernel = kernel[0];
	if (inhibMethod == 2)
		filter2D(inhibitor, inhibMat, depth, dogKernel);

	for (int i = 0; i< ntheta; i++)
	{
		/*Mat showImg = inhibitor.clone();
		normalize(showImg, showImg, 0.0, 1.0, NORM_MINMAX);
		imshow("inhib" + to_string(i), showImg);
		waitKey(0);*/

		Mat finalMat(nh, nw, CV_64F);//一个难找的bug, finalMat必须在循环体内部定义，因为finalMat用于push_back进输出参数，若定义在循环体外部，相当于push_back的是同一个对象，没有新的对象
										//当函数体退出时，会销毁函数体内部定义的变量。引用、指针除外。此处说明push_back进输出参数的对象也不会被销毁。
		double alpha = 1.0;
		finalMat.forEach<double>([srcImg, i, alpha, inhibMat](double& p, const int* pos)->void {
		p = srcImg[i].at<double>(pos[0], pos[1]) - alpha * inhibMat.at<double>(pos[0], pos[1]);
		if (p < 0) p = 0; });
		dstImg.push_back(finalMat);
	}
}

void ViewImage(Mat& dstImg, Mat& orienImg, const vector<Mat>& srcImg, const vector<double> theta)
{
	int nh = srcImg[0].rows;
	int nw = srcImg[0].cols;
	int ntheta = srcImg.size();
	double maxValue = -1000;
	int orienIndex = 0;
	dstImg = Mat(srcImg[0].rows, srcImg[0].cols, CV_64F, Scalar(0.0));
	orienImg = Mat(srcImg[0].rows, srcImg[0].cols, CV_64F, Scalar(0.0));
	for (int r = 0; r < nh; r++)
	{
		double* ptrDst = dstImg.ptr<double>(r);
		double* ptrOrien = orienImg.ptr<double>(r);
		for (int c = 0; c < nw; c++)
		{
			maxValue = -1000;
			orienIndex = 0;
			for (int i = 0; i < ntheta; i++)
			{
				if (srcImg[i].at<double>(r, c) > maxValue)
				{
					maxValue = srcImg[i].at<double>(r, c);
					orienIndex = i;
				}
			}
			ptrDst[c] = maxValue;
			ptrOrien[c] = theta[orienIndex];
		}
	}
	/*Mat showImg = dstImg.clone();
	normalize(showImg, showImg, 0, 1, NORM_MINMAX);
	imshow("viewImg", showImg);
	waitKey(0);*/
}

void Thinning(Mat& dstImg, const Mat& srcImg, const Mat& orienMat)
{
	dstImg = srcImg.clone();

	int nh = srcImg.rows;
	int nw = srcImg.cols;
	for (int r = 1; r < nh - 1; r++)
	{
		for (int c = 1; c < nw - 1; c++)
		{
			//if (dstImg.at<double>(r, c) < 0)//半波整流
			//	dstImg.at<double>(r, c) = 0;

			double angle = orienMat.at<double>(r, c);
			if (angle >= CV_PI)
				angle -= CV_PI;
			double v0 = srcImg.at<double>(r, c);
			if (angle >= 3.0 / 4.0 * CV_PI)//角度问题需要注意，theta中0°表示的gabor滤波器是纵向，y方向长于x方向
			{
				double fraction = (angle - 3.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
				double v1 = srcImg.at<double>(r, c - 1) * fraction + srcImg.at<double>(r - 1, c - 1) * (1 - fraction);
				double v2 = srcImg.at<double>(r, c + 1) * fraction + srcImg.at<double>(r + 1, c + 1) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<double>(r, c) = 0.0;
			}
			else if (angle >= 1.0 / 2.0 * CV_PI)
			{
				double fraction = (angle - 1.0 / 2.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
				double v1 = srcImg.at<double>(r - 1, c - 1) * fraction + srcImg.at<double>(r - 1, c) * (1 - fraction);
				double v2 = srcImg.at<double>(r + 1, c + 1) * fraction + srcImg.at<double>(r + 1, c) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<double>(r, c) = 0.0;
			}
			else if (angle >= 1.0 / 4.0 * CV_PI)
			{
				double fraction = (angle - 1.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
				double v1 = srcImg.at<double>(r + 1, c) * fraction + srcImg.at<double>(r + 1, c - 1) * (1 - fraction);
				double v2 = srcImg.at<double>(r - 1, c) * fraction + srcImg.at<double>(r - 1, c + 1) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<double>(r, c) = 0.0;
			}
			else if (angle >= 0.0)
			{
				double fraction = angle  / (1.0 / 4.0 * CV_PI);
				double v0 = srcImg.at<double>(r, c);
				double v1 = srcImg.at<double>(r - 1, c + 1) * fraction + srcImg.at<double>(r , c + 1) * (1 - fraction);
				double v2 = srcImg.at<double>(r + 1, c - 1) * fraction + srcImg.at<double>(r, c - 1) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<double>(r, c) = 0.0;
			}			
		}
	}
}

void Hysteresis(Mat& srcImg, double tlow, double thigh)
{
	//Mat padSrc(srcImg.rows + 1, srcImg.cols + 1, CV_64F, Scalar(0.0));
	//srcImg.copyTo(padSrc(Rect(1, 1, srcImg.rows, srcImg.cols)));
	
	double maxVal, minVal;
	minMaxLoc(srcImg, &minVal, &maxVal);
	if (maxVal != minVal)
		srcImg.forEach<double>([minVal, maxVal](double &p, const int * position)->void {p = (p - minVal) / (maxVal - minVal); });//forEach is parralel, can increase performance
	srcImg(Range(0,1), Range(0, srcImg.cols)) = 0.0;//图像周边置0
	srcImg(Range(srcImg.rows - 1, srcImg.rows), Range(0, srcImg.cols)) = 0.0;
	srcImg(Range(0, srcImg.rows), Range(0, 1)) = 0.0;
	srcImg(Range(0, srcImg.rows), Range(srcImg.cols - 1, srcImg.cols)) = 0.0;
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

void InvertImg(Mat& srcImg)
{
	srcImg.forEach<double>([](double &p, const int * position)->void {p = -p;});
	double maxVal, minVal;
	minMaxLoc(srcImg, &minVal, &maxVal);
	if (maxVal != minVal)
		srcImg.forEach<double>([minVal, maxVal](double &p, const int * position)->void {p = (p - minVal) / (maxVal - minVal); });
}

void Evaluate(double& p, double& efp, double& efn, const Mat& rstImg, const Mat& gtImg, int neigbSize)
{
	int padsz = (neigbSize - 1) / 2;
	Mat padRst(rstImg.rows + padsz, rstImg.cols + padsz, rstImg.type(), Scalar(1.0));
	Mat padGt(gtImg.rows + padsz, gtImg.cols + padsz, gtImg.type(), Scalar(1.0));
	rstImg.copyTo(padRst(Rect(1, 1, rstImg.rows, rstImg.cols)));
	gtImg.copyTo(padGt(Rect(1, 1, gtImg.rows, gtImg.cols)));
	int E = 0;//true positive numbers
	int Efp = 0;
	for (int r = padsz; r < padRst.rows - padsz; r++)
	{
		for (int c = padsz; c < padRst.cols - padsz; c++)
		{
			if (abs(padRst.at<double>(r, c)) < 0.0001)
			{
				bool hasNegbor = false;
				for (int x = -padsz; x <= padsz; x++)
				{
					if (hasNegbor)
						break;
					for (int y = -padsz; y <= padsz; y++)
					{
						if (abs(padGt.at<double>(r + x, c + y)) < 0.0001)
						{
							E += 1;
							hasNegbor = true;
							padGt.at<double>(r + x, c + y) = 1;
							break;
						}
					}
				}
				if (!hasNegbor)
					Efp += 1;
			}
		}
	}
	int Efn = 0;
	padGt.forEach<double>([&Efn](double &p, const int * position)->void {if (abs(p) < 0.0001) { Efn += 1; }; });
	p = E * 1.0 / (E + Efp + Efn);
	efp = Efp * 1.0 / E;
	int Egt = 0;
	gtImg.forEach<double>([&Egt](double &p, const int * position)->void {if (abs(p) < 0.0001) { Egt += 1; }; });
	efn = Efn * 1.0 / Egt;
}

double NonCRF(Mat srcImg, Mat gtImg)
{
	if (srcImg.type() != CV_64F)
		srcImg.convertTo(srcImg, CV_64F, 1 / 255.0);
	if (gtImg.type() != CV_64F)
		gtImg.convertTo(gtImg, CV_64F, 1 / 255.0);
	bool halfwave = 1;
	double lamda = 10;//wavelength
	//vector<double> lamdaVec = { 3,4,5,6,7,8,9,10,11,12,13,15,15,16,17,18,19,21 };
	vector<double> lamdaVec = { 3,5,7,9,11,13,15,17,19,21,23, 25, 37 };// , 25, 27, 29, 31, 33, 35};

	vector<double> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);
	double sigma = 1.0;
	double gamma = 0.3;//aspect ratio
	double bandwidth = 1;
	int ntheta = 16;
	int nphi = 2;
	double arrphi[2] = { 0, 0.5*CV_PI};
	vector<double> phi = { arrphi, arrphi + 2 };
	vector<double> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	int supPhases = 2;
	int inhibMethod = 2;//2:isotropic, 3:antisotropic, 1:no inhibition
	int inhibSup = 3;
	double alpha = 1.0;
	double k1 = 1;
	double k2 = 4;
	double tlow = 0.2;
	double thigh = 0.3;
	double p, efp, efn;
	vector<vector<Mat>> gaborImgs;
	//GaborFilter(gaborImgs, srcImg, halfwave, lamda, sigma, theta, phi, gamma, bandwidth);
	GaborFilterMulScale(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);
	//GaborFilterMulScaleBend(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);

	vector<Mat> phaseSupImgs;
	PhaseSuppos(phaseSupImgs, gaborImgs, ntheta, nphi, supPhases);
	
	vector<Mat> inhibImgs;
	Inhibition(inhibImgs, phaseSupImgs, inhibMethod, inhibSup, sigma, alpha, k1, k2);	
	//Mat showImg1 = inhibImgs[0].clone();
	//normalize(showImg1, showImg1, 0, 1, NORM_MINMAX);
	//imshow("inhibImgs", showImg1);
	//waitKey(0);

	Mat viewImg, orienImg;
	ViewImage(viewImg, orienImg, inhibImgs, theta);			
	//Mat showImg = orienImg.clone();
	//normalize(showImg, showImg, 0, 1, NORM_MINMAX);
	//imshow("inhibImgs", showImg);
	//imwrite("viewImg.pgm", viewImg);
	//imwrite("viewori.pgm", orienImg);
	//waitKey(0);
	
	Mat showImg2 = viewImg.clone();
	normalize(showImg2, showImg2, 0, 1, NORM_MINMAX);
	imshow("viewImgmatlab", showImg2);
	waitKey(0);

	Mat thinImg;
	Thinning(thinImg, viewImg, orienImg);
	Mat showImg1 = thinImg.clone();
	normalize(showImg1, showImg1, 0, 1, NORM_MINMAX);
	imshow("thinImg", showImg1);
	waitKey(0);

	Hysteresis(thinImg, tlow, thigh);
	
	InvertImg(thinImg);

	//Evaluate(p, efp, efn, thinImg, gtImg, 5);
	//cout << "p: " << p << ", efp: " << efp << ", efn: " << efn << endl;
	//namedWindow("NCRF image", 0);
	imshow("NCRF image", thinImg);
	imshow("ground truth", gtImg);
	waitKey(0);
	return p;

}

double NonCRF(Mat srcImg, Mat gtImg, vector<Mat> kernel, bool isDisplay)
{
	if (srcImg.type() != CV_64F)
		srcImg.convertTo(srcImg, CV_64F, 1 / 255.0);
	if (gtImg.type() != CV_64F)
		gtImg.convertTo(gtImg, CV_64F, 1 / 255.0);
	bool halfwave = 1;
	double lamda = 10;//wavelength
	double sigma = 1.0;
	double gamma = 0.5;//aspect ratio

	vector<double> lamdaVec = { 3,5,7,9,11,13,15,17,19,21,23,25,27 };
	vector<double> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);

	double bandwidth = 1;
	int ntheta = 8;
	int nphi = 2;
	double arrphi[2] = { 0, 0.5*CV_PI };
	vector<double> phi = { arrphi, arrphi + 2 };
	vector<double> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	int supPhases = 1;
	int inhibMethod = 2;//isotropic or antisotropic
	int inhibSup = 3;
	double alpha = 1.0;
	double k1 = 1;
	double k2 = 4;
	double tlow = 0.05;
	double thigh = 0.2;
	double p, efp, efn;
	vector<vector<Mat>> gaborImgs;
	GaborFilter(gaborImgs, srcImg, halfwave, lamda, sigma, theta, phi, gamma, bandwidth);
	//GaborFilterMulScale(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);
	vector<Mat> phaseSupImgs;
	PhaseSuppos(phaseSupImgs, gaborImgs, ntheta, nphi, supPhases);

	vector<Mat> inhibImgs;
	Inhibition(inhibImgs, phaseSupImgs, inhibMethod, inhibSup, sigma, kernel);
	/*vector<Mat> inhibImgs1;
	Inhibition(inhibImgs1, inhibImgs, inhibMethod, inhibSup, sigma, kernel);
	vector<Mat> inhibImgs2;
	Inhibition(inhibImgs2, inhibImgs1, inhibMethod, inhibSup, sigma, kernel);
	inhibImgs = inhibImgs2;*/

	Mat viewImg, orienImg;
	ViewImage(viewImg, orienImg, inhibImgs, theta);


	Mat thinImg;
	Thinning(thinImg, viewImg, orienImg);

	Hysteresis(thinImg, tlow, thigh);

	InvertImg(thinImg);

	Evaluate(p, efp, efn, thinImg, gtImg, 5);
	
	if (isDisplay)// && p > 0.1)
	{
		cout << "p: " << p << ", efp: " << efp << ", efn: " << efn << endl;
		//namedWindow("NCRF image", 0);
		imshow("NCRF image", thinImg);
		//imshow("ground truth", gtImg);
		waitKey(1);
	}
	return p;
}

double NonCRF(Mat srcImg, Mat gtImg, const vector<Mat>& kernel, const vector<double> coefs, bool isDisplay)
{
	if (srcImg.type() != CV_64F)
		srcImg.convertTo(srcImg, CV_64F, 1 / 255.0);
	if (gtImg.type() != CV_64F)
		gtImg.convertTo(gtImg, CV_64F, 1 / 255.0);
	bool halfwave = 1;
	double lamda = 10;//wavelength
	double sigma = 1.0;
	double gamma = 0.5;//aspect ratio

	vector<double> lamdaVec = { 3,5,7,9,11,13,15,17,19,21};
	vector<double> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);

	double bandwidth = 1;
	int ntheta = 12;
	int nphi = 2;
	double arrphi[2] = { 0, 0.5*CV_PI };
	vector<double> phi = { arrphi, arrphi + 2 };
	vector<double> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	int supPhases = 1;
	int inhibMethod = 2;//isotropic or antisotropic
	int inhibSup = 3;
	double alpha = 1.0;
	double k1 = 1;
	double k2 = 4;
	double tlow = 0.05;
	double thigh = 0.25;
	double p, efp, efn;
	vector<vector<Mat>> gaborImgs;
	GaborFilter(gaborImgs, srcImg, halfwave, lamda, sigma, theta, phi, gamma, bandwidth);
	//GaborFilterMulScale(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);
	vector<Mat> phaseSupImgs;
	PhaseSuppos(phaseSupImgs, gaborImgs, ntheta, nphi, supPhases);

	vector<Mat> inhibImgs;
	//Inhibition(inhibImgs, phaseSupImgs, inhibMethod, inhibSup, sigma, alpha, k1, k2);
	//Inhibition(inhibImgs, phaseSupImgs, inhibMethod, inhibSup, sigma, kernel);
	Inhibition(inhibImgs, phaseSupImgs, inhibMethod, inhibSup, sigma, kernel, coefs);
	/*vector<Mat> inhibImgs1;
	Inhibition(inhibImgs1, inhibImgs, inhibMethod, inhibSup, sigma, kernel);
	vector<Mat> inhibImgs2;
	Inhibition(inhibImgs2, inhibImgs1, inhibMethod, inhibSup, sigma, kernel);
	inhibImgs = inhibImgs2;*/

	Mat viewImg, orienImg;
	ViewImage(viewImg, orienImg, inhibImgs, theta);


	Mat thinImg;
	Thinning(thinImg, viewImg, orienImg);

	Hysteresis(thinImg, tlow, thigh);

	InvertImg(thinImg);

	Evaluate(p, efp, efn, thinImg, gtImg, 5);

	if (isDisplay)// && p > 0.1)
	{
		//cout << "p: " << p << ", efp: " << efp << ", efn: " << efn << endl;
		//namedWindow("NCRF image", 0);
		//imshow("NCRF image", thinImg);
		//imshow("ground truth", gtImg);
		//waitKey(1);
	}
	return p;
}

//function used inside
Mat GaborKernel2d(double sigma, double lamda, double theta, double phi, double gamma, double bandwidth)
{
	double slratio = (1.0 / CV_PI) * sqrtf((log(2.0) / 2.0)) * ((pow(2, bandwidth) + 1) * 1.0 / (pow(2, bandwidth) - 1));
	if (sigma == 0)
		sigma = slratio * lamda;
	else if (lamda == 0)
		lamda = sigma / slratio;
	int n;
	if (gamma <= 1 && gamma > 0)
		n = int(ceilf(2.5 * sigma / gamma));
	else
		n = int(ceilf(2.5 * sigma));	
	double gamma2 = gamma * gamma;
	double b = 1.0 / (2 * sigma * sigma);
	double a = b / CV_PI;
	double f = 2.0 * CV_PI / lamda;	
	double posSum = 0;
	double negSum = 0;
	Mat kernel(2 * n + 1, 2 * n + 1, CV_64F);
	for (int r = 0; r < 2*n + 1; r++)
	{
		double* ptr = kernel.ptr<double>(r);
		for (int c = 0; c < 2*n + 1; c++)
		{
			double xp = -(c - n) * cos(theta) + (r - n) * sin(theta);//由于filter2D实际上做的是相关而非卷积，因此要把和旋转180°，x,y都要反号。同时，图像坐标中Y轴朝下，y再次反号
			double yp = (c - n) * sin(theta) + (r - n) * cos(theta);
			ptr[c] = a * exp(-b * (xp * xp + gamma2 * (yp * yp))) * cos(f * xp + phi);
			if (ptr[c] > 0)
				posSum += ptr[c];
			else
				negSum += ptr[c];
		}
	}
	negSum = abs(negSum);
	double meanSum = (posSum + negSum) / 2.0;
	if (meanSum > 0)//归一化系数
	{
		posSum /= meanSum;
		negSum /= meanSum;
	}
	
	for (int r = 0; r < 2 * n + 1; r++)//正负平衡
	{
		double* ptr = kernel.ptr<double>(r);
		for (int c = 0; c < 2 * n + 1; c++)
		{	
			if (ptr[c] > 0)
				ptr[c] = negSum * ptr[c];
			else
				ptr[c] = posSum * ptr[c];
		}
	}

	/*Mat show1 = kernel.clone();
	normalize(show1, show1, 0, 1, NORM_MINMAX);
	namedWindow("gabor kernel", 0);
	imshow("gabor kernel", show1);          
	waitKey(0);*/
	return kernel;
}

Mat GaborKernel2d(double sigma, double lamda, double theta, double phi, double gamma, double bandwidth, double radiu)
{
	return BendGabor2d(sigma, lamda, theta, phi, gamma, bandwidth, radiu);
}


Mat BendGabor2d(double sigma, double lamda, double theta, double phi, double gamma, double bandwidth, double radiu)
{
	double slratio = (1.0 / CV_PI) * sqrtf((log(2.0) / 2.0)) * ((pow(2, bandwidth) + 1) * 1.0 / (pow(2, bandwidth) - 1));
	if (sigma == 0)
		sigma = slratio * lamda;
	else if (lamda == 0)
		lamda = sigma / slratio;
	int n;
	if (gamma <= 1 && gamma > 0)
		n = int(ceilf(2.5 * sigma / gamma));
	else
		n = int(ceilf(2.5 * sigma));
	n = n * 2;
	double gamma2 = gamma * gamma;
	double b = 1.0 / (2 * sigma * sigma);
	double a = b / CV_PI;
	double f = 2.0 * CV_PI / lamda;
	double posSum = 0;
	double negSum = 0;
	Mat kernel(2 * n + 1, 2 * n + 1, CV_64F);
	for (int r = 0; r < 2 * n + 1; r++)
	{
		double* ptr = kernel.ptr<double>(r);
		for (int c = 0; c < 2 * n + 1; c++)
		{
			double xp = -(c - n) * cos(theta) + (r - n) * sin(theta);//由于filter2D实际上做的是相关而非卷积，因此要把和旋转180°，x,y都要反号。同时，图像坐标中Y轴朝下，y再次反号
			double yp = (c - n) * sin(theta) + (r - n) * cos(theta);
			ptr[c] = a * exp(-b * (xp * xp + gamma2 * (yp * yp))) * cos(f * xp + phi);
			
		}
	}
	n = n / 2;
	int kersz = 2 * n + 1;
	Point2f center(n, n);
	Point2f circleCenter(n, n + abs(radiu));//圆心位置
	Mat xhh(kersz, kersz, CV_64F, Scalar(0));//小灰灰
	for (int r = 0; r < kersz; r++)
	{
		for (int c = 0; c < kersz; c++)
		{
			
			/*double radiu1 = radiu - (c - center.y);
			double len = r - center.y;
			double theta = len / radiu1;
			int y1 = center.y + (c - center.y) + radiu1 * sin(theta) * sin(theta);
			int x1 = center.x + radiu1 * sin(theta) * cos(theta);
			if (x1 < 0 || x1 >= kersz || y1 < 0 || y1 >= kersz)
				continue;
			xhh.at<double>(x1, y1) = kernel.at<double>(r+n, c+n);*/

			double dx = r - circleCenter.x;
			double dy = c - circleCenter.y;
			//if (radiu < 0) dy *= -1;
			double radiu1 = sqrt(dx*dx + dy*dy);
			double theta = atan(dx / dy);
			//if (theta < 0) theta += 2 * CV_PI;
			double len = radiu1 * theta;
			int x0 = circleCenter.x - len + 0.5;//关键的0.5！ 四舍五入
			int y0 = circleCenter.y - radiu1 + 0.5;
			//if(radiu < 0) y0 = radiu1 - circleCenter.y  + 0.5;
			if (x0 < 0 || x0 >= kernel.rows || y0 < 0 || y0 > kernel.cols)
				continue;
			if (1)// radiu >= 0)
			{
				xhh.at<double>(r, c) = kernel.at<double>(x0 + n, y0 + n);
				if (xhh.at<double>(r, c) > 0)
					posSum += xhh.at<double>(r, c);
				else
					negSum += xhh.at<double>(r, c);
			}
			else
			{
				xhh.at<double>(r, n + (n - c)) = kernel.at<double>(x0 + n, y0 + n);
				if (xhh.at<double>(r, n + (n - c)) > 0)
					posSum += xhh.at<double>(r, n + (n - c));
				else
					negSum += xhh.at<double>(r, n + (n - c));
			}

			
		}
	}
	Mat result = xhh.clone();
	if (radiu < 0)
	{
		Mat roteMat = getRotationMatrix2D(Point2f(center),CV_PI, 1.0);
		Mat result;
		warpAffine(xhh, result, roteMat, xhh.size());
	}
	

	negSum = abs(negSum);
	double meanSum = (posSum + negSum) / 2.0;
	if (meanSum > 0)//归一化系数
	{
		posSum /= meanSum;
		negSum /= meanSum;
	}

	for (int r = 0; r < 2 * n + 1; r++)//正负平衡
	{
		double* ptr = result.ptr<double>(r);
		for (int c = 0; c < 2 * n + 1; c++)
		{
			if (ptr[c] > 0)
				ptr[c] = negSum * ptr[c];
			else
				ptr[c] = posSum * ptr[c];
		}
	}

	//Mat show1 = xhh.clone();
	//normalize(show1, show1, 0, 1, NORM_MINMAX);
	//namedWindow("gabor kernel", 0);
	//imshow("gabor kernel", show1);
	//waitKey(0);

	
	//normalize(xhh, xhh, 0, 1, NORM_MINMAX);
	/*Mat roteMat = getRotationMatrix2D(Point2f(center), theta * 180 / (2 * CV_PI), 1.0);
	Mat result;
	warpAffine(xhh, result, roteMat, xhh.size());*/

	/*Mat show2 = xhh.clone();
	normalize(show2, show2, 0, 1, NORM_MINMAX);
	namedWindow("gabor bend", 0);
	imshow("gabor bend", show2);
	waitKey(0);*/
	return result;
}

Mat DogKernel2d(double sigma, int k1, int k2)
{
	int n = ceil(sigma) * (3 * k2 + k1) - 1;
	int kerSz = 2 * n + 1;
	Mat kernel(kerSz, kerSz, CV_64F);
	double norm_L1 = 0.0;
	for (int r = 0; r < kerSz; r++)
	{
		for (int c = 0; c < kerSz; c++)
		{
			int x = c - n;
			int y = r - n;
			double sigma1 = k2 * sigma;
			double sigma2 = k1  * sigma;
			kernel.at<double>(r, c) = exp((-(x*x + y*y)) / (2.0 * sigma1*sigma1)) / (2.0 * CV_PI*sigma1*sigma1) 
				- exp((-(x*x + y*y)) / (2.0 * sigma2*sigma2)) / (2.0 * CV_PI*sigma2*sigma2);
			if (kernel.at<double>(r, c) < 0)
				kernel.at<double>(r, c) = 0;
			norm_L1 += kernel.at<double>(r, c);
		}
	}
	kernel.forEach<double>([norm_L1](double& p, const int*)->void {
		if (abs(norm_L1) > 0.0001) p = p / norm_L1;
		else p = 0; });
	return kernel;
}

Mat createFftImage(const Mat& srcImg, int fh, int fw)
{
	int imh = srcImg.rows;
	int imw = srcImg.cols;

	int nh = max(fh, imh) + max(fh, imh) % 2;
	int nw = max(fw, imw) + max(fw, imw) % 2;

	Mat fftImg;
	return fftImg;
}

Mat convolution(const Mat& srcImg, const Mat& filterKernel, const Mat& fftImg)
{
	Mat fftKernel = createFftImage(filterKernel, filterKernel.rows, filterKernel.cols);
	Mat convResult = fftImg.mul(fftKernel);//Matrix dot multiply
	
	Mat dstImg;
	return dstImg;
}

//--------fastener detection
Mat NonCrf_fastener(Mat srcImg, int inhibModel, double tl, double th)
{
	if (srcImg.type() != CV_64F)
		srcImg.convertTo(srcImg, CV_64F, 1 / 255.0);
	bool halfwave = 1;
	double lamda = 10;//wavelength
					 //vector<double> lamdaVec = { 3,4,5,6,7,8,9,10,11,12,13,15,15,16,17,18,19,21 };
	vector<double> lamdaVec = /*{ 3,5,7,9,11, */{/*13, 15, 17, 19, 21, 23, */31};/* , 27, 29, 31, 33, 35};*/

	vector<double> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);
	double sigma = 5.0;
	double gamma = 0.5;//aspect ratio
	double bandwidth = 1;
	int ntheta = 12;
	int nphi = 2;
	double arrphi[2] = { 0, 0.5*CV_PI };
	vector<double> phi = { arrphi, arrphi + 2 };
	vector<double> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	int supPhases = 2;
	int inhibMethod = inhibModel;//2:isotropic, 3:antisotropic, 1:no inhibition
	int inhibSup = 3;
	double alpha = 1.0;
	double k1 = 2;
	double k2 = 4;
	double tlow = tl;
	double thigh = th;
	double p, efp, efn;
	vector<vector<Mat>> gaborImgs;
	//GaborFilter(gaborImgs, srcImg, halfwave, lamda, sigma, theta, phi, gamma, bandwidth);
	GaborFilterMulScale(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);
	//GaborFilterMulScaleBend(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);

	vector<Mat> phaseSupImgs;
	PhaseSuppos(phaseSupImgs, gaborImgs, ntheta, nphi, supPhases);

	vector<Mat> inhibImgs;
	Inhibition(inhibImgs, phaseSupImgs, inhibMethod, inhibSup, sigma, alpha, k1, k2);

	Mat viewImg, orienImg;
	ViewImage(viewImg, orienImg, inhibImgs, theta);

	Mat showImg2 = viewImg.clone();
	normalize(showImg2, showImg2, 0, 1, NORM_MINMAX);
	//imshow("viewImgmatlab", showImg2);
	//waitKey(0);

	Mat thinImg;
	Thinning(thinImg, viewImg, orienImg);
	Mat showImg1 = thinImg.clone();
	normalize(showImg1, showImg1, 0, 1, NORM_MINMAX);
	//imshow("thinImg", showImg1);
	//waitKey(0);
	Hysteresis(thinImg, tlow, thigh);
	InvertImg(thinImg);
	//imshow("NCRF image", thinImg);
	//waitKey(0);
	return thinImg;

}

Mat drawButterflyNCRF()
{
	double ring1 = 42;
	double ring2 = 150;
	double theta = 70.0 / 180.0 * CV_PI;
	Mat resultImg(600, 600, CV_64F, Scalar(1.0));
	int centx = 300;
	int centy = 300;

	Mat RF = GaborKernel2d(8.5, 30, 0, 0, 0.5, 1);
	normalize(RF, RF, 0, 1, NORM_MINMAX);
	Mat dog1 = DogKernel2d(12, 1,8);
	normalize(dog1, dog1, 0, 1, NORM_MINMAX);
	imshow("dog1", dog1); waitKey();
	Mat dogsave = dog1.clone();
	dogsave.convertTo(dogsave, CV_8U, 255);
	imwrite("dog.jpg", dogsave);
	Mat dog2 = dog1.clone();
	dog2.forEach<double>([](double& p, const int* pos)
		->void {p = abs(p - 1); });
	normalize(dog2, dog2, 0, 1, NORM_MINMAX);
	imshow("dog1", dog2); waitKey();
	for (int r = 0; r < dog2.rows; r++)
	{
		for (int c = 0; c < dog2.cols; c++)
		{
			double digree = 60.0;
			if (abs((r - centx)*1.0 / (c - centy)) > tan(digree / 180.0 * CV_PI))
				dog2.at<double>(r, c) = abs(1 - dog2.at<double>(r, c));
			if (sqrt((r - centx)*(r - centx) + (c - centy)*(c - centy)) < ring1)
				dog2.at<double>(r, c) = RF.at<double>(r - centx + ring1, c - centy + ring1);
			if (sqrt((r - centx)*(r - centx) + (c - centy)*(c - centy)) > ring2)
				dog2.at<double>(r, c) = 1;
		}
	}
	imshow("dog1", dog2); waitKey();
	

	return resultImg;
}

Mat drawRNCRF()
{
	double ring1 = 42;
	double ring2 = 150;
	double theta = 70.0 / 180.0 * CV_PI;
	Mat resultImg(600, 600, CV_64F, Scalar(1.0));
	int centx = 300;
	int centy = 300;

	Mat RF = GaborKernel2d(8.5, 30, 0, 0, 0.5, 1);
	normalize(RF, RF, 0, 1, NORM_MINMAX);
	Mat dog1 = DogKernel2d(12, 1, 8);
	normalize(dog1, dog1, 0, 1, NORM_MINMAX);
	imshow("dog1", dog1); waitKey();
	Mat dogsave = dog1.clone();
	dogsave.convertTo(dogsave, CV_8U, 255);
	imwrite("dog.jpg", dogsave);
	Mat dog2 = dog1.clone();
	dog2.forEach<double>([](double& p, const int* pos)
		->void {p = abs(p - 1); });
	normalize(dog2, dog2, 0, 1, NORM_MINMAX);
	imshow("dog1", dog2); waitKey();
	for (int r = 0; r < dog2.rows; r++)
	{
		for (int c = 0; c < dog2.cols; c++)
		{
			//double digree = 89.0;
			//if (abs((r - centx)*1.0 / (c - centy)) > tan(digree / 180.0 * CV_PI))
			//	dog2.at<double>(r, c) = abs(1 - dog2.at<double>(r, c));
			if(c == centy)
				dog2.at<double>(r, c) = abs(1 - dog2.at<double>(r, c));
			int R = 300;
			int RX = centx;
			int RY1 = centy + R;
			int RY2 = centy - R;
			double eps = 1.0;
			if(abs(sqrt((r - RX)*(r - RX)+(c - RY1)*(c - RY1))-R) < 1.0 || abs(sqrt((r - RX)*(r - RX) + (c - RY2)*(c - RY2)) - R) < 1.0)
				dog2.at<double>(r, c) = abs(1 - dog2.at<double>(r, c));
			if (sqrt((r - centx)*(r - centx) + (c - centy)*(c - centy)) < ring1)
				dog2.at<double>(r, c) = RF.at<double>(r - centx + ring1, c - centy + ring1);
			if (sqrt((r - centx)*(r - centx) + (c - centy)*(c - centy)) > ring2)
				dog2.at<double>(r, c) = 1;
		}
	}
	imshow("dog1", dog2); waitKey();
	//normalize(dog2, dog2, 0, 1, NORM_MINMAX);

	dog2.convertTo(dog2, CV_8U, 255);
	imwrite("pic/R-NCRF.jpg", dog2);
	imshow("rncrf", dog2); waitKey();

	return resultImg;
}