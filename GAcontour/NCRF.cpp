#include "NCRF.h"
#include <stack>

int GaborFilter(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave, 
	float lamda, float sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth)
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
	Mat kernel(kersz, kersz, CV_32F);
	vector<Mat> phiVec;
	for (int i = 0; i < ntheta; i++)
	{
		for (int j = 0; j < nphi; j++)
		{
			GaborKernel2d(kernel, sigma, lamda, theta[i], phi[j], gamma, bandwidth);
			
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
					float* ptr = tmpResult.ptr<float>(r);
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
	vector<float> lamda, vector<float> sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth)
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
				Mat kernel(kersz, kersz, CV_32F);
				GaborKernel2d(kernel, sigma.at(s), lamda.at(s), theta[i], phi[j], gamma, bandwidth);				
				Mat scaleMat;
				filter2D(srcImg, scaleMat, srcImg.depth(), kernel);
				if (s == 0)
					tmpResult = scaleMat.clone();
				else
				{
					tmpResult.forEach<float>([scaleMat](float& p, const int* pos)->void { p = p + scaleMat.at<float>(pos[0], pos[1]); });
				}
			}
			int sz = lamda.size();
			tmpResult.forEach<float>([sz](float& p, const int* pos)->void {p = p * 1.0 / sz; });

			if (halfwave)//半波整流
			{
				for (int r = 0; r < tmpResult.rows; r++)
				{
					float* ptr = tmpResult.ptr<float>(r);
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
			Mat tmpImg(nh, nw, CV_32F);
			for (int r = 0; r < nh; r++)
			{
				float* ptr = tmpImg.ptr<float>(r);			
				for (int c = 0; c < nw; c++)
				{
					ptr[c] = 0;
					for (int j = 0; j < nphi; j++)
					{
						const float* phi1_ptr = srcImg[i][j].ptr<float>(r);
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
			Mat tmpImg(nh, nw, CV_32F);
			for (int r = 0; r < nh; r++)
			{
				float* ptr = tmpImg.ptr<float>(r);
				for (int c = 0; c < nw; c++)
				{
					ptr[c] = 0;
					for (int j = 0; j < nphi; j++)
					{
						ptr[c] += srcImg[i][j].at<float>(r, c) * srcImg[i][j].at<float>(r, c);
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
			Mat tmpImg(nh, nw, CV_32F);
			for (int r = 0; r < nh; r++)
			{
				float* ptr = tmpImg.ptr<float>(r);
				for (int c = 0; c < nw; c++)
				{
					ptr[c] = 0;
					for (int j = 0; j < nphi; j++)
					{
						const float* phi1_ptr = srcImg[i][j].ptr<float>(r);
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

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float sigma, float alpha, int k1, int k2)
{
	int nh = srcImg[0].rows;
	int nw = srcImg[0].cols;
	int ntheta = srcImg.size();
	float initValue = 0.0;
	if (supMethod == 3)//max sup, infinit norm
		initValue = -1000.0;//-INF

	Mat inhibitor(srcImg[0].size(), CV_32F, Scalar(initValue));
	if (inhibMethod == 2)
	{
		for (int r = 0; r < nh; r++)
		{
			float* ptr = inhibitor.ptr<float>(r);
			for (int c = 0; c < nw; c++)
			{
				for (int i = 0; i < ntheta; i++)
				{
					if (supMethod == 1)
						ptr[c] += abs(srcImg[i].at<float>(r, c));
					if (supMethod == 2)
						ptr[c] += srcImg[i].at<float>(r, c) * srcImg[i].at<float>(r, c);
					if (supMethod == 3)
					{
						if (ptr[c] < srcImg[i].at<float>(r, c))
							ptr[c] = srcImg[i].at<float>(r, c);
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
	
	Mat inhibMat(nh, nw, CV_32F);

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

			Mat finalMat(nh, nw, CV_32F);//一个难找的bug, finalMat必须在循环体内部定义，因为finalMat用于push_back进输出参数，若定义在循环体外部，相当于push_back的是同一个对象，没有新的对象
										 //当函数体退出时，会销毁函数体内部定义的变量。引用、指针除外。此处说明push_back进输出参数的对象也不会被销毁。
			
			finalMat.forEach<float>([srcImg,i,alpha,inhibMat](float& p, const int* pos)->void {
				p = srcImg[i].at<float>(pos[0], pos[1]) - alpha * inhibMat.at<float>(pos[0], pos[1]);
				if (p < 0) p = 0; });//Set negtive value to zero!
			/*for (int r = 0; r < srcImg[i].rows; r++)
			{
				for (int c = 0; c < srcImg[i].cols; c++)
				{
					finalMat.at<float>(r, c) = srcImg[i].at<float>(r, c) - alpha * inhibMat.at<float>(r, c);
					if (finalMat.at<float>(r, c) < 0)
						finalMat.at<float>(r, c) = 0;
				}
			}*/
			dstImg.push_back(finalMat);	

		}
		if (inhibMethod == 3)// anisotropic inhibition
		{
			Mat finalMat(nh, nw, CV_32F);
			filter2D(srcImg[i], inhibMat, depth, dogKernel);
			finalMat.forEach<float>([srcImg, i, alpha, inhibMat](float& p, const int* pos)->void {
				p = srcImg[i].at<float>(pos[0], pos[1]) - alpha * inhibMat.at<float>(pos[0], pos[1]); 
				if (p < 0) p = 0; });
			dstImg.push_back(finalMat);
		}
		if (inhibMethod == 1)// no inhibition
		{
			dstImg.push_back(srcImg[i]);
		}
	}
}

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float sigma, const vector<Mat>& kernel)
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
			finalMat.forEach<float>([tmpInhib](float& p, const int* pos)->void {p += tmpInhib.at<float>(pos[0], pos[1]); });
		}
		dstImg.push_back(finalMat);
	}
}

void ViewImage(Mat& dstImg, Mat& orienImg, const vector<Mat>& srcImg, const vector<float> theta)
{
	int nh = srcImg[0].rows;
	int nw = srcImg[0].cols;
	int ntheta = srcImg.size();
	float maxValue = -1000;
	int orienIndex = 0;
	dstImg = Mat(srcImg[0].rows, srcImg[0].cols, CV_32F, Scalar(0.0));
	orienImg = Mat(srcImg[0].rows, srcImg[0].cols, CV_32F, Scalar(0.0));
	for (int r = 0; r < nh; r++)
	{
		float* ptrDst = dstImg.ptr<float>(r);
		float* ptrOrien = orienImg.ptr<float>(r);
		for (int c = 0; c < nw; c++)
		{
			maxValue = -1000;
			orienIndex = 0;
			for (int i = 0; i < ntheta; i++)
			{
				if (srcImg[i].at<float>(r, c) > maxValue)
				{
					maxValue = srcImg[i].at<float>(r, c);
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
			//if (dstImg.at<float>(r, c) < 0)//半波整流
			//	dstImg.at<float>(r, c) = 0;

			float angle = orienMat.at<float>(r, c);
			if (angle >= CV_PI)
				angle -= CV_PI;
			float v0 = srcImg.at<float>(r, c);
			if (angle >= 3.0 / 4.0 * CV_PI)//角度问题需要注意，theta中0°表示的gabor滤波器是纵向，y方向长于x方向
			{
				float fraction = (angle - 3.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
				float v1 = srcImg.at<float>(r, c - 1) * fraction + srcImg.at<float>(r - 1, c - 1) * (1 - fraction);
				float v2 = srcImg.at<float>(r, c + 1) * fraction + srcImg.at<float>(r + 1, c + 1) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<float>(r, c) = 0.0;
			}
			else if (angle >= 1.0 / 2.0 * CV_PI)
			{
				float fraction = (angle - 1.0 / 2.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
				float v1 = srcImg.at<float>(r - 1, c - 1) * fraction + srcImg.at<float>(r - 1, c) * (1 - fraction);
				float v2 = srcImg.at<float>(r + 1, c + 1) * fraction + srcImg.at<float>(r + 1, c) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<float>(r, c) = 0.0;
			}
			else if (angle >= 1.0 / 4.0 * CV_PI)
			{
				float fraction = (angle - 1.0 / 4.0 * CV_PI) / (1.0 / 4.0 * CV_PI);
				float v1 = srcImg.at<float>(r + 1, c) * fraction + srcImg.at<float>(r + 1, c - 1) * (1 - fraction);
				float v2 = srcImg.at<float>(r - 1, c) * fraction + srcImg.at<float>(r - 1, c + 1) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<float>(r, c) = 0.0;
			}
			else if (angle >= 0.0)
			{
				float fraction = angle  / (1.0 / 4.0 * CV_PI);
				float v0 = srcImg.at<float>(r, c);
				float v1 = srcImg.at<float>(r - 1, c + 1) * fraction + srcImg.at<float>(r , c + 1) * (1 - fraction);
				float v2 = srcImg.at<float>(r + 1, c - 1) * fraction + srcImg.at<float>(r, c - 1) * (1 - fraction);
				if (v0 < v1 || v0 <= v2)
					dstImg.at<float>(r, c) = 0.0;
			}			
		}
	}
}

void Hysteresis(Mat& srcImg, float tlow, float thigh)
{
	//Mat padSrc(srcImg.rows + 1, srcImg.cols + 1, CV_32F, Scalar(0.0));
	//srcImg.copyTo(padSrc(Rect(1, 1, srcImg.rows, srcImg.cols)));
	
	double maxVal, minVal;
	minMaxLoc(srcImg, &minVal, &maxVal);
	if (maxVal != minVal)
		srcImg.forEach<float>([minVal, maxVal](float &p, const int * position)->void {p = (p - minVal) / (maxVal - minVal); });//forEach is parralel, can increase performance
	srcImg(Range(0,1), Range(0, srcImg.cols)) = 0.0;//图像周边置0
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

void InvertImg(Mat& srcImg)
{
	srcImg.forEach<float>([](float &p, const int * position)->void {p = -p;});
	double maxVal, minVal;
	minMaxLoc(srcImg, &minVal, &maxVal);
	if (maxVal != minVal)
		srcImg.forEach<float>([minVal, maxVal](float &p, const int * position)->void {p = (p - minVal) / (maxVal - minVal); });
}

void Evaluate(float& p, float& efp, float& efn, const Mat& rstImg, const Mat& gtImg, int neigbSize)
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
			if (abs(padRst.at<float>(r, c)) < 0.0001)
			{
				bool hasNegbor = false;
				for (int x = -padsz; x <= padsz; x++)
				{
					if (hasNegbor)
						break;
					for (int y = -padsz; y <= padsz; y++)
					{
						if (abs(padGt.at<float>(r + x, c + y)) < 0.0001)
						{
							E += 1;
							hasNegbor = true;
							padGt.at<float>(r + x, c + y) = 1;
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
	padGt.forEach<float>([&Efn](float &p, const int * position)->void {if (abs(p) < 0.0001) { Efn += 1; }; });
	p = E * 1.0 / (E + Efp + Efn);
	efp = Efp * 1.0 / E;
	int Egt = 0;
	gtImg.forEach<float>([&Egt](float &p, const int * position)->void {if (abs(p) < 0.0001) { Egt += 1; }; });
	efn = Efn * 1.0 / Egt;
}

float NonCRF(Mat srcImg, Mat gtImg)
{
	if (srcImg.type() != CV_32F)
		srcImg.convertTo(srcImg, CV_32F, 1 / 255.0);
	if (gtImg.type() != CV_32F)
		gtImg.convertTo(gtImg, CV_32F, 1 / 255.0);
	bool halfwave = 1;
	float lamda = 10;//wavelength
	//vector<float> lamdaVec = { 3,4,5,6,7,8,9,10,11,12,13,15,15,16,17,18,19,21 };
	vector<float> lamdaVec = { 3,5,7,9,11,13,15,17,19,21,23,25,27 };

	vector<float> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);
	float sigma = 1.0;
	float gamma = 0.5;//aspect ratio
	float bandwidth = 1;
	int ntheta = 16;
	int nphi = 2;
	float arrphi[2] = { 0, 0.5*CV_PI};
	vector<float> phi = { arrphi, arrphi + 2 };
	vector<float> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	int supPhases = 2;
	int inhibMethod = 2;//2:isotropic, 3:antisotropic, 1:no inhibition
	int inhibSup = 3;
	float alpha = 1.0;
	float k1 = 1;
	float k2 = 4;
	float tlow = 0.05;
	float thigh = 0.3;
	float p, efp, efn;
	vector<vector<Mat>> gaborImgs;
	//GaborFilter(gaborImgs, srcImg, halfwave, lamda, sigma, theta, phi, gamma, bandwidth);
	GaborFilterMulScale(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);
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
	
	
	/*viewImg = imread("matlabview.pgm");
	cvtColor(viewImg, viewImg, COLOR_BGR2GRAY);
	viewImg.convertTo(viewImg, CV_32F);
	orienImg = imread("matlabori.pgm");
	cvtColor(orienImg, orienImg, COLOR_BGR2GRAY);
	viewImg.convertTo(orienImg, CV_32F);
	Mat showImg2 = viewImg.clone();
	normalize(showImg2, showImg2, 0, 1, NORM_MINMAX);
	imshow("viewImgmatlab", showImg2);
	waitKey(0);*/

	Mat thinImg;
	Thinning(thinImg, viewImg, orienImg);
	Mat showImg1 = thinImg.clone();
	normalize(showImg1, showImg1, 0, 1, NORM_MINMAX);
	imshow("thinImg", showImg1);
	waitKey(0);

	Hysteresis(thinImg, tlow, thigh);
	
	InvertImg(thinImg);

	Evaluate(p, efp, efn, thinImg, gtImg, 5);
	cout << "p: " << p << ", efp: " << efp << ", efn: " << efn << endl;
	namedWindow("NCRF image", 0);
	imshow("NCRF image", thinImg);
	imshow("ground truth", gtImg);
	waitKey(0);
	return p;

}

float NonCRF(Mat srcImg, Mat gtImg, vector<Mat> kernel, bool isDisplay)
{
	if (srcImg.type() != CV_32F)
		srcImg.convertTo(srcImg, CV_32F, 1 / 255.0);
	if (gtImg.type() != CV_32F)
		gtImg.convertTo(gtImg, CV_32F, 1 / 255.0);
	bool halfwave = 1;
	float lamda = 10;//wavelength
	float sigma = 1.0;
	float gamma = 0.5;//aspect ratio

	vector<float> lamdaVec = { 3,5,7,9,11,13,15,17,19,21,23,25,27 };
	vector<float> sigmaVec;
	for (auto i : lamdaVec)
		sigmaVec.push_back(i*0.2);

	float bandwidth = 1;
	int ntheta = 8;
	int nphi = 2;
	float arrphi[2] = { 0, 0.5*CV_PI };
	vector<float> phi = { arrphi, arrphi + 2 };
	vector<float> theta;
	for (int i = 0; i < ntheta; i++)
		theta.push_back(2 * CV_PI * i / ntheta);
	int supPhases = 1;
	int inhibMethod = 2;//isotropic or antisotropic
	int inhibSup = 3;
	float alpha = 1.0;
	float k1 = 1;
	float k2 = 4;
	float tlow = 0.05;
	float thigh = 0.25;
	float p, efp, efn;
	vector<vector<Mat>> gaborImgs;
	//GaborFilter(gaborImgs, srcImg, halfwave, lamda, sigma, theta, phi, gamma, bandwidth);
	GaborFilterMulScale(gaborImgs, srcImg, halfwave, lamdaVec, sigmaVec, theta, phi, gamma, bandwidth);
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

//function used inside
void GaborKernel2d(Mat& kernel, float sigma, float lamda, float theta, float phi, float gamma, float bandwidth)
{
	float slratio = (1.0 / CV_PI) * sqrtf((log(2.0) / 2.0)) * ((pow(2, bandwidth) + 1) * 1.0 / (pow(2, bandwidth) - 1));
	if (sigma == 0)
		sigma = slratio * lamda;
	else if (lamda == 0)
		lamda = sigma / slratio;
	int n;
	if (gamma <= 1 && gamma > 0)
		n = int(ceilf(2.5 * sigma / gamma));
	else
		n = int(ceilf(2.5 * sigma));	
	float gamma2 = gamma * gamma;
	float b = 1.0 / (2 * sigma * sigma);
	float a = b / CV_PI;
	float f = 2.0 * CV_PI / lamda;	
	float posSum = 0;
	float negSum = 0;
	for (int r = 0; r < 2*n + 1; r++)
	{
		float* ptr = kernel.ptr<float>(r);
		for (int c = 0; c < 2*n + 1; c++)
		{
			float xp = -(c - n) * cos(theta) + (r - n) * sin(theta);//由于filter2D实际上做的是相关而非卷积，因此要把和旋转180°，x,y都要反号。同时，图像坐标中Y轴朝下，y再次反号
			float yp = (c - n) * sin(theta) + (r - n) * cos(theta);
			ptr[c] = a * exp(-b * (xp * xp + gamma2 * (yp * yp))) * cos(f * xp + phi);
			if (ptr[c] > 0)
				posSum += ptr[c];
			else
				negSum += ptr[c];
		}
	}
	negSum = abs(negSum);
	float meanSum = (posSum + negSum) / 2.0;
	if (meanSum > 0)//归一化系数
	{
		posSum /= meanSum;
		negSum /= meanSum;
	}
	
	for (int r = 0; r < 2 * n + 1; r++)//正负平衡
	{
		float* ptr = kernel.ptr<float>(r);
		for (int c = 0; c < 2 * n + 1; c++)
		{	
			if (ptr[c] > 0)
				ptr[c] = negSum * ptr[c];
			else
				ptr[c] = posSum * ptr[c];
		}
	}
}

Mat DogKernel2d(float sigma, int k1, int k2)
{
	int n = ceil(sigma) * (3 * k2 + k1) - 1;
	int kerSz = 2 * n + 1;
	Mat kernel(kerSz, kerSz, CV_32F);
	float norm_L1 = 0.0;
	for (int r = 0; r < kerSz; r++)
	{
		for (int c = 0; c < kerSz; c++)
		{
			int x = c - n;
			int y = r - n;
			float sigma1 = k2 * sigma;
			float sigma2 = k1  * sigma;
			kernel.at<float>(r, c) = exp((-(x*x + y*y)) / (2.0 * sigma1*sigma1)) / (2.0 * CV_PI*sigma1*sigma1) 
				- exp((-(x*x + y*y)) / (2.0 * sigma2*sigma2)) / (2.0 * CV_PI*sigma2*sigma2);
			if (kernel.at<float>(r, c) < 0)
				kernel.at<float>(r, c) = 0;
			norm_L1 += kernel.at<float>(r, c);
		}
	}
	kernel.forEach<float>([norm_L1](float& p, const int*)->void {
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