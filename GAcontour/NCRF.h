#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

void NonCRF(Mat srcImg, Mat gtImg);

class GaParWrapper :public ParallelLoopBody
{
	//friend void NonCRF(Mat, Mat);
public:
	GaParWrapper() = default;
	GaParWrapper(Mat& src, Mat& gt) :srcImg(src), gtImg(gt) {};
	virtual ~GaParWrapper() {};
	virtual void operator()(const Range& range) const { NonCRF(srcImg, gtImg); };
	//virtual void operator()(const Range& range) const { cout << "parfor const" << endl; };
private:
	Mat srcImg, gtImg;
};
int GaborFilter(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	float lamda, float sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth);
void PhaseSuppos(vector<Mat>& dstImg, const vector<vector<Mat>>& srcImg, int ntheta, int nphi, int supMethod);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float sigma, float alpha, int k1, int k2);

void ViewImage(Mat& dstImg, Mat& orienImg, const vector<Mat>& srcImg, const vector<float> theta);

void Thinning(Mat& dstImg, const Mat& srcImg, const Mat& orienMat);

void Hysteresis(Mat& srcImg, float tlow, float thigh);

void InvertImg(Mat& srcImg);

void Evaluate(float& p, float& efp, float& efn, const Mat& rstImg, const Mat& gtImg, int neigbSize);



//function used inside
void GaborKernel2d(Mat& kernel, float sigma, float lamda, float theta, float phi, float gamma, float bandwidth);

Mat DogKernel2d(float sigma, int k1, int k2);

Mat createFftImage(const Mat& srcImg, int fh, int fw);

Mat convolution(const Mat& srcImg, const Mat& filterKernel, const Mat& fftImg);