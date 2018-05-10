#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

float NonCRF(Mat srcImg, Mat gtImg);
float NonCRF(Mat srcImg, Mat gtImg, vector<Mat> kernel, bool isDisplay = false);//overload function for genetic algorithm, return the p value, as a fitness value

class GaParWrapper :public ParallelLoopBody
{
	//friend void NonCRF(Mat, Mat);
public:
	GaParWrapper() = default;
	GaParWrapper(const Mat& src, const Mat& gt, vector<vector<Mat>>& ker, int popsize) :srcImg(src), gtImg(gt), kernel(ker) { fitness = new float[popsize]; };
	virtual ~GaParWrapper() { delete[] fitness; };
	inline virtual void operator()(const Range& range) const;
	//virtual void operator()(const Range& range) const { cout << "parfor const" << endl; };
	float* fitness;
private:
	Mat srcImg, gtImg;
	vector<vector<Mat>> kernel;
	int popsize;
};
inline void GaParWrapper::operator()(const Range& range) const
{ 
	for (int i = range.start; i < range.end; i++)
	{
		fitness[i] = NonCRF(srcImg, gtImg, kernel[i]);
	}
}


int GaborFilter(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	float lamda, float sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth);
int GaborFilterMulScale(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	vector<float> lamda, vector<float> sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth);
void PhaseSuppos(vector<Mat>& dstImg, const vector<vector<Mat>>& srcImg, int ntheta, int nphi, int supMethod);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float sigma, float alpha, int k1, int k2);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float alpha, const vector<Mat>& kernel);

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