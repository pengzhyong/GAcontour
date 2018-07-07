#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

float NonCRF(Mat srcImg, Mat gtImg);
float NonCRF(Mat srcImg, Mat gtImg, vector<Mat> kernel, bool isDisplay = false);//overload function for genetic algorithm, return the p value, as a fitness value
float NonCRF(Mat srcImg, Mat gtImg, const vector<Mat>& kernel, const vector<float> coefs, bool isDisplay = false);
typedef pair<vector<Mat>, vector<float>> individual;

class GaParWrapper :public ParallelLoopBody
{
	//friend void NonCRF(Mat, Mat);
public:
	GaParWrapper() = default;
	GaParWrapper(const Mat& src, const Mat& gt, const vector<individual>& pop, int popsize) :srcImg(src), gtImg(gt), population(pop) { fitness = new float[popsize]; };
	virtual ~GaParWrapper() { delete[] fitness; };
	inline virtual void operator()(const Range& range) const;
	//virtual void operator()(const Range& range) const { cout << "parfor const" << endl; };
	float* fitness;
private:
	Mat srcImg, gtImg;
	vector<individual> population;
	int popsize;
};
inline void GaParWrapper::operator()(const Range& range) const
{ 
	for (int i = range.start; i < range.end; i++)
		fitness[i] = NonCRF(srcImg, gtImg, population.at(i).first, population.at(i).second, 1);
}


int GaborFilter(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	float lamda, float sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth);
int GaborFilterMulScale(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	vector<float> lamda, vector<float> sigma, vector<float> theta, vector<float> phi, float gamma, float bandwidth);
void PhaseSuppos(vector<Mat>& dstImg, const vector<vector<Mat>>& srcImg, int ntheta, int nphi, int supMethod);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float sigma, float alpha, int k1, int k2);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float alpha, const vector<Mat>& kernel);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, float sigma, const vector<Mat>& kernel, const vector<float>& coefs);

void ViewImage(Mat& dstImg, Mat& orienImg, const vector<Mat>& srcImg, const vector<float> theta);

void Thinning(Mat& dstImg, const Mat& srcImg, const Mat& orienMat);

void Hysteresis(Mat& srcImg, float tlow, float thigh);

void InvertImg(Mat& srcImg);

void Evaluate(float& p, float& efp, float& efn, const Mat& rstImg, const Mat& gtImg, int neigbSize);



//function used inside
Mat GaborKernel2d(float sigma, float lamda, float theta, float phi, float gamma, float bandwidth);

Mat GaborKernel2d(float sigma, float lamda, float theta, float phi, float gamma, float bandwidth, float radiu);

Mat BendGabor2d(float sigma, float lamda, float theta, float phi, float gamma, float bandwidth, float radiu);

Mat DogKernel2d(float sigma, int k1, int k2);

Mat createFftImage(const Mat& srcImg, int fh, int fw);

Mat convolution(const Mat& srcImg, const Mat& filterKernel, const Mat& fftImg);

//-------fastener detection
Mat NonCrf_fastener(Mat srcImg, int inhibModel, float tl, float th);
Mat drawButterflyNCRF();
Mat drawRNCRF();