#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

double NonCRF(Mat srcImg, Mat gtImg);
double NonCRF(Mat srcImg, Mat gtImg, vector<Mat> kernel, bool isDisplay = false);//overload function for genetic algorithm, return the p value, as a fitness value
double NonCRF(Mat srcImg, Mat gtImg, const vector<Mat>& kernel, const vector<double> coefs, bool isDisplay = false);
typedef pair<vector<Mat>, vector<double>> individual;

class GaParWrapper :public ParallelLoopBody
{
	//friend void NonCRF(Mat, Mat);
public:
	GaParWrapper() = default;
	GaParWrapper(const Mat& src, const Mat& gt, const vector<individual>& pop, int popsize) :srcImg(src), gtImg(gt), population(pop) { fitness = new double[popsize]; };
	virtual ~GaParWrapper() { delete[] fitness; };
	inline virtual void operator()(const Range& range) const;
	//virtual void operator()(const Range& range) const { cout << "parfor const" << endl; };
	double* fitness;
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
	double lamda, double sigma, vector<double> theta, vector<double> phi, double gamma, double bandwidth);
int GaborFilterMulScale(vector<vector<Mat>>& dstImg, const Mat srcImg, bool halfwave,
	vector<double> lamda, vector<double> sigma, vector<double> theta, vector<double> phi, double gamma, double bandwidth);
void PhaseSuppos(vector<Mat>& dstImg, const vector<vector<Mat>>& srcImg, int ntheta, int nphi, int supMethod);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, double sigma, double alpha, int k1, int k2);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, double alpha, const vector<Mat>& kernel);

void Inhibition(vector<Mat>& dstImg, vector<Mat>& srcImg, int inhibMethod, int supMethod, double sigma, const vector<Mat>& kernel, const vector<double>& coefs);

void ViewImage(Mat& dstImg, Mat& orienImg, const vector<Mat>& srcImg, const vector<double> theta);

void Thinning(Mat& dstImg, const Mat& srcImg, const Mat& orienMat);

void Hysteresis(Mat& srcImg, double tlow, double thigh);

void InvertImg(Mat& srcImg);

void Evaluate(double& p, double& efp, double& efn, const Mat& rstImg, const Mat& gtImg, int neigbSize);



//function used inside
Mat GaborKernel2d(double sigma, double lamda, double theta, double phi, double gamma, double bandwidth);

Mat GaborKernel2d(double sigma, double lamda, double theta, double phi, double gamma, double bandwidth, double radiu);

Mat BendGabor2d(double sigma, double lamda, double theta, double phi, double gamma, double bandwidth, double radiu);

Mat DogKernel2d(double sigma, int k1, int k2);

Mat createFftImage(const Mat& srcImg, int fh, int fw);

Mat convolution(const Mat& srcImg, const Mat& filterKernel, const Mat& fftImg);

//-------fastener detection
Mat NonCrf_fastener(Mat srcImg, int inhibModel, double tl, double th);
Mat drawButterflyNCRF();
Mat drawRNCRF();