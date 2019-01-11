#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
typedef pair<vector<Mat>, vector<double>> individual;

void Ga_init(int maskSize, int ntheta, vector<vector<double>>& population, int popSize);//单个模板，把模板按行展开，存储成vector<double>
void Ga_fitness(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat& groundTruth2,
	vector<individual>& population, vector<double>& fitValues, int kernelSize, int ntheta);
void Ga_select(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat& groundTruth2,
	vector<individual>& population, bool elitism, int kernelSize, int ntheta);

//交叉
void Ga_cross(vector<vector<double>>& population, double cross_rate, int crossMethod = 1);

void Ga_cross2d(vector<individual>& population, double cross_rate, int kernelSize, int ntheta);
//变异
void Ga_mutation(vector<individual>& popultion, double mutation_rate);

void GA(Mat& trainData1, Mat& trainData2, Mat& groundTruth1, Mat& groundTruth2);

void loadImg(vector<Mat>& srcVec);

void loadMat(vector<Mat>& groundTruth);

void savePopulation(string fileName, const vector<vector<double>>& population);
void saveFitvalue(string fileNameAve, const vector<double>& avefit, const vector<double> maxfit);
void loadPopulation(string fileName, vector<vector<double>>& population);
void showResult(const vector<double>& population, int ntheta, const Mat& srcImg, const Mat& gtImg);
double maxFit(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat groundTruth2,
	vector<individual>& population, int kernelSize, int ntheta, int& maxIndex);
void saveMask(const Mat& kernel, string fileName);
void loadMask(Mat& kernel, int kersize, string fileName);