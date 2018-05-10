#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Ga_init(int maskSize, int ntheta, vector<vector<float>>& population, int popSize);//单个模板，把模板按行展开，存储成vector<float>
void Ga_fitness(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat& groundTruth2,
	const vector<vector<float>>& population, vector<float>& fitValues, int kernelSize, int ntheta);
void Ga_select(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat& groundTruth2,
	vector<vector<float>>& population, bool elitism, int kernelSize, int ntheta);

//交叉
void Ga_cross(vector<vector<float>>& population, double cross_rate, int crossMethod = 1);

//变异
void Ga_mutation(vector<vector<float>>& popultion, double mutation_rate);

void GA(Mat& trainData1, Mat& trainData2, Mat& groundTruth1, Mat& groundTruth2);

void loadImg(vector<Mat>& srcVec);

void loadMat(vector<Mat>& groundTruth);

void savePopulation(string fileName, const vector<vector<float>>& population);
void saveFitvalue(string fileNameAve, const vector<float>& avefit, const vector<float> maxfit);
void loadPopulation(string fileName, vector<vector<float>>& population);
void showResult(const vector<float>& population, int ntheta, const Mat& srcImg, const Mat& gtImg);
float maxFit(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat groundTruth2,
	vector<vector<float>>& population, int kernelSize, int ntheta, int& maxIndex);