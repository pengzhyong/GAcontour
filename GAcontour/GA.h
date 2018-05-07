#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Ga_init(int maskSize, int ntheta, vector<vector<float>>& population, int popSize);//����ģ�壬��ģ�尴��չ�����洢��vector<float>
void Ga_fitness(const Mat& trainData, const Mat& groundTruth,
	const vector<vector<float>>& population, vector<float>& fitValues, int kernelSize, int ntheta);
void Ga_select(const Mat& trainData, const Mat groundTruth,
	vector<vector<float>>& population, bool elitism, int kernelSize, int ntheta);

//����
void Ga_cross(vector<vector<float>>& population, double cross_rate);

//����
void Ga_mutation(vector<vector<float>>& popultion, double mutation_rate);

void GA(Mat& trainData, Mat& groundTruth);

void loadImg(vector<Mat>& srcVec);

void loadMat(vector<Mat>& groundTruth);