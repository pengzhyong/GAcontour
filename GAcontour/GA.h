#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void Ga_init(int maskSize, vector<vector<float>>& population, int popSize);//����ģ�壬��ģ�尴��չ�����洢��vector<float>
void Ga_fitness(const vector<Mat>& trainData, const vector<Mat>& groundTruth, 
	const vector<vector<int>>& population, vector<double>& fitValues);
void Ga_select(const vector<Mat>& trainData, const vector<Mat>& groundTruth, 
	vector<vector<float>>& population, bool elitism);

//����
void Ga_cross(vector<vector<float>>& population, double cross_rate);

//����
void Ga_mutation(vector<vector<float>>& popultion, double mutation_rate);

void GA();

void loadImg(vector<Mat>& srcVec);

void loadMat(vector<Mat>& groundTruth);