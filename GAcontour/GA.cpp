#include <numeric>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include "GA.h"
#include "postProcess.h"
#include "NCRF.h"

vector<float> avefit, maxfit;
int maxFitIndex = 0;

void Ga_init(int maskSize, int ntheta, vector<individual>& population, int popSize)
{
	population.clear();
	maskSize = maskSize % 2 == 0 ? maskSize + 1 : maskSize;//保证模板尺寸为奇数
	Mat maskMat(1, ntheta*maskSize*maskSize, CV_32F);
	/*Mat gaborKer(maskSize, maskSize, CV_32F);
	GaborKernel2d(gaborKer, 1.0, 10, 0, 0, 0.5, 1);*/
	for (int i = 0; i < popSize; i++)
	{
		individual person;
		Mat maskMat1(maskSize, maskSize, CV_32F);
		Mat maskMat2(maskSize, maskSize, CV_32F);
		maskMat1.forEach<float>([](float& p, const int* pos)->void { p = (rand() % 400) * 0.01 - 2.0; });//random value in [-2,2];
		maskMat2.forEach<float>([](float& p, const int* pos)->void { p = (rand() % 400) * 0.01 - 2.0; });
		vector<float> coefs;
		for (int i = 0; i < 2 * ntheta; i++)
			person.second.push_back((rand() % 400) * 0.01 - 2.0);
		person.first.push_back(maskMat1);
		person.first.push_back(maskMat2);
		population.push_back(person);
	}
	
}

//计算每个个体的适应度
void Ga_fitness(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat& groundTruth2,
	vector<individual>& population, vector<float>& fitValues, int kernelSize, int ntheta)
{
	time_t tic, toc;
	time(&tic);
	float avePerform = 0;

	GaParWrapper gapar1(trainData1, groundTruth1, population, population.size());
	parallel_for_(Range(0, population.size()), gapar1);
	fitValues = vector<float>(gapar1.fitness, gapar1.fitness + population.size());
	/*GaParWrapper gapar2(trainData2, groundTruth2, kernel, population.size());
	parallel_for_(Range(0, population.size()), gapar2);
	vector<float> fitValues2 = vector<float>(gapar2.fitness, gapar2.fitness + population.size());
	for (int i = 0; i < population.size(); i++)
		fitValues.push_back(0.5 * (fitValues1.at(i) + fitValues2.at(i)) + min(fitValues1.at(i), fitValues2.at(i)));*/

	avePerform = std::accumulate(fitValues.begin(), fitValues.end(), 0.0);
	avePerform /= population.size();

	maxFitIndex = max_element(fitValues.begin(), fitValues.end()) - fitValues.begin();
	float maxfitVal = fitValues.at(max_element(fitValues.begin(), fitValues.end()) - fitValues.begin());
	cout << ", average perform: " << avePerform << " maxfit: " << maxfitVal << " ";
	avefit.push_back(avePerform);// save to global variance for save result
	maxfit.push_back(maxfitVal);

}

//选择，产生新一代种群
void Ga_select(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat& groundTruth2, 
	vector<individual>& population, bool elitism, int kernelSize, int ntheta)
{
	vector<individual> new_population;
	vector<float> fitValue;
	//time_t tic, toc;
	//time(&tic);
	Ga_fitness(trainData1, trainData2, groundTruth1, groundTruth2, population, fitValue, kernelSize, ntheta);
	//time(&toc);
	//cout << "Ga_fitness time: " << toc - tic << endl;
	int firstPerson = 0;
	if (elitism)//精英保留
	{
		auto maxIndex = max_element(fitValue.begin(), fitValue.end()) - fitValue.begin();
		new_population.push_back(population.at(maxIndex));
		firstPerson = 1;
	}

	vector<double> sum_fit;
	for (auto i : fitValue)
		sum_fit.push_back(i);
	for (int i = 1; i < sum_fit.size(); i++)//适应度累加
		sum_fit.at(i) += sum_fit.at(i - 1);

	for (int i = firstPerson; i < population.size(); i++)
	{
		int posValue = (rand() % 1000) * 0.001 * sum_fit.back();
		int first = 0;
		int last = sum_fit.size() - 1;
		int mid = 0.5 * (first + last) + 0.5;//加0.5, 四舍五入
		int index = 0;
		while (1)
		{
			if (posValue == sum_fit.at(mid))
			{
				index = mid;
				break;
			}
			if (posValue > sum_fit.at(mid))
				first = mid;
			else
				last = mid;
			if (last - first == 1)
			{
				index = first;//或者 = last, 会有一点偏差
				break;
			}
			mid = 0.5 * (first + last);
		}
		new_population.push_back(population.at(index));
	}
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	shuffle(new_population.begin(), new_population.end(), std::default_random_engine(seed));

	population = new_population;
}

//交叉
void Ga_cross(vector<vector<float>>& population, double cross_rate, int crossMethod)
{
	int len = population.at(0).size();
	for (int person = 0; person < population.size(); person = person + 2)
	{
		if (crossMethod == 1)
		{
			//单点交叉
			double randNums = (rand() % 10 + 0.5) * 0.1;
			if (randNums > cross_rate)//交叉率
				continue;
			int pos = rand() % 10 * 0.1 * len;
			vector<float> seg1(population.at(person).begin() + pos, population.at(person).end());//选出的基因片段
			vector<float> seg2(population.at(person + 1).begin() + pos, population.at(person + 1).end());

			population.at(person) = vector<float>(population.at(person).begin(), population.at(person).begin() + pos);
			population.at(person).insert(population.at(person).end(), seg2.begin(), seg2.end());
			population.at(person + 1) = vector<float>(population.at(person + 1).begin(), population.at(person + 1).begin() + pos);
			population.at(person + 1).insert(population.at(person + 1).end(), seg1.begin(), seg1.end());
		}
		
		if (crossMethod == 2)
		{
			//均匀交叉
			for (int i = 0; i < len; i++)
			{
				float r = rand() % 1000 * 0.001;
				if (r < 0.001)
				{
					population.at(person).at(i) = population.at(person).at(i) * 0.5 + population.at(person + 1).at(i) * 0.5;
					population.at(person + 1).at(i) = population.at(person + 1).at(i) * 0.5 + population.at(person).at(i) * 0.5;
				}
			}
		}
	}
}

void Ga_cross2d(vector<individual>& population, double cross_rate, int kernelSize, int ntheta)
{
	int nmask = population.at(0).first.size();
	int maskSize = population.at(0).first.at(0).rows;
	for (int person = 0; person < population.size(); person = person + 2)
	{
		float randNums1 = rand() % 100 * 0.01;
		if (randNums1 < cross_rate)
		{
			int kerIndex = rand() % 100 * 0.01 * nmask;
			int randRow = rand() % 100 * 0.01 * maskSize;
			int randCol = rand() % 100 * 0.01 * maskSize;
			Mat tmp(maskSize, maskSize, CV_32F, Scalar(0.0));
			Mat indivKer1 = population.at(person).first.at(kerIndex);
			Mat indivKer2 = population.at(person + 1).first.at(kerIndex);

			tmp.forEach<float>([indivKer1, kerIndex, randRow, randCol](float& p, const int* pos)
				->void {if (pos[0] >= randRow && pos[1] >= randCol) p = indivKer1.at<float>(pos[0], pos[1]); });
			indivKer1.forEach<float>([indivKer2, kerIndex, randRow, randCol](float& p, const int* pos)
				->void {if (pos[0] >= randRow && pos[1] >= randCol) p = indivKer2.at<float>(pos[0], pos[1]); });
			indivKer2.forEach<float>([tmp, kerIndex, randRow, randCol](float& p, const int* pos)
				->void {if (pos[0] >= randRow && pos[1] >= randCol) p = tmp.at<float>(pos[0], pos[1]); });
		}
		//系数染色体 均匀交叉
		float c = 0.5;
		for (int i = 0; i < nmask * ntheta; i++)
		{
			float r = rand() % 1000 * 0.001;
			if (r < 0.01)
			{
				population.at(person).second.at(i) = population.at(person).second.at(i) * c + population.at(person + 1).second.at(i) * (1 - c);
				population.at(person + 1).second.at(i) = population.at(person + 1).second.at(i) * c + population.at(person).second.at(i) * (1 - c);
			}
		}
	}
}

//变异
void Ga_mutation(vector<individual>& popultion, double mutation_rate)
{
	int nmask = popultion.at(0).first.size();
	int maskSize = popultion.at(0).first.at(0).rows;
	for (int person = 0; person < popultion.size(); person++)
	{
		//-----mask mutation
		for (int i = 0; i < nmask; i++)
		{
			float randNums = rand() % 100 * 0.01;
			if (randNums < mutation_rate)
			{
				float randCoef = rand() % 400 * 0.01 - 2;//[-2,2]
				float randRows = rand() % 100 * 0.01 * maskSize;
				float randCols = rand() % 100 * 0.01 * maskSize;
				popultion.at(person).first.at(i).at<float>(randRows, randCols) *= randCoef;
			}
		}
		//------coefs muation
		float randNums = rand() % 100 * 0.01;
		if (randNums < mutation_rate)
		{
			float randCoef = rand() % 400 * 0.01 - 2;//[-2,2]
			float randPos = rand() % 100 * 0.01 * popultion.at(0).second.size();
			popultion.at(person).second.at(randPos) *= randCoef;
		}
	}
}

void GA(Mat& trainData1, Mat& trainData2, Mat& groundTruth1, Mat& groundTruth2)
{
	double cross_rate = 0.9;//交叉率
	double mutation_rate = 0.1;//变异率
	bool elitism = true;//是否保留精英
	int population_size = 20;//种群大小
	int generations = 0;//进化代数

	vector<individual> population;//2 mask + 2*ntheta coef, pair.fist denote 2 mask, pair.second denote coefs

	float sigma = 1.0;// must be same with the values in NonCRF
	int k1 = 1, k2 = 4;
	int maskSize = ceil(sigma) * (3 * k2 + k1) - 1;;//模板尺寸
	int ntheta = 8;
	Ga_init(maskSize, ntheta, population, population_size);
	//loadPopulation("pop_shuffle_4100.txt", population);
	//int maxInd = 3;
	//float maxfitVal = maxFit(trainData1, trainData2, groundTruth1, groundTruth2, population, maskSize, ntheta, maxInd);
	//cout << endl << "maxfit value: " << maxfitVal << ", maxIndex:" << maxInd << endl;
	//showResult(population[maxInd], ntheta, trainData1, groundTruth1);
	//showResult(population[maxInd], ntheta, trainData2, groundTruth2);

	/*for (int i = 0; i < population.size(); i++)
	{
		showResult(population[i], ntheta, trainData, groundTruth);
	}*/
	time_t startTime, curTime, prevTime;
	time(&startTime);//计时操作，以免超时
	while (++generations)
	{
		time(&prevTime);

		cout << "generations: " << generations << " ";
		Ga_select(trainData1, trainData2, groundTruth1, groundTruth2, population, elitism, maskSize, ntheta);
		//showResult(population[maxFitIndex], ntheta, trainData1, groundTruth1);

		//Ga_cross(population, cross_rate, 2);
		Ga_cross2d(population, cross_rate, maskSize, ntheta);
		Ga_mutation(population, mutation_rate);

		time(&curTime);
		
		int totalTime = curTime - startTime;
		int diffTime = curTime - prevTime;
		
		cout << "time: " << diffTime << ", total times: " << totalTime << endl;
		if (generations % 100 == 0)
		{
			//savePopulation("pop_shuffle_" + to_string(generations) + ".txt", population);
			//saveFitvalue("fitval.txt", avefit, maxfit);
		}
	}

	vector<float> fitValue;
	Ga_fitness(trainData1, trainData2, groundTruth1, groundTruth2, population, fitValue, maskSize, ntheta);
	int maxIndex = max_element(fitValue.begin(), fitValue.end()) - fitValue.begin();
	double maxFitVale = fitValue.at(maxIndex);
}

void loadImg(vector<Mat>& srcVec)
{
	string basePath = "C:\\Users\\pengzhyong\\Desktop\\CV\\BSDS500\\images\\train\\myData";
	int imageNums = 1;
	for (int i = 0; i < imageNums; i++)
	{
		string imagePath = basePath + "\\" +  to_string(i + 1) + ".jpg";
		Mat tmpMat = imread(imagePath);
		cvtColor(tmpMat, tmpMat, COLOR_BGR2GRAY);
		tmpMat.convertTo(tmpMat, CV_32F, 1 / 255.0);
		srcVec.push_back(tmpMat);
	}
}

void loadMat(vector<Mat>& groundTruth)
{
	string basePath = "C:\\Users\\pengzhyong\\Desktop\\CV\\BSDS500\\groundTruth\\myTrain";
	int imageNums = 1;
	for (int i = 0; i < imageNums; i++)
	{
		string imagePath = basePath + "\\" + to_string(i + 1) + ".yml";
		
		Mat tmpMat;
		FileStorage matFile(imagePath, FileStorage::READ);
		matFile["first"] >> tmpMat;
		//cvtColor(tmpMat, tmpMat, COLOR_BGR2GRAY);
		tmpMat.convertTo(tmpMat, CV_32F);
		groundTruth.push_back(tmpMat);
	}
}

void savePopulation(string fileName, const vector<vector<float>>& population)
{
	ofstream outFile(fileName);
	for (auto person : population)
	{
		for (auto i : person)
		{
			outFile << i << " ";
		}
		outFile << endl;
	}
	outFile.close();
}

void saveFitvalue(string fileNameAve, const vector<float>& avefit, const vector<float> maxfit)
{
	ofstream outFile(fileNameAve);
	for (int i = 0; i < avefit.size(); i++)
		outFile << avefit.at(i) << " ";
	outFile << endl;
	for (int i = 0; i < maxfit.size(); i++)
		outFile << maxfit.at(i) << " ";
	outFile.close();
}

void loadPopulation(string fileName, vector<vector<float>>& population)
{
	ifstream inFile(fileName);
	string str;
	char ch[10];
	int i = 0;
	while (getline(inFile, str))
	{
		istringstream istr(str);
		vector<float> individual;
		while (istr >> ch)
		{
			individual.push_back(atof(ch));
		}
		population.push_back(individual);
		i++;
	}
}

void showResult(const vector<float>& population, int ntheta, const Mat& srcImg, const Mat& gtImg)
{
	int kernelSize = sqrt(population.size() / ntheta) - 1;
	vector<Mat> indivKer;
	for (int j = 0; j < ntheta; j++)
	{
		Mat singleKernel(kernelSize, kernelSize, CV_32F);
		int stIndex = j * kernelSize * kernelSize;
		// 1D chromosome to 2D kernel
		singleKernel.forEach<float>([population, stIndex, kernelSize](float& p, const int* pos)
			->void { p = population.at(stIndex + pos[0] * kernelSize + pos[1]); });
		indivKer.push_back(singleKernel);

		/*namedWindow("best kernel", 0);
		imshow("best kernel", singleKernel);
		waitKey(0);*/
	}
	float p = NonCRF(srcImg, gtImg, indivKer, 1);	
}
float maxFit(const Mat& trainData1, const Mat& trainData2, const Mat groundTruth1, const Mat groundTruth2,
	vector<individual>& population, int kernelSize, int ntheta, int& maxIndex)
{
	vector<float> fitness;
	Ga_fitness(trainData1, trainData2, groundTruth1, groundTruth2, population, fitness, kernelSize, ntheta);
	maxIndex = max_element(fitness.begin(), fitness.end()) - fitness.begin();
	return fitness.at(maxIndex);
}