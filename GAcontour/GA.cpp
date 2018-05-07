#include <numeric>
#include "GA.h"
#include "postProcess.h"
#include "NCRF.h"
vector<float> trainSeq;
//#include <time.h>

void Ga_init(int maskSize, int ntheta, vector<vector<float>>& population, int popSize)
{
	population.clear();
	maskSize = maskSize % 2 == 0 ? maskSize + 1 : maskSize;//保证模板尺寸为奇数
	Mat individualMat(1, ntheta*maskSize*maskSize, CV_32F);
	float upperValue = 100;//模板最大最小值
	float lowerValue = -100;
	for (int i = 0; i < popSize; i++)
	{
		vector<float> individual;
		individualMat.forEach<float>([](float& p, const int* pos)->void { p = (rand() % 400) * 0.01 - 2.0; });//random value in [-2,2];
		for (int j = 0; j < individualMat.cols; j++)
			individual.push_back(individualMat.at<float>(0,j));
		population.push_back(individual);
	}
	
}

//计算每个个体的适应度
void Ga_fitness(const Mat& trainData, const Mat& groundTruth, 
	const vector<vector<float>>& population, vector<float>& fitValues, int kernelSize, int ntheta)
{
	time_t tic, toc;
	time(&tic);
	float avePerform = 0;
	vector<vector<Mat>> kernel;
	for (int i = 0; i < population.size(); i++)
	{
		vector<Mat > indivKer;
		for (int j = 0; j < ntheta; j++)
		{
			Mat singleKernel(kernelSize, kernelSize, CV_32F);
			int stIndex = j * kernelSize * kernelSize;
			// 1D chromosome to 2D kernel
			singleKernel.forEach<float>([population, i, stIndex, kernelSize](float& p, const int* pos)
				->void { p = population[i][stIndex + pos[0] * kernelSize + pos[1]]; });
			indivKer.push_back(singleKernel);
		}	
		kernel.push_back(indivKer);
		/*float p = NonCRF(trainData, groundTruth, kernel);
		fitValues.push_back(p);
		avePerform += p;*/
	}

	GaParWrapper gapar(trainData, groundTruth, kernel, population.size());
	parallel_for_(Range(0, population.size()), gapar);
	fitValues = vector<float>(gapar.fitness, gapar.fitness + population.size());
	avePerform = std::accumulate(fitValues.begin(), fitValues.end(), 0.0);
	avePerform /= population.size();
	cout << ", average perform: " << avePerform << " ";
}

//选择，产生新一代种群
void Ga_select(const Mat& trainData, const Mat groundTruth, 
	vector<vector<float>>& population, bool elitism, int kernelSize, int ntheta)
{
	vector<vector<float>> new_population;
	vector<float> fitValue;
	//time_t tic, toc;
	//time(&tic);
	Ga_fitness(trainData, groundTruth, population, fitValue, kernelSize, ntheta);
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
	population = new_population;
}

//交叉
void Ga_cross(vector<vector<float>>& population, double cross_rate)
{
	int len = population.at(0).size();
	for (int person = 0; person < population.size(); person = person + 2)
	{
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
}

//变异
void Ga_mutation(vector<vector<float>>& popultion, double mutation_rate)
{
	int len = popultion.at(0).size();
	for (int person = 0; person < popultion.size(); person++)
	{
		double randNums = (rand() % 10 + 0.5) * 0.1;
		if (randNums > mutation_rate)
			continue;
		int muPos = rand() % 100 * 0.01 * popultion.at(person).size();
		float randCoef = rand() % 40 * 0.1 - 2;//[-2,2]
		popultion.at(person).at(muPos) *= randCoef;
	}
}

void GA(Mat& trainData, Mat& groundTruth)
{
	double cross_rate = 0.8;//交叉率
	double mutation_rate = 0.01;//变异率
	bool elitism = false;//是否保留精英
	int population_size = 100;//种群大小
	int generations = 0;//进化代数
	vector<vector<float>> population;

	float sigma = 1.0;// must be same with the values in NonCRF
	int k1 = 1, k2 = 4;
	int maskSize = ceil(sigma) * (3 * k2 + k1) - 1;;//模板尺寸
	int ntheta = 8;
	Ga_init(maskSize, ntheta, population, population_size);
	time_t startTime, curTime, prevTime;
	time(&startTime);//计时操作，以免超时
	while (++generations)
	{
		time(&prevTime);

		cout << "generations: " << generations << " ";
		Ga_select(trainData, groundTruth,population, elitism, maskSize, ntheta);
		Ga_cross(population, cross_rate);
		Ga_mutation(population, mutation_rate);

		time(&curTime);
		
		int totalTime = curTime - startTime;
		int diffTime = curTime - prevTime;
		
		cout << "time useed: " << diffTime << ", total times: " << totalTime << endl;
	}

	vector<float> fitValue;
	Ga_fitness(trainData, groundTruth, population, fitValue, maskSize, ntheta);
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
