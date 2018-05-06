#include "GA.h"
#include "postProcess.h"
vector<float> trainSeq;
//#include <time.h>

void Ga_init(int maskSize, vector<vector<float>>& population, int popSize)
{
	population.clear();
	maskSize = maskSize % 2 == 0 ? maskSize + 1 : maskSize;//��֤ģ��ߴ�Ϊ����
	float upperValue = 100;//ģ�������Сֵ
	float lowerValue = -100;
	for (int i = 0; i < popSize; i++)
	{
		vector<float> individual;
		float sumCoef = 0;
		float sigma1 = rand() % 100 * 0.01 + 0.8;//0.2;
		float sigma2 = rand() % 100 * 0.01 + 1.6;//0.8;
		for (int j = 0; j < maskSize * maskSize; j++)
		{
			int disx = j / maskSize - maskSize / 2;
			int disy = j % maskSize - maskSize / 2;
			
			float g1 = 1 / sqrt(2 * CV_PI * sigma1 * sigma1) * (exp(-(disx*disx + disy * disy) / (2 * sigma1 * sigma1)));
			float g2 = 1 / sqrt(2 * CV_PI * sigma2 * sigma2) * (exp(-(disx*disx + disy * disy) / (2 * sigma2 * sigma2)));

			//���ֵ
			float randValue = rand() % 100 * 0.01 * 20 - 10;//-10~10
			individual.push_back(randValue);
			sumCoef += randValue;
		}
		for (int j = 0; j < maskSize * maskSize; j++)
		{
			individual.at(j) /= sumCoef;
		}
		population.push_back(individual);

		/*Mat msk(maskSize, maskSize, CV_32F);
		for (int r = 0; r < maskSize; r++)
		{
			for (int c = 0; c < maskSize; c++)
			{
				msk.at<float>(r, c) = individual.at(r * maskSize + c);
			}
		}
		normalize(msk, msk, 0, 1, NORM_MINMAX);
		namedWindow("individual", 0);
		imshow("individual", msk);
		waitKey();*/
	}
	
}

//����ÿ���������Ӧ��
void Ga_fitness(const vector<Mat>& trainData, const vector<Mat>& groundTruth, 
	const vector<vector<float>>& population, vector<float>& fitValues)
{
	double avePerform = 0;
	int maskSz = sqrt(population.at(0).size());
	for (int per = 0; per < population.size(); per++)
	{

		Mat maskPer(maskSz, maskSz, CV_32F);//����ģ��
		for (int r = 0; r < maskSz; r++)
		{
			for (int c = 0; c < maskSz; c++)
			{
				maskPer.at<float>(r, c) = population.at(per).at(r * maskSz + c);
			}
		}
		//Pֵ��Ϊ������׼, P = Card(E) / (Card(E) + Card(Efp) + Card(Efn))
		float perform = 0;
		for (int im = 0; im < trainData.size(); im++)
		{
			/*imshow("op", trainData.at(im));
			waitKey();*/
			Mat dstImg;
			filter2D(trainData.at(im), dstImg, trainData.at(im).depth(), maskPer);
			//imshow("mask", maskPer);
			//waitKey();
			/*imshow("op", dstImg);
			waitKey();*/
   			NMS(dstImg, dstImg);//�ͺ���ֵ����
			/*imshow("op", dstImg);
			waitKey();*/

			threshold(dstImg, dstImg, 0.0001, 1, THRESH_BINARY);
			/*imshow("op", dstImg);
			waitKey();*/

			float E, Efp, Efn;
			E = Efp = Efn = 0;
			for (int r = 0; r < trainData.at(im).rows; r++)
			{
				for (int c = 0; c < trainData.at(im).cols; c++)
				{
					if (dstImg.at<float>(r, c) > 0.0001 && trainData.at(im).at<float>(r, c) > 0.0001)
						E++;
					else if (dstImg.at<float>(r, c) > 0.0001 && trainData.at(im).at<float>(r, c) <= 0.0001)
						Efp++;
					else if (dstImg.at<float>(r, c) <= 0.0001 && trainData.at(im).at<float>(r, c) > 0.0001)
						Efn++;
				}
			}
			perform += E / (E + Efp + Efn);
		}
		perform /= trainData.size();
		fitValues.push_back(perform);
		avePerform += perform;
	}
	avePerform /= population.size();
	cout << " average perform: " << avePerform << " ";
	trainSeq.push_back(avePerform);
}

//ѡ�񣬲�����һ����Ⱥ
void Ga_select(const vector<Mat>& trainData, const vector<Mat>& groundTruth, 
	vector<vector<float>>& population, bool elitism)
{
	vector<vector<float>> new_population;
	vector<float> fitValue;

	Ga_fitness(trainData, groundTruth, population, fitValue);

	int firstPerson = 0;
	if (elitism)//��Ӣ����
	{
		auto maxIndex = max_element(fitValue.begin(), fitValue.end()) - fitValue.begin();
		new_population.push_back(population.at(maxIndex));
		firstPerson = 1;
	}


	vector<double> sum_fit;
	for (auto i : fitValue)
		sum_fit.push_back(i);
	for (int i = 1; i < sum_fit.size(); i++)//��Ӧ���ۼ�
		sum_fit.at(i) += sum_fit.at(i - 1);

	for (int i = firstPerson; i < population.size(); i++)
	{
		int posValue = (rand() % 1000) * 0.001 * sum_fit.back();
		int first = 0;
		int last = sum_fit.size() - 1;
		int mid = 0.5 * (first + last) + 0.5;//��0.5, ��������
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
				index = first;//���� = last, ����һ��ƫ��
				break;
			}
			mid = 0.5 * (first + last);
		}
		new_population.push_back(population.at(index));
	}
	population = new_population;
}

//����
void Ga_cross(vector<vector<float>>& population, double cross_rate)
{
	int len = population.at(0).size();
	for (int person = 0; person < population.size(); person = person + 2)
	{
		double randNums = (rand() % 10 + 0.5) * 0.1;
		if (randNums > cross_rate)//������
			continue;
		int pos = rand() % 10 * 0.1 * len;
		vector<float> seg1(population.at(person).begin() + pos, population.at(person).end());//ѡ���Ļ���Ƭ��
		vector<float> seg2(population.at(person + 1).begin() + pos, population.at(person + 1).end());

		population.at(person) = vector<float>(population.at(person).begin(), population.at(person).begin() + pos);
		population.at(person).insert(population.at(person).end(), seg2.begin(), seg2.end());
		population.at(person + 1) = vector<float>(population.at(person + 1).begin(), population.at(person + 1).begin() + pos);
		population.at(person + 1).insert(population.at(person + 1).end(), seg1.begin(), seg1.end());
	}
}

//����
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

void GA()
{
	double cross_rate = 0.8;//������
	double mutation_rate = 0.01;//������
	bool elitism = false;//�Ƿ�����Ӣ
	int population_size = 3000;//��Ⱥ��С
	int generations = 1;//��������
	vector<vector<float>> population;

	int maskSize = 13;//ģ��ߴ�
	Ga_init(maskSize, population, population_size);
	vector<Mat> trainData;
	vector<Mat> goundTruth;
	loadImg(trainData);
	loadMat(goundTruth);

	time_t startTime, curTime, prevTime;
	time(&startTime);//��ʱ���������ⳬʱ
	while (generations++)
	{
		time(&prevTime);

		cout << "generations: " << generations << " ";
		Ga_select(trainData, goundTruth,population, elitism);
		Ga_cross(population, cross_rate);
		Ga_mutation(population, mutation_rate);

		time(&curTime);
		
		int totalTime = curTime - startTime;
		int diffTime = curTime - prevTime;
		
		cout << "time useed: " << diffTime << "total times: " << totalTime << endl;
	}

	vector<float> fitValue;
	Ga_fitness(trainData, goundTruth, population, fitValue);
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
