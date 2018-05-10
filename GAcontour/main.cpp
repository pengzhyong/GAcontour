#include <iostream>
#include <ctime>
#include "postProcess.h"
#include "GA.h"
#include "NCRF.h"
using namespace std;

int main()
{
	//int maskSize = 9;
	//vector<float> individual;
	//vector<float> individual1;
	//vector<float> individual2;
	//float sumCoef = 0;
	//float a1 = 0;
	//float a2 = 0;
	//float sigma1 = rand() % 100 * 0.01 + 1.6;//0.2;
	//float sigma2 = rand() % 100 * 0.01 + 0.8;//0.8;
	//sigma2 = 2;
	//sigma1 = 3 * sigma2;
	//for (int j = 0; j < maskSize * maskSize; j++)
	//{
	//	int disx = j / maskSize - maskSize / 2;
	//	int disy = j % maskSize - maskSize / 2;
	//	float g1 = 1 / sqrt(2 * CV_PI * sigma1 * sigma1) * (exp(-(disx*disx + disy * disy) / (2 * sigma1 * sigma1)));
	//	//float g2 = 1 / sqrt(2 * CV_PI * sigma2 * sigma2) * (exp(-(disx*disx + disy * disy) / (2 * sigma2 * sigma2)));
	//	float g2 = (-2 * sigma2 * sigma2 + disx * disx + disy * disy) / (2 * CV_PI * sigma2 * sigma2 * sigma2 * sigma2 * sigma2 *sigma2) * exp(-(disx*disx + disy * disy) / (2 * sigma2 * sigma2));
	//	individual.push_back(g1 - g2);
	//	individual1.push_back(g1);
	//	individual2.push_back(g2);
	//	sumCoef += g1 - g2;
	//	a1 += g1;
	//	a2 += g2;
	//}
	//for (int j = 0; j < maskSize * maskSize; j++)
	//{
	//	individual.at(j) /= sumCoef;
	//	individual1.at(j) /= a1;
	//	individual2.at(j) /= a2;
	//}
	//Mat img = imread("myData\\1.jpg");
	//img.convertTo(img, CV_32F, 1 /255.0);
	//cvtColor(img, img, COLOR_BGR2GRAY);
	//Mat msk(maskSize, maskSize, CV_32F);
	//Mat msk1(maskSize, maskSize, CV_32F);
	//Mat msk2(maskSize, maskSize, CV_32F);
	//for (int r = 0; r < maskSize; r++)
	//{
	//	for (int c = 0; c < maskSize; c++)
	//	{
	//		msk.at<float>(r, c) = individual.at(r * maskSize + c);
	//		msk1.at<float>(r, c) = individual1.at(r * maskSize + c);
	//		msk2.at<float>(r, c) = individual2.at(r * maskSize + c);
	//	}
	//}
	//Mat img1, img2;
	//filter2D(img, img, img.depth(), msk);
	//filter2D(img, img1, img.depth(), msk1);
	//filter2D(img, img2, img.depth(), msk2);
	////img = img1 - img2;
	//normalize(img, img, 0, 1, NORM_MINMAX);
	//imshow("img", img2);
	//waitKey();
	//GA();
	string path = "C:\\Users\\pengzhyong\\Desktop\\CV\\бшнд\\contour detection\\contours\\images\\";

	Mat srcImg1 = imread(path + "bear_3.pgm");
	Mat srcImg2 = imread(path + "goat_3.pgm");
	Mat srcImg3 = imread(path + "rino.pgm");


	Mat gtImg1 = imread(path + "gt\\bear_3_gt_binary.pgm");
	Mat gtImg2 = imread(path + "gt\\goat_3_gt_binary.pgm");
	Mat gtImg3 = imread(path + "gt\\rino_gt_binary.pgm");


	vector<Mat> trainVec, gtVec;
	cvtColor(srcImg1, srcImg1, COLOR_BGR2GRAY);
	cvtColor(srcImg2, srcImg2, COLOR_BGR2GRAY);
	cvtColor(srcImg3, srcImg3, COLOR_BGR2GRAY);


	cvtColor(gtImg1, gtImg1, COLOR_BGR2GRAY);
	cvtColor(gtImg2, gtImg2, COLOR_BGR2GRAY);
	cvtColor(gtImg3, gtImg3, COLOR_BGR2GRAY);


	srcImg1.convertTo(srcImg1, CV_32F, 1 / 255.0);
	srcImg2.convertTo(srcImg2, CV_32F, 1 / 255.0);
	srcImg3.convertTo(srcImg3, CV_32F, 1 / 255.0);


	gtImg1.convertTo(gtImg1, CV_32F, 1 / 255.0);
	gtImg2.convertTo(gtImg2, CV_32F, 1 / 255.0);
	gtImg3.convertTo(gtImg3, CV_32F, 1 / 255.0);



	NonCRF(srcImg1, gtImg1);
	//GA(srcImg2, srcImg2, gtImg2, gtImg2);
	//time_t tic, toc;
	//time(&tic);
	//NonCRF(srcImg, gtImg);
	////parallel_for_(Range(1, 2), GaParWrapper(srcImg, gtImg));	
	//time(&toc);
	//cout << (toc - tic) << endl;
	
	//system("pause");
	return 0;
}