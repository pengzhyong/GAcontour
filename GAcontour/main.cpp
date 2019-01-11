#include <iostream>
#include <ctime>
#include <stdio.h>
//#include <random.h>
#include "postProcess.h"
#include "GA.h"
#include "NCRF.h"
#include "bendGabar.h"
#include "circleFilter.h"
#include "myCanny.h"
using namespace std;

Mat src0;
double resolution = 0.5;
double canny_t = 30;
double circle_t = 10;
double rmin = 60;
double rmax = 70;
Mat syntheticFastener(double c, double radius = 100, double radius_hex = 70);
Mat lineInNoise(int width, int cells, double rate);
Mat lineFilter(size_t size, double sigma1, double sigma2);
Mat myFilter2D(const Mat src, const Mat ker);

Mat syntheticFastenerContour(double c, double radius = 100, double radius_hex = 70);
void MHL(Mat cntImg, Mat srcImg);
Mat DetectCircle(Mat& srcImg, double rmin, double rmax, double& relative_error)
{
	Mat rstImg = srcImg.clone();
	//if (rstImg.type() != CV_8U)
	//	cout << "type not CV_8U!, type: " << rstImg.type() << endl;
	

	rstImg.convertTo(rstImg, CV_8U, 255);
	//rstImg.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {p = 255 - p; });
	//imshow("rstImg", rstImg); waitKey(0);
	vector<Vec3f> circles;
	HoughCircles(rstImg, circles, HOUGH_GRADIENT, 2, rstImg.rows / 4, 200, 10, rmin, rmax);// , rmin, rmax);
	//HoughCircles(rstImg, circles, HOUGH_GRADIENT,
	//	resolution, rstImg.rows / 4, 200, circle_t, rmin, rmax);
	//Mat rt = src0.clone();
	if (circles.size() == 0)
	{
		relative_error = 1;
		return rstImg;
	}
	for (size_t i = 0; i < 1; i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		relative_error = sqrt((center.x - srcImg.cols / 2)*(center.x - srcImg.cols / 2) + 
			(center.y - srcImg.rows / 2)*(center.y - srcImg.rows / 2))*1.0 / (0.5*(rmin + rmax));
		/*cout << "center error is: "
			<< sqrt((center.x - srcImg.cols / 2)*(center.x - srcImg.cols / 2) + (center.y - srcImg.rows / 2)*(center.y - srcImg.rows / 2))
			<< "the error/radius is: "
			<< sqrt((center.x - srcImg.cols / 2)*(center.x - srcImg.cols / 2) + (center.y - srcImg.rows / 2)*(center.y - srcImg.rows / 2))*1.0 / (0.5*(rmin + rmax))
			<< endl;*/
		int radius = cvRound(circles[i][2]);
		// draw the circle center
		circle(rstImg, center, 3, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(rstImg, center, radius, Scalar(0), 3, 8, 0);
	}
	return rstImg;
}
void MarkCenter(Mat& srcImg)
{
	int width = srcImg.rows;
	int height = srcImg.cols;
	int marklen = 20;
	line(srcImg, Point(width / 2 - marklen, height / 2), Point(width / 2 + marklen, height / 2), Scalar(0));
	line(srcImg, Point(width / 2, height / 2 - marklen), Point(width / 2, height / 2 + marklen), Scalar(0));

}

int main()
{
	//===============西安交通大学学报修改稿实验
	/*
	/*出现一黑一白两条直线，原因未知
	Mat syn1 = lineInNoise(300, 60, 0.7);
	String winname = "syn1";
	namedWindow(winname, 0);
	imshow(winname, syn1);
	waitKey();
	Mat ft = lineFilter(45, -1);
	String win1 = "win1";
	normalize(ft, ft, 0,1, NORM_MINMAX);
	namedWindow(win1, 0);

	imshow(win1, ft);
	waitKey();
	Mat syn1_dst;
	filter2D(syn1, syn1_dst, syn1.depth(), ft);
	Mat syn_float(syn1);
	syn_float.convertTo(syn_float, CV_32F, 1.0 / 255);

	Mat ret = myFilter2D(syn_float, ft);
	normalize(ret, ret, 0,1, NORM_MINMAX);
	imshow("ret", ret);
	waitKey();
	for (int i = 0; i < syn1.rows; i++)
	{
		syn1_dst.at<char>(i, 0) = 0;
	}
	normalize(syn1_dst, syn1_dst, 0,255, NORM_MINMAX);

	imshow("syn1_dst", syn1_dst);
	waitKey();*_/

	Mat syn1 = lineInNoise(800, 40, 0.5);
	String winname = "syn1";
	namedWindow(winname, 1);
	imshow(winname, syn1);
	waitKey();
	int ksz = 101;

	Mat syn1_save = syn1.clone();
	syn1_save = syn1_save(Rect(ksz / 2, ksz / 2, syn1_save.rows - ksz, syn1_save.cols - ksz));
	syn1_save.forEach<char>([](char& p, const int* pos)->void {p = 255 - p; });
	imwrite("西交学报R-CNRF演示.jpg", syn1_save);
	
	Mat ft = lineFilter(ksz, 35,20);
	String win1 = "win1";
	normalize(ft, ft, 0, 1, NORM_MINMAX);
	namedWindow(win1, 1);

	imshow(win1, ft);
	waitKey();
	//Mat syn1_dst;
	//filter2D(syn1, syn1_dst, syn1.depth(), ft);
	Mat syn_float(syn1);
	syn_float.convertTo(syn_float, CV_32F, 1.0 / 255);

	Mat ret = myFilter2D(syn_float, ft);
	normalize(ret, ret, 0, 1, NORM_MINMAX);
	String strret = "ret";
	namedWindow(strret, 1);
	imshow(strret, ret);
	waitKey();
	
	Mat ret_save = ret.clone();
	ret_save.convertTo(ret_save, CV_8U, 255);
	imwrite("西交学报R-CNRF演示1.jpg", ret_save);

	ret_save.forEach<char>([](char& p, const int* pos)->void {p = 255 - p; });

	Mat rst = ret_save + syn1_save;
	normalize(rst, rst, 0, 255, NORM_MINMAX);
	imshow("rst", rst); waitKey();
	imwrite("西交R-NCRF_2.jpg", rst);

	*/
	//=================

	/*VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened())
	{
		cout << "the camera cannot open" << endl; 
		system("pause");
	}
	Mat frame;
	Mat rst;
	while (1)
	{
		cap.read(frame);
		if (frame.empty())
		{
			cout << "empty frame! " << endl;
			waitKey(10);
			continue;
		}
		blur(frame, frame, Size(5, 5));
		Mat dst = myspace::Mycanny(frame, 0.05, 0.2);
		imshow("frame", dst); waitKey(10);
	}*/
 	string mulu = "C:\\Users\\pzy\\Desktop\\小论文\\螺母\\分类\\水渍\\";
	int num = 0;
	int i = 0;
	while (0 && ++i<=num)
	{
		cout << i << endl;
		string fname = mulu + to_string(i) + "_看图王.jpg";
		if(i < 10)
			fname = mulu + "0" + to_string(i) + "_看图王.jpg";
		Mat src = imread(fname);
		//Mat src = imread("C:\\Users\\pzy\\Desktop\\小论文\\螺母\\lena.jpg");
		cvtColor(src, src, COLOR_BGR2GRAY);
		pyrDown(src, src);
		src0 = src.clone();

		//pyrDown(src, src);
		//pyrDown(src, src);
		imshow("src", src);
		Mat canImg;
		Mat s1 = src.clone();
		Mat s2 = src.clone();
		Mat s3 = src.clone();
		Mat s4 = src.clone();
		double tlow = 0.08;
		double thigh = 0.16;
		Mat dst1 = myspace::Mycanny(s1, tlow, thigh, 0, 0);
		Mat dst2 = myspace::Mycanny(s2, tlow, thigh, 0, 1);
		Mat dst3 = myspace::Mycanny(s3, tlow, thigh, 1, 0);
		Mat dst4 = myspace::Mycanny(s4, tlow, thigh, 1, 1);
		dst1.convertTo(dst1, CV_8U, 255);
		dst2.convertTo(dst2, CV_8U, 255);
		dst3.convertTo(dst3, CV_8U, 255);
		dst4.convertTo(dst4, CV_8U, 255);
		imshow("myCanny", dst1);
		imshow("myCanny excitation", dst2);
		imshow("myCanny inhibition", dst3);
		imshow("myCanny both", dst4);
		//waitKey();
		double bar;
		Mat c1 = DetectCircle(dst1, rmin, rmax, bar);
		Mat c2 = DetectCircle(dst2, rmin, rmax, bar);
		Mat c3 = DetectCircle(dst3, rmin, rmax, bar);
		Mat c4 = DetectCircle(dst4, rmin, rmax, bar);
		imshow("c1", c1);
		imshow("c2", c2);
		imshow("c3", c3);
		imshow("c4", c4);
		waitKey();
	}
	//Mat src = imread("C:\\Users\\pzy\\Desktop\\小论文\\螺母\\分类\\正常\\09_看图王.jpg");
	Mat src = imread("C:\\Users\\pengzhyong\\Desktop\\小论文\\螺母\\kj2.jpg");
	cvtColor(src, src, COLOR_BGR2GRAY);
	//rotate(src, src, ROTATE_90_COUNTERCLOCKWISE);
	//pyrDown(src, src);
	//pyrDown(src, src);
	//pyrDown(src, src);

	//src = syn1;
	//imshow("src", src); 
	//waitKey();
	Mat can;
	Canny(src, can, 80, 150);
	can.forEach<uchar>([](uchar& p, const int*)
		->void{p = 255 - p; });
	//imshow("can", can);
	imwrite("pic\\can.jpg", can);
	//waitKey();

	//合成噪声实验
	//Mat src = imread("C:\\Users\\pzy\\Desktop\\小论文\\螺母\\kj2.jpg");
	//int outimes = 30;
	//double stride = 0.01;//信噪比变化的单位幅度
	//vector<int> v1, v2, v3, v4, v5;//
	//double snr = 0.7;                                                                                                                    
	//while (outimes--)
	//{
	//	int times = 100;//100 times experiments
	//	int t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;//成功定位次数
	//	double error_t = 0.05;//相对误差，用于判定圆定位是否准确
	//	snr += stride;
	//	while (times--)
	//	{
	//		Mat src = syntheticFastener(snr, 120, 80);
	//		imshow("src", src); waitKey(0);
	//		//cvtColor(src, src, COLOR_BGR2GRAY);
	//		pyrDown(src, src);
	//		Mat s1 = src.clone();
	//		Mat s2 = src.clone();
	//		Mat s3 = src.clone();
	//		Mat s4 = src.clone();
	//		Mat s5 = src.clone();
	//		Mat s6 = src.clone();
	//		s5.forEach<uchar>([](uchar& p, const int* pos)
	//			->void {p = 255 - p; });
	//		double tlow = 0.1;
	//		double thigh = 0.2;
	//		Mat dst1 = myspace::Mycanny(s1, 0.3, 0.6, 0, 0);
	//		Mat dst2 = myspace::Mycanny(s2, 0.3, 0.6, 0, 1);
	//		Mat dst3 = myspace::Mycanny(s3, 0.3, 0.5, 1, 0);
	//		Mat dst4 = myspace::Mycanny(s4, 0.3, 0.6, 1, 1);
	//		Mat dst5;
	//		Canny(s5, dst5, 1, 999);
	//		imshow("myCanny", dst1);
	//		imshow("myCanny excitation", dst2);
	//		imshow("myCanny inhibition", dst3);
	//		imshow("myCanny both", dst4);
	//		imshow("Canny", dst5);
	//		Mat dst6 = NonCrf_fastener(s6, 2, 0.05, 0.2);
	//		waitKey();
	//		double re1 = 0, re2 = 0, re3 = 0, re4 = 0;
	//		Mat c1 = DetectCircle(dst1, 59, 61, re1);
	//		Mat c2 = DetectCircle(dst2, 59, 61, re2);
	//		Mat c3 = DetectCircle(dst3, 59, 61, re3);
	//		Mat c4 = DetectCircle(dst4, 59, 61, re4);
	//		if (re1 < error_t) t1++;
	//		if (re2 < error_t) t2++;
	//		if (re3 < error_t) t3++;
	//		if (re4 < error_t) t4++;
	//	}
	//	v1.push_back(t1);
	//	v2.push_back(t2);
	//	v3.push_back(t3);
	//	v4.push_back(t4);
	//}
	//cout << "v1: " << endl;
	//for (auto i : v1) cout << i << ", ";cout << endl;
	//cout << "v2: " << endl;
	//for (auto i : v2) cout << i << ", ";cout << endl; 
	//cout << "v3: " << endl;
	//for (auto i : v3) cout << i << ", ";cout << endl; 
	//cout << "v4: " << endl;
	//for (auto i : v4) cout << i << ", ";cout << endl;

	int times = 100;//100 times experiments
	int t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0;
	double st1, st2, st3, st4, st5, ed1, ed2, ed3, ed4, ed5;
	double tm1=0, tm2=0, tm3=0, tm4=0, tm5=0;
	double error_t = 0.05;
	double snr = 0.1;
	while (times--)
	{
		Mat src = syntheticFastener(1-snr, 120, 80);
		//imshow("src", src);
		//cvtColor(src, src, COLOR_BGR2GRAY);
		pyrDown(src, src);
		Mat s1 = src.clone();
		Mat s2 = src.clone();
		Mat s3 = src.clone();
		Mat s4 = src.clone();
		Mat s5 = src.clone();
		Mat s6 = src.clone();

		s5.forEach<uchar>([](uchar& p, const int* pos)
			->void {p = 255 - p; });
		double tlow = 0.1;
		double thigh = 0.2;
		st1 = clock();
		Mat dst1 = myspace::Mycanny(s1, 0.3, 0.6, 0, 0);
		ed1 = clock();

		st2 = clock();
		Mat dst2 = myspace::Mycanny(s3, 0.3, 0.5, 1, 0);
		ed2 = clock();

		st3 = clock();
		Mat dst3 = myspace::Mycanny(s2, 0.3, 0.6, 0, 1);
		ed3 = clock();

		st4 = clock();
		Mat dst4 = myspace::Mycanny(s4, 0.3, 0.6, 1, 1);
		ed4 = clock();

		//Mat dst5;
		//Canny(s5, dst5, 1, 999);
		st5 = clock();
		Mat dst5 = NonCrf_fastener(s6, 2, 0.05, 0.2);
		ed5 = clock();

		tm1 += ed1 - st1;
		tm2 += ed2 - st2;
		tm3 += ed3 - st3;
		tm4 += ed4 - st4;
		tm5 += ed5 - st5;


		//imshow("myCanny", dst1);
		//imshow("myCanny excitation", dst2);
		//imshow("myCanny inhibition", dst3);
		//imshow("myCanny both", dst4);
		//imshow("Canny", dst5);
		//waitKey();
		double re1 = 0, re2 = 0, re3 = 0, re4 = 0, re5;
		Mat c1 = DetectCircle(dst1, 59, 61, re1);
		Mat c2 = DetectCircle(dst2, 59, 61, re2);
		Mat c3 = DetectCircle(dst3, 59, 61, re3);
		Mat c4 = DetectCircle(dst4, 59, 61, re4);
		Mat c5 = DetectCircle(dst5, 59, 61, re5);
		if (re1 < error_t) t1++;
		if (re2 < error_t) t2++;
		if (re3 < error_t) t3++;
		if (re4 < error_t) t4++;
		if (re5 < error_t) t5++;


		MarkCenter(c1);
		MarkCenter(c2);
		MarkCenter(c3);
		MarkCenter(c4);
		MarkCenter(c5);

		/*imshow("c1", c1);
		imshow("c2", c2);
		imshow("c3", c3);
		imshow("c4", c4);
		imwrite("pic/c1.jpg", c1);
		imwrite("pic/c2.jpg", c2);
		imwrite("pic/c3.jpg", c3);
		imwrite("pic/c4.jpg", c4);
		waitKey();*/
		cout << times << ":  " << re1 << "; " << re2 << "; " << re3 << "; " << re4 << "; " << re5 << endl;
	}
	cout << "t1_Canny: " << t1 << ", time cost: " << tm1 << endl;
	cout << "t2_Inhibition: " << t2 << ", time cost: " << tm2 << endl;
	cout << "t3_Exatation: " << t3 << ", time cost: " << tm3 << endl;
	cout << "t4_Both: " << t4 << ", time cost: " << tm4 << endl;
	cout << "t5_NCRF: " << t5 << ", time cost: " << tm5 << endl;

	getchar();

	//Mat src = syntheticFastener(0.9, 120, 80);
	////imshow("src", src);
	////cvtColor(src, src, COLOR_BGR2GRAY);
	//pyrDown(src, src);
	//imwrite("experiment\\src.jpg", src);
	//Mat wt(src.size(), CV_8U, Scalar(255));
	//imwrite("experiment\\white.jpg", wt);
	//
	//Mat s1 = src.clone();
	//Mat s2 = src.clone();
	//Mat s3 = src.clone();
	//Mat s4 = src.clone();
	//Mat s5 = src.clone();
	//s5.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {p = 255 - p; });
	//double tlow = 0.1;
	//double thigh = 0.2;
	//Mat dst1 = myspace::Mycanny(s1, 0.3, 0.6 ,0, 0);
	//Mat dst2 = myspace::Mycanny(s2, 0.3, 0.6, 0, 1);
	//Mat dst3 = myspace::Mycanny(s3, 0.3, 0.6, 1, 0);
	//Mat dst4 = myspace::Mycanny(s4, 0.3, 0.6, 1, 1);
	//dst1.convertTo(dst1, CV_8U, 255);
	//dst2.convertTo(dst2, CV_8U, 255);
	//dst3.convertTo(dst3, CV_8U, 255);
	//dst4.convertTo(dst4, CV_8U, 255);
	//imshow("myCanny", dst1);
	//imshow("myCanny excitation", dst2);
	//imshow("myCanny inhibition", dst3);
	//imshow("myCanny both", dst4);
	//imwrite("experiment\\dst1.jpg", dst1);
	//imwrite("experiment\\dst2.jpg", dst2);
	//imwrite("experiment\\dst3.jpg", dst3);
	//imwrite("experiment\\dst4.jpg", dst4);
	////imshow("Canny", dst5);
	////waitKey();
	//double tmp;
 //	Mat c1 = DetectCircle(dst1, 59, 61, tmp);
	//Mat c2 = DetectCircle(dst2, 59, 61, tmp);
	//Mat c3 = DetectCircle(dst3, 59, 61, tmp);
	//Mat c4 = DetectCircle(dst4, 59, 61, tmp);
	//MarkCenter(c1);
	//MarkCenter(c2);
	//MarkCenter(c3);
	//MarkCenter(c4);
	//imshow("c1", c1);
	//imshow("c2", c2);
	//imshow("c3", c3);
	//imshow("c4", c4);
	//imwrite("experiment\\c1.jpg", c1);
	//imwrite("experiment\\c2.jpg", c2);
	//imwrite("experiment\\c3.jpg", c3);
	//imwrite("experiment\\c4.jpg", c4);
	//waitKey();
	Mat canImg;
	Mat s1 = src.clone();
	Mat s2 = src.clone();
	Mat s3 = src.clone();
	Mat s4 = src.clone();
	Mat s5 = src.clone();
	double tlow = 0.1;
	double thigh = 0.2;
	Mat dst1 = myspace::Mycanny(s1, tlow, thigh, 0, 0);
	Mat dst2 = myspace::Mycanny(s2, tlow, thigh, 0, 1);
	Mat dst3 = myspace::Mycanny(s3, tlow, thigh, 1, 0);
	Mat dst4 = myspace::Mycanny(s4, tlow, thigh, 1, 1);
	dst1.convertTo(dst1, CV_8U, 255);
	dst2.convertTo(dst2, CV_8U, 255);
	dst3.convertTo(dst3, CV_8U, 255);
	dst4.convertTo(dst4, CV_8U, 255);

	imshow("myCanny", dst1);
	imshow("myCanny excitation", dst2);
	imshow("myCanny inhibition", dst3);
	imshow("myCanny both", dst4);
	waitKey();
	Mat dst5 = NonCrf_fastener(s5, 2, 0.05, 0.2);

	double bar;
	Mat c1 = DetectCircle(dst1, rmin, rmax, bar);
	Mat c2 = DetectCircle(dst2, rmin, rmax, bar);
	Mat c3 = DetectCircle(dst3, rmin, rmax, bar);
	Mat c4 = DetectCircle(dst4, rmin, rmax, bar);
	/*imshow("c1", c1);
	imshow("c2", c2);
	imshow("c3", c3);
	imshow("c4", c4);
	waitKey();*/
	//Mat drawImg = dst2.clone();
    MHL(dst2, src);


	//Canny(src, canImg, 60, 250);
	//imshow("canImg", canImg); 
	//waitKey(0);

	drawRNCRF();

	//drawButterflyNCRF();
	//int maskSize = 9;
	//vector<double> individual;
	//vector<double> individual1;
	//vector<double> individual2;
	//double sumCoef = 0;
	//double a1 = 0;
	//double a2 = 0;
	//double sigma1 = rand() % 100 * 0.01 + 1.6;//0.2;
	//double sigma2 = rand() % 100 * 0.01 + 0.8;//0.8;
	//sigma2 = 2;
	//sigma1 = 3 * sigma2;
	//for (int j = 0; j < maskSize * maskSize; j++)
	//{
	//	int disx = j / maskSize - maskSize / 2;
	//	int disy = j % maskSize - maskSize / 2;
	//	double g1 = 1 / sqrt(2 * CV_PI * sigma1 * sigma1) * (exp(-(disx*disx + disy * disy) / (2 * sigma1 * sigma1)));
	//	//double g2 = 1 / sqrt(2 * CV_PI * sigma2 * sigma2) * (exp(-(disx*disx + disy * disy) / (2 * sigma2 * sigma2)));
	//	double g2 = (-2 * sigma2 * sigma2 + disx * disx + disy * disy) / (2 * CV_PI * sigma2 * sigma2 * sigma2 * sigma2 * sigma2 *sigma2) * exp(-(disx*disx + disy * disy) / (2 * sigma2 * sigma2));
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
	//img.convertTo(img, CV_64F, 1 /255.0);
	//cvtColor(img, img, COLOR_BGR2GRAY);
	//Mat msk(maskSize, maskSize, CV_64F);
	//Mat msk1(maskSize, maskSize, CV_64F);
	//Mat msk2(maskSize, maskSize, CV_64F);
	//for (int r = 0; r < maskSize; r++)
	//{
	//	for (int c = 0; c < maskSize; c++)
	//	{
	//		msk.at<double>(r, c) = individual.at(r * maskSize + c);
	//		msk1.at<double>(r, c) = individual1.at(r * maskSize + c);
	//		msk2.at<double>(r, c) = individual2.at(r * maskSize + c);
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
  	string path = "C:\\Users\\pengzhyong\\Desktop\\CV\\论文\\contour detection\\contours\\images\\";

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

	srcImg1.convertTo(srcImg1, CV_64F, 1 / 255.0);
	srcImg2.convertTo(srcImg2, CV_64F, 1 / 255.0);
	srcImg3.convertTo(srcImg3, CV_64F, 1 / 255.0);

	gtImg1.convertTo(gtImg1, CV_64F, 1 / 255.0);
	gtImg2.convertTo(gtImg2, CV_64F, 1 / 255.0);
	gtImg3.convertTo(gtImg3, CV_64F, 1 / 255.0);

	//Mat circleImg(srcImg3.size(), srcImg3.type(), Scalar(0));
	//circle(circleImg, Point2f(srcImg3.rows / 2, srcImg3.cols / 2), 100, Scalar(1));
	///*circleImg.forEach<double>([](double& p, const int* pos)
	//	->void {double rd = rand() % 2; if (rd < 0.001) p = 1.0; });*/
	//for (int r = 0; r < circleImg.rows; r++)
	//{  
	//	for (int c = 0; c < circleImg.cols; c++)
	//	{
	//		double rd = rand() % 100 * 0.01;
	//		if (rd < 0.3)
	//			circleImg.at<double>(r, c) = 1.0;
	//	}
	//}
	//imshow("circle", circleImg);
	//waitKey(0);
	//Mat c1 = CircleDetection(circleImg, 100);
	//Mat c2 = CircleDetection(c1, 100);
	//Mat c3 = CircleDetection(c2, 100);
	//Mat c4 = CircleDetection(c3, 100);
	//Mat c5 = CircleDetection(c4, 100);
	//Mat c6 = CircleDetection(c5, 100);
	//Mat c7 = CircleDetection(c6, 100);

	/*Mat fastener = syntheticFastener(0.9,120,80);
	imshow("systhetic", fastener);
	waitKey();
	imwrite("pic/lmImg.jpg", fastener);*/


	//Mat lm1ROI = fastener(Rect(fastener.rows/2.0-100, fastener.cols / 2.0 - 100, 200, 200));
	//imshow("ROI", lm1ROI); waitKey();
	//vector<Vec2f> lines0;
	//lm1ROI.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {if (p > 0)p = 0; else p = 255; });
	//HoughLines(lm1ROI, lines0, 1, 5*CV_PI / 180.0, 40);// , 0, 0, 10.0 / 180.0*CV_PI);
	//for (size_t i = 0; i < lines0.size(); i++)
	//{
	//	double rho = lines0[i][0];
	//	double theta = lines0[i][1];
	//	double a = cos(theta), b = sin(theta);
	//	double x0 = a*rho, y0 = b*rho;
	//	Point pt1(cvRound(x0 + 1000 * (-b)),
	//		cvRound(y0 + 1000 * (a)));
	//	Point pt2(cvRound(x0 - 1000 * (-b)),
	//		cvRound(y0 - 1000 * (a)));
	//	line(lm1ROI, pt1, pt2, Scalar(255), 1, 8);
	//}
	//imshow("test line", lm1ROI);
	//waitKey();


	string path_lm = "C:\\Users\\pengzhyong\\Desktop\\螺母\\分类\\背景\\";
	Mat lmImg = imread(path_lm + "03_看图王.jpg");
	lmImg.convertTo(lmImg, CV_8U);
	pyrDown(lmImg, lmImg);
	//pyrDown(lmImg, lmImg);
	//pyrDown(lmImg, lmImg);


	/*while (max(lmImg.rows, lmImg.cols) > 100)
	{
		pyrDown(lmImg, lmImg);
	}*/
	cvtColor(lmImg, lmImg, COLOR_BGR2GRAY);

	Mat srcImg = lmImg.clone();
	//lmImg = fastener.clone();

	//----------canny
	blur(lmImg, lmImg, Size(3,3));
	Mat cannyImg;
	Canny(lmImg, cannyImg, 80,120, 3);
	cannyImg.forEach<uchar>([](uchar& p, const int* pos)
		->void {if (p > 0) p = 0; else p = 255; });
	imshow("canny", cannyImg);
	waitKey();
	imwrite("pic/canny_0.jpg", cannyImg);
	lmImg.convertTo(lmImg, CV_64F, 1 / 255.0);
	Mat roteMat = getRotationMatrix2D(Point2f(lmImg.cols / 2.0, lmImg.rows / 2.0), 0, 1.0);
	Mat result;
	warpAffine(srcImg, result, roteMat, lmImg.size());
	lmImg = result.clone();
	vector<Vec3f> circles1;
	Mat tmp1 = cannyImg.clone();
	//tmp1.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {if (p > 0)p = 0; else p = 255; });
	//imshow("tmp1", tmp1);
	//waitKey();
	Mat circleImg1 = tmp1.clone();
	HoughCircles(tmp1, circles1, HOUGH_GRADIENT,
		resolution, tmp1.rows / 4, canny_t, circle_t, rmin, rmax);
	for (size_t i = 0; i < 1; i++)
	{
		Point center(cvRound(circles1[i][0]), cvRound(circles1[i][1]));
		int radius = cvRound(circles1[i][2]);
		// draw the circle center
		circle(circleImg1, center, 3, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(circleImg1, center, radius, Scalar(0), 3, 8, 0);
	}
	imshow("hough circle1", circleImg1);
	waitKey();
	imwrite("pic/cannycircle_0.jpg", circleImg1);

	//-----------non-crf
	//fastener.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {if (p > 0)p = 255; else p = 0; });
	Mat lm1 = NonCrf_fastener(srcImg, 2, 0.05, 0.2);
   	MHL(lm1, srcImg);

	//Mat tmp = lm1.clone();
	//tmp.convertTo(tmp, CV_8U, 255);
	//Mat circleImg = tmp.clone();
	////circleImg.forEach<uchar>([](uchar& p, const int* pos)
	////	->void {if (p >0)p = 0; else p = 255; });
	//vector<Vec3f> circles;
	//HoughCircles(tmp, circles, HOUGH_GRADIENT,
	//	1, tmp.rows / 4, 100, 100, 90, 110);
	//for (size_t i = 0; i < circles.size(); i++)
	//{
	//	Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
	//	int radius = cvRound(circles[i][2]);
	//	// draw the circle center
	//	circle(circleImg, center, 3, Scalar(0), -1, 8, 0);
	//	// draw the circle outline
	//	circle(circleImg, center, radius, Scalar(0), 3, 8, 0);
	//}
	//imshow("hough circle NCRF", circleImg);
	//waitKey();

#if 0
	//------LSD line segment detector
	lm1.convertTo(lm1, CV_8U, 255);
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	//Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
	vector<Vec4f> lines_std;
	// Detect the lines
	ls->detect(lm1, lines_std);
	vector<double> lines_len;
	for (auto i : lines_std)
		lines_len.push_back((i[0] - i[2])*(i[0] - i[2]) + (i[1] - i[3])*(i[1] - i[3]));
	sort(lines_len.begin(), lines_len.end());
	double mantain_rate = 0.35;//线段中保留比例
	vector<Vec4f> lines_new;
	for (auto i : lines_std)
	{
		if (((i[0] - i[2])*(i[0] - i[2]) + (i[1] - i[3])*(i[1] - i[3])) > lines_len.at(lines_len.size() * (1 - mantain_rate)))
			lines_new.push_back(i);
	}

	Mat drawnLines(lm1);
	ls->drawSegments(drawnLines, lines_new);
	imshow("Standard refinement", drawnLines);
	waitKey(0);
	//----------find hexagon by angle difference
	//----------if a segment is of hexagon, the different of angle to other edges is 0, 60, 120
	vector<double> lines_k;
	for (auto i : lines_new)
	{
		double slope = atan((i[1] - i[3]) / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;
	const double T = 5.0 / 180.0 * CV_PI;
	for (int i = 0; i < lines_new.size(); i++)
	{
		int cnt = 0;
		for (int j = 0; j < lines_new.size(); j++)
		{
			if(j==i) continue;
			if(abs(abs(lines_k[i] - lines_k[j]-1.0 * CV_PI / 3.0)) < T 
				|| abs(abs(lines_k[i] - lines_k[j] - 2.0 * CV_PI / 3.0)) < T)
			{
				cnt++;
			}
		}
		double len = sqrt((lines_new[i][0] - lines_new[i][2])*(lines_new[i][0] - lines_new[i][2]) + (lines_new[i][1] - lines_new[i][3])*(lines_new[i][1] - lines_new[i][3]));
		if (cnt >= 3)
			hexagon.push_back(lines_new[i]);
	}

	Mat drawnLines2(lm1);
	ls->drawSegments(drawnLines2, hexagon);
	namedWindow("Standard", 0);
	imshow("Standard", drawnLines2);
	waitKey(0);
#endif
	//---------HoughLine detector
	lm1.convertTo(lm1, CV_8U, 255);
	imwrite("pic/ncrf_0.jpg", lm1);;

	//lm1.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {if (p == 0) p = 255;
	//	else p = 0; });
	Mat dst_lines(lm1.size(), CV_8UC3);

	vector<Vec3f> circles2;
	HoughCircles(lm1, circles2, HOUGH_GRADIENT,
		1, lm1.rows / 4, 60, 10, 55, 60);
	Point center0(cvRound(circles2[0][0]), cvRound(circles2[0][1]));
	int radius0 = cvRound(circles2[0][2]);
	for (size_t i = 0; i < min((int)circles2.size(),1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(lm1, center, 3, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(lm1, center, radius, Scalar(0), 3, 8, 0);
	}
	imshow("hough circle2", lm1);
	imwrite("pic/ncrfcircle_0.jpg", lm1);;
	waitKey();
#if 0
	lm1.forEach<uchar>([](uchar& p, const int* pos)
		->void {if (p > 0)p = 0; else p = 255; });
	vector<Vec2f> lines;
	Mat lm1ROI = lm1(Rect(center0.x-radius0,center0.y-radius0, 2.0*radius0, 2.0*radius0));

	HoughLines(lm1ROI, lines, 3, 5*CV_PI / 180.0, 50);// , 0, 0, 10.0 / 180.0*CV_PI);
	lm1ROI = srcImg(Rect(center0.x - 1.5*radius0, center0.y - 1.5*radius0, 3 * radius0, 3 * radius0));
	for (size_t i = 0; i < lines.size(); i++)
	{
		double rho = lines[i][0];
		double theta = lines[i][1];
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		Point pt1(cvRound(x0 + 1000 * (-b)),
			cvRound(y0 + 1000 * (a)));
		Point pt2(cvRound(x0 - 1000 * (-b)),
			cvRound(y0 - 1000 * (a)));
		line(lm1ROI, pt1, pt2, Scalar(255), 1, 8);
	}
	imshow("lm1ROI", lm1ROI); waitKey();
#else
	lm1.forEach<uchar>([](uchar& p, const int* pos)
		->void {if (p > 0)p = 0; else p = 255; });
	vector<Vec4i> lines;
	HoughLinesP(lm1, lines, 1, 5 * CV_PI / 180, 20, 10, 5);
	for (size_t i = 0; i < lines.size(); i++)
	{
		line(dst_lines, Point(lines[i][0], lines[i][1]),
			Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
	}
#endif
	//namedWindow("houghLine", 0);
	imshow("houghLine", dst_lines);
	waitKey();

	//----------find hexagon by angle difference
	//----------if a segment is of hexagon, the different of angle to other edges is 0, 60, 120
	Mat hexMat(lm1.size(), CV_8U, Scalar(255));
	vector<double> lines_k;
	for (auto i : lines)
	{

		double slope = atan((i[1] - i[3]) * 1.0 / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;
	const double T = 15.0 / 180.0 * CV_PI;

	//-------画图用，不画圆外直线
	Point center(cvRound(circles2[0][0]), cvRound(circles2[0][1]));
	int radius = cvRound(circles2[0][2]);
	//--------
	for (int i = 0; i < lines.size(); i++)
	{
		int cnt = 0;
		for (int j = 0; j < lines.size(); j++)
		{
			if (j == i) continue;
			if (abs(abs(abs(lines_k[i]) - abs(lines_k[j])) - 1.0 * CV_PI / 3.0) < T
				|| abs(abs(abs(lines_k[i]) - abs(lines_k[j])) - 2.0 * CV_PI / 3.0) < T)
			{
				cnt++;
			}
		}
		double len = sqrt((lines[i][0] - lines[i][2])*(lines[i][0] - lines[i][2]) + (lines[i][1] - lines[i][3])*(lines[i][1] - lines[i][3]));
		if (cnt >= 2)// && len < 100)
		{
			if(sqrt((lines[i][0]-center.x)*(lines[i][0] - center.x)+ (lines[i][1] - center.y)*(lines[i][1] - center.y))>=radius-5
				||sqrt((lines[i][0] - center.x)*(lines[i][0] - center.x) + (lines[i][1] - center.y)*(lines[i][1] - center.y)) < radius*2.0/3.0
				||sqrt((lines[i][2] - center.x)*(lines[i][2] - center.x) + (lines[i][3] - center.y)*(lines[i][3] - center.y))>=radius-5
				||sqrt((lines[i][2] - center.x)*(lines[i][2] - center.x) + (lines[i][3] - center.y)*(lines[i][3] - center.y)) <radius*2.0 / 3.0)
				continue;
			hexagon.push_back(lines[i]);
			line(hexMat, Point2f(lines[i][0], lines[i][1]), Point2f(lines[i][2], lines[i][3]), Scalar(0));
		}
	}

	for (size_t i = 0; i < min((int)circles2.size(),1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(hexMat, center, 2, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(hexMat, center, radius, Scalar(0), 2, 8, 0);
	}

	Mat drawnLines2(hexMat);
	//namedWindow("Standard", 0);
	imshow("Standard", drawnLines2);
	waitKey(0);
	imwrite("pic/circleHex.jpg", drawnLines2);

	Mat drawcorner(drawnLines2.size(), drawnLines2.type(),Scalar(0, 0, 0));
	drawcorner = drawnLines2.clone();
	drawcorner = srcImg.clone();
	for (size_t i = 0; i < circles2.size(); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(drawcorner, center, 2, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(drawcorner, center, radius, Scalar(0), 2, 8, 0);
	}
	//vector<Point2f> hex_corners;
	map<double, Point2f> hex_corners;
	const double segdistT = 40;//线段距离阈值，低于此阈值才被认为是真正的六角螺母边
	for (int i = 0; i < hexagon.size(); i++)
	{
		double slope1 = atan((hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]));
		for (int j = 0; j < hexagon.size(); j++)
		{
			if (j == i) continue;
			double slope2 = atan((hexagon[j][1] - hexagon[j][3]) * 1.0 / (hexagon[j][0] - hexagon[j][2]));
			if (abs(abs(slope2 - slope1) - 1.0 * CV_PI / 3.0) < T 
				|| abs(abs(slope2 - slope1) - 2.0 * CV_PI / 3.0) < T)
			{
				double dist1, dist2, dist3, dist4;
				dist1 = sqrt((hexagon[i][0] - hexagon[j][0])*(hexagon[i][0] - hexagon[j][0])
					+ (hexagon[i][1] - hexagon[j][1])*(hexagon[i][1] - hexagon[j][1]));
				dist2 = sqrt((hexagon[i][0] - hexagon[j][2])*(hexagon[i][0] - hexagon[j][2])
					+ (hexagon[i][1] - hexagon[j][3])*(hexagon[i][1] - hexagon[j][3]));
				dist3 = sqrt((hexagon[i][2] - hexagon[j][0])*(hexagon[i][2] - hexagon[j][0])
					+ (hexagon[i][3] - hexagon[j][1])*(hexagon[i][3] - hexagon[j][1]));
				dist4 = sqrt((hexagon[i][2] - hexagon[j][2])*(hexagon[i][2] - hexagon[j][2])
					+ (hexagon[i][3] - hexagon[j][3])*(hexagon[i][3] - hexagon[j][3]));
				double segdist = min(min(dist1, dist2), min(dist3, dist4));//定义线段之间的距离为两线段端点之间的最短距离
				
  				if (1)//segdist < segdistT)
				{
					//line(drawcorner, Point(hexagon[i][0], hexagon[i][1]), Point(hexagon[i][2], hexagon[i][3]), Scalar(0, 255, 0));
					line(drawcorner, Point(hexagon[j][0], hexagon[j][1]), Point(hexagon[j][2], hexagon[j][3]), Scalar(0));

					double k1, k2, b1, b2;
					k1 = (hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]);
					k2 = (hexagon[j][1] - hexagon[j][3]) * 1.0 / (hexagon[j][0] - hexagon[j][2]);
					b1 = hexagon[i][1] - k1 * hexagon[i][0];
					b2 = hexagon[j][1] - k2 * hexagon[j][0];
					double cross_x = (b2 - b1) / (k1 - k2);
					double cross_y = k2 * cross_x + b2;
					//hex_corners.insert(segdist, Point2f(cross_x, cross_y));
					if(sqrt((cross_x-center.x)*(cross_x - center.x)+ (cross_y - center.y)*(cross_y - center.y))<radius-5
						&& sqrt((cross_x - center.x)*(cross_x - center.x) + (cross_y - center.y)*(cross_y - center.y))>radius*3.22 / 4.0)
   						hex_corners[segdist] = Point2f(cross_x, cross_y);//六角角点map, 线段距离为键，交点坐标为值
				}
			}
		}
	}
	Mat finalRst = drawcorner.clone();
	for (auto it = hex_corners.begin(); it != hex_corners.end(); ++it)
	{
		circle(finalRst, Point(it->second.x, it->second.y), 5, Scalar(255),1);
	}
	imshow("hex_corner", finalRst);
	imwrite("pic/hexcorner.jpg", finalRst);
	waitKey();

	//-----------------distance constrain to pick out 3 convex points
	double len = 20.0;
	len = (rmax + rmin) * 0.909 / 2.0;
	double st_dst[3] = { len, 1.732*len, 2.0*len };
	double dist_T =3;
	vector<Point2f> hexVec;
	for (auto it = hex_corners.begin(); it != hex_corners.end(); ++it)
	{
		hexVec.push_back(Point2f(it->second.x, it->second.y));
	}
	vector<int> votes(hexVec.size(), 0);
	for (int i = 0; i < hexVec.size(); i++)
	{
		for (int j = 0; j < hexVec.size(); j++)
		{
			if (i != j)
			{
				double dist = sqrt((hexVec[i].x - hexVec[j].x)*(hexVec[i].x - hexVec[j].x) + (hexVec[i].y - hexVec[j].y)*(hexVec[i].y - hexVec[j].y));
				if (abs(dist - st_dst[0]) < dist_T || abs(dist - st_dst[1]) < dist_T || abs(dist - st_dst[2]) < dist_T)
					continue;
				votes[j]++;
			}
		}
	}
	int convex_cnt = 0;
	double min_votes = 100;
	vector<Point2f> three_convex;
	while (convex_cnt<3 && votes.size() > 0)
	{
		int min_index = 0;
		for (int i= 0; i < votes.size(); i++)
		{
			if (votes[i] < min_votes)
			{
				min_votes = votes[i];
				min_index = i;
			}
		}
		votes[min_index] = 200;
		three_convex.push_back(hexVec[min_index]);
		convex_cnt++;
		min_votes = 100;
	}

	Mat threePoints(drawcorner);
	for (int i = 0; i < three_convex.size(); i++)
	{
		circle(threePoints, Point(three_convex[i].x, three_convex[i].y), 5, Scalar(255), 1);
		line(threePoints, Point(three_convex[i % 3].x, three_convex[i % 3].y), Point(three_convex[(i + 1) % 3].x, three_convex[(i + 1) % 3].y), Scalar(0), 1);
		
	}

	imshow("3Points", threePoints);
	imwrite("pic/three_corners.jpg", threePoints);
	waitKey();

	vector<Point2f> trLines;
	double k1 = -1.0 / ((three_convex[0].y - three_convex[1].y) / (three_convex[0].x - three_convex[1].x));
	double b1 = 0.5*(three_convex[0].y + three_convex[1].y) - k1*(0.5*(three_convex[0].x + three_convex[1].x));
	double k2 = -1.0 / ((three_convex[0].y - three_convex[2].y) / (three_convex[0].x - three_convex[2].x));
	double b2 = 0.5*(three_convex[0].y + three_convex[2].y) - k2*(0.5*(three_convex[0].x + three_convex[2].x));
	double k3 = -1.0 / ((three_convex[1].y - three_convex[2].y) / (three_convex[1].x - three_convex[2].x));
	double b3 = 0.5*(three_convex[1].y + three_convex[2].y) - k3*(0.5*(three_convex[1].x + three_convex[2].x));

 	double cross1_x = (b2 - b1) / (k1 - k2);
	double cross1_y = k1 * cross1_x + b1;

	double cross2_x = (b3 - b1) / (k1 - k3);
	double cross2_y = k1 * cross2_x + b1;
	//cross2_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross2_y = k1 * cross2_x + b1;

	double cross3_x = (b2 - b3) / (k3 - k2);
	double cross3_y = k2 * cross3_x + b2;
	//cross3_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross3_y = k2*cross2_x + b2;

	trLines.push_back(Point2f(cross1_x, cross1_y));
	trLines.push_back(Point2f(cross2_x, cross2_y));
	trLines.push_back(Point2f(cross3_x, cross3_y));
	double centX = (cross1_x + cross2_x + cross3_x) / 3.0;
	double centY = (cross1_y + cross2_y + cross3_y) / 3.0;

	double err = sqrt((centX - center.x)*(centX - center.x) + (centY - center.y)*(centY - center.y));
	double edgeLen = sqrt((three_convex[0].x - three_convex[1].x)*(three_convex[0].x - three_convex[1].x) 
		+ (three_convex[0].y - three_convex[1].y)*(three_convex[0].y - three_convex[1].y));
	cout << "the distance between circle center and hexagon center is " << err << endl;
	cout << "hexagon edge length is " << edgeLen << endl;
	double err_r = err / edgeLen;
	cout << "the relative error is " << err_r << endl;
	//circle(threePoints, Point(cross1_x, cross1_y), 1, Scalar(0), 1);
	//circle(threePoints, Point(cross2_x, cross2_y), 1, Scalar(0), 1);
	//circle(threePoints, Point(cross3_x, cross3_y), 1, Scalar(0), 1);
	line(threePoints, Point(cross2_x, centY - 80), Point(cross2_x, centY + 50), Scalar(0));
	line(threePoints, Point(centX - 50, k1*(centX-50)+b1), Point(centX + 50, k1*(centX + 50) + b1), Scalar(0));
	line(threePoints, Point(centX - 50, k2*(centX - 50) + b2), Point(centX + 50, k2*(centX + 50) + b2), Scalar(0));
	circle(threePoints, Point(centX, centY), 2, Scalar(255), -1);


	//namedWindow("finalCenter", 0);
	imshow("finalCenter", threePoints);
	imwrite("pic/finalCenter.jpg", threePoints);
	waitKey();

	//Mat lm2 = NonCrf_fastener(lm1, 2, 0.1, 0.2);
	//foo();
	//NonCRF(lmImg, gtImg1);
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

Mat syntheticFastener(double c, double radius, double radius_hex)
{
	Mat fastener(500, 500, CV_8U, Scalar(255));
	int centx = 250;
	int centy = 250;
	circle(fastener, Point(centx, centy), radius, Scalar(100), -1);
	
	Point hex[6];
	double rad = 60.0 / 180.0 * CV_PI;
	double dx = radius_hex * cos(rad);
	double dy = radius_hex * sin(rad);
	hex[0].x = centx + dx;
	hex[0].y = centy - dy;
	hex[1].x = centx - dx;
	hex[1].y = centy - dy;
	hex[2].x = centx - radius_hex;
	hex[2].y = centy;
	hex[3].x = centx - dx;
	hex[3].y = centy + dy;
	hex[4].x = centx + dx;
	hex[4].y = centy + dy;
	hex[5].x = centx + radius_hex;
	hex[5].y = centy;
	/*for (int i = 0; i < 5; i++)
	{
		line(fastener, hex[i], hex[i + 1], Scalar(255));
	}
	line(fastener, hex[5], hex[0], Scalar(255));*/
	fillConvexPoly(fastener, hex, 6, Scalar(200));
	fastener.forEach<uchar>([c](uchar& p, const int* pos)
		->void {double rd = rand() % 1000 * 0.001; if (rd < c) p = rand() % 255; });
	//=========draw the cross in the center
	//line(fastener, Point(centx - 50, centy), Point(centx + 50, centy), Scalar(255));
	//line(fastener, Point(centx, centy-50), Point(centx, centy+50), Scalar(255));


	return fastener;
}

//合成噪声图像，用于西交学报演示R-NCRF原理
Mat lineInNoise(int width, int cells, double rate)
{
	//double rate = 0.5;//一个格子中线段长度占格子长度的比
	Mat fastener(width, width, CV_8U, Scalar(0));
	int centx = width/2;
	int centy = width/2;
	int stride = width / cells;
	double half_len = rate*stride*0.5;
	for (int i = 0; i < cells; i++)
	{
		for (int j = 0; j < cells; j++)
		{
			int r = stride / 2 + i*stride;
			int c = stride / 2 + j*stride;
			int theta = rand() % 360 - 180;
			double rd = theta*CV_PI / 180.0;
			if (i == cells*3 / 4) rd = 0;
			int r1 = r + half_len*sin(rd);
			int c1 = c + half_len*cos(rd);
			int r2 = r - half_len*sin(rd);
			int c2 = c - half_len*cos(rd);
			line(fastener, Point(r, c), Point(r1, c1), Scalar(255),1);
			line(fastener, Point(r, c), Point(r2, c2), Scalar(255),1);
		}
	}
	return fastener;
}
Mat lineFilter(size_t size, double sigma1, double sigma2)
{
	Mat filter(size, size, CV_32FC1);
	float s1 = 2 * sigma1*sigma1;
	float s2 = 2 * sigma2*sigma2;
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int x = i - size / 2, y = j - size / 2;
			float dog = exp(-(x*x + y*y) / s1) / (CV_PI*s1) - exp(-(x*x + y*y) / s2) / (CV_PI*s2);
			if (dog < 0) dog = 0;
			filter.ptr<float>(i)[j] = dog;
		}
	}
	filter.forEach<float>([size](float& p, const int* pos)->void {if (pos[1] != size / 2) p *= -1; });
	return filter;
}
Mat myFilter2D(const Mat src, const Mat ker)
{
	Mat dst(src.size(), src.type());
	int half_sz = ker.rows / 2;
	float sums = 0;
	for (int i = 0; i < ker.rows; i++)
	{
		for (int j = 0; j < ker.cols; j++)
		{
			sums += ker.at<float>(i, j);
		}
	}
	for (int r = half_sz; r < src.rows - half_sz; r++)
	{
		for (int c = half_sz; c < src.cols - half_sz; c++)
		{
			float sumval = 0;
			for (int i = 0; i < ker.rows; i++)
			{
				for (int j = 0; j < ker.cols; j++)
				{
					sumval += src.at<float>(r + i - half_sz, c + j - half_sz)*ker.at<float>(i, j);
				}
			}
			dst.at<float>(r, c) = sumval/abs(sums);
		}
	}
	Mat ret = dst(Rect(half_sz, half_sz, src.cols - ker.cols, src.rows - ker.rows));
	return ret;
}

Mat syntheticFastenerContour(double c, double radius, double radius_hex)
{
	Mat fastener(500, 500, CV_8U, Scalar(255));
	int centx = 250;
	int centy = 250;
	circle(fastener, Point(centx, centy), radius, Scalar(100), 1);

	Point hex[6];
	double rad = 60.0 / 180.0 * CV_PI;
	double dx = radius_hex * cos(rad);
	double dy = radius_hex * sin(rad);
	hex[0].x = centx + dx;
	hex[0].y = centy - dy;
	hex[1].x = centx - dx;
	hex[1].y = centy - dy;
	hex[2].x = centx - radius_hex;
	hex[2].y = centy;
	hex[3].x = centx - dx;
	hex[3].y = centy + dy;
	hex[4].x = centx + dx;
	hex[4].y = centy + dy;
	hex[5].x = centx + radius_hex;
	hex[5].y = centy;
	for (int i = 0; i < 5; i++)
	{
		line(fastener, hex[i], hex[i + 1], Scalar(0));
	}
	line(fastener, hex[5], hex[0], Scalar(0));
	//fillConvexPoly(fastener, hex, 6, Scalar(200));
	fastener.forEach<uchar>([c](uchar& p, const int* pos)
		->void {double rd = rand() % 1000 * 0.001; if (rd < c) p = rand() % 255; });
	return fastener;
}
void MHL(Mat cntImg, Mat srcImg)
{
#if 0
	//------LSD line segment detector
	lm1.convertTo(lm1, CV_8U, 255);
	Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_STD);
	//Ptr<LineSegmentDetector> ls = createLineSegmentDetector(LSD_REFINE_NONE);
	vector<Vec4f> lines_std;
	// Detect the lines
	ls->detect(lm1, lines_std);
	vector<double> lines_len;
	for (auto i : lines_std)
		lines_len.push_back((i[0] - i[2])*(i[0] - i[2]) + (i[1] - i[3])*(i[1] - i[3]));
	sort(lines_len.begin(), lines_len.end());
	double mantain_rate = 0.35;//线段中保留比例
	vector<Vec4f> lines_new;
	for (auto i : lines_std)
	{
		if (((i[0] - i[2])*(i[0] - i[2]) + (i[1] - i[3])*(i[1] - i[3])) > lines_len.at(lines_len.size() * (1 - mantain_rate)))
			lines_new.push_back(i);
	}

	Mat drawnLines(lm1);
	ls->drawSegments(drawnLines, lines_new);
	imshow("Standard refinement", drawnLines);
	waitKey(0);
	//----------find hexagon by angle difference
	//----------if a segment is of hexagon, the different of angle to other edges is 0, 60, 120
	vector<double> lines_k;
	for (auto i : lines_new)
	{
		double slope = atan((i[1] - i[3]) / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;
	const double T = 5.0 / 180.0 * CV_PI;
	for (int i = 0; i < lines_new.size(); i++)
	{
		int cnt = 0;
		for (int j = 0; j < lines_new.size(); j++)
		{
			if (j == i) continue;
			if (abs(abs(lines_k[i] - lines_k[j] - 1.0 * CV_PI / 3.0)) < T
				|| abs(abs(lines_k[i] - lines_k[j] - 2.0 * CV_PI / 3.0)) < T)
			{
				cnt++;
			}
		}
		double len = sqrt((lines_new[i][0] - lines_new[i][2])*(lines_new[i][0] - lines_new[i][2]) + (lines_new[i][1] - lines_new[i][3])*(lines_new[i][1] - lines_new[i][3]));
		if (cnt >= 3)
			hexagon.push_back(lines_new[i]);
	}

	Mat drawnLines2(lm1);
	ls->drawSegments(drawnLines2, hexagon);
	namedWindow("Standard", 0);
	imshow("Standard", drawnLines2);
	waitKey(0);
#endif
	//---------HoughLine detector
	cntImg.convertTo(cntImg, CV_8U, 255);
	imwrite("pic/ncrf_0.jpg", cntImg);;

	//-------------霍夫圆检测------------
	//lm1.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {if (p == 0) p = 255;
	//	else p = 0; });
	Mat dst_lines(cntImg.size(), CV_8UC3);
	vector<Vec3f> circles2;
	HoughCircles(cntImg, circles2, HOUGH_GRADIENT,
		resolution, cntImg.rows / 4, canny_t, circle_t, rmin, rmax);
	Point center0(cvRound(circles2[0][0]), cvRound(circles2[0][1]));
	int radius0 = cvRound(circles2[0][2]);
	Mat draw_circle = cntImg.clone();
	for (size_t i = 0; i < min((int)circles2.size(), 1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(draw_circle, center, 3, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(draw_circle, center, radius, Scalar(0), 3, 8, 0);
	}
	imshow("hough circle2", draw_circle);
	imwrite("pic/ncrfcircle_0.jpg", draw_circle);
	waitKey();

	//----------霍夫概率直线检测--------
	//cntImg.forEach<uchar>([](uchar& p, const int* pos)
	//	->void {if (p > 0)p = 0; else p = 255; });
	vector<Vec4i> lines;
	//Mat barImg = cntImg(Rect(center0.x - radius0, center0.y -radius0, 2 * radius0, 2 * radius0));
	//imshow("barImg", barImg); waitKey();
	bitwise_not(cntImg, cntImg);
	imshow("cntImg", cntImg); waitKey();
	//HoughLinesP(cntImg, lines, 1, 3 * CV_PI / 180, 10, 20, 5);//----useful parameters
	//HoughLinesP(cntImg, lines, 1, 3 * CV_PI / 180, 5, 20, 10);//----for dst3
	//HoughLinesP(cntImg, lines, 3, 3 * CV_PI / 180, 15, 20, 8);//--useful parameters
	HoughLinesP(cntImg, lines, 3, 3 * CV_PI / 180, 15, 20, 8);

	//HoughLinesP(barImg, lines, 1, 3 * CV_PI / 180, 20, 20, 10);
	/*for (auto& i : lines)
	{
		i[0] += center0.x;
		i[1] += center0.y;
		i[2] += center0.x;
		i[3] += center0.y;

	}*/

	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	line(dst_lines, Point(lines[i][0], lines[i][1]),
	//		Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 1, 8);
	//}
	////namedWindow("houghLine", 0);
	//imshow("houghLine", dst_lines);
	//waitKey();

	//----------find hexagon by angle difference
	//----------if a segment is of hexagon, the different of angle to other edges is 0, 60, 120
	Mat hexMat(cntImg.size(), CV_8U, Scalar(255));
	vector<double> lines_k;//保存所有直线斜率
	for (auto i : lines)
	{
		double slope = atan((i[1] - i[3]) * 1.0 / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;

	const double T = 5.0 / 180.0 * CV_PI;
	//double len = 20.0;
	double len = 0.84*radius0;//通过测量得出的六角边与圆半径比值
	len = 0.80 *radius0;
	double st_dst[3] = { len, 1.732*len, 2.0*len };
	double dist_T = 3;

	Point center(cvRound(circles2[0][0]), cvRound(circles2[0][1]));
	int radius = cvRound(circles2[0][2]);
	for (int i = 0; i < lines.size(); i++)
	{
		int cnt = 0;
		for (int j = 0; j < lines.size(); j++)
		{
			if (j == i) continue;
			if (abs(abs(abs(lines_k[i]) - abs(lines_k[j])) - 1.0 * CV_PI / 3.0) < T
				|| abs(abs(abs(lines_k[i]) - abs(lines_k[j])) - 2.0 * CV_PI / 3.0) < T)
			{
				cnt++;
			}
		}
		if (cnt >= 2)// && len < 100)//一条线段，至少与其他两条线段满足角度约束才认为是候选线段
		{
			if (sqrt((lines[i][0] - center.x)*(lines[i][0] - center.x) + (lines[i][1] - center.y)*(lines[i][1] - center.y)) >= radius - 3
				|| sqrt((lines[i][0] - center.x)*(lines[i][0] - center.x) + (lines[i][1] - center.y)*(lines[i][1] - center.y)) < radius*1.0 / 3.0
				|| sqrt((lines[i][2] - center.x)*(lines[i][2] - center.x) + (lines[i][3] - center.y)*(lines[i][3] - center.y)) >= radius - 3
				|| sqrt((lines[i][2] - center.x)*(lines[i][2] - center.x) + (lines[i][3] - center.y)*(lines[i][3] - center.y)) < radius*1.0 / 3.0)
				continue;
			hexagon.push_back(lines[i]);//候选边
			line(hexMat, Point2f(lines[i][0], lines[i][1]), Point2f(lines[i][2], lines[i][3]), Scalar(0));//画出圆内候选边
		}
	}
	//画出
	for (size_t i = 0; i < min((int)circles2.size(), 1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(hexMat, center, 2, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(hexMat, center, radius, Scalar(0), 2, 8, 0);
	}
	imshow("Standard", hexMat);
	waitKey(0);
	imwrite("pic/circleHex.jpg", hexMat);

	Mat drawcorner(hexMat.size(), hexMat.type(), Scalar(255));
	//Mat drawcorner = hexMat.clone();
	cntImg.forEach<uchar>([](uchar& p, const int* pos)
		->void {if (p > 0)p = 0; else p = 255; });
	drawcorner = hexMat.clone();

	for (size_t i = 0; i < min((int)circles2.size(), 1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(drawcorner, center, 3, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(drawcorner, center, radius, Scalar(0), 2, 8, 0);
	}
	//-----------选出候选角点-------------
	vector<Point2f> hex_corners;
	vector<double> k_value;//候选角点对应的角平分线斜率
	for (int i = 0; i < hexagon.size(); i++)
	{
		double slope1 = atan((hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]));
		for (int j = i+1; j < hexagon.size(); j++)
		{
			//if (j == i) continue;
			//line(drawcorner, Point(hexagon[i][0], hexagon[i][1]), Point(hexagon[i][2], hexagon[i][3]), Scalar(0, 255, 0));
			line(drawcorner, Point(hexagon[j][0], hexagon[j][1]), Point(hexagon[j][2], hexagon[j][3]), Scalar(0));
			double k1, k2, b1, b2;
			k1 = (hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]);
			k2 = (hexagon[j][1] - hexagon[j][3]) * 1.0 / (hexagon[j][0] - hexagon[j][2]);
			b1 = hexagon[i][1] - k1 * hexagon[i][0];
			b2 = hexagon[j][1] - k2 * hexagon[j][0];
			double cross_x = (b2 - b1) / (k1 - k2);
			double cross_y = k2 * cross_x + b2;
			//hex_corners.insert(segdist, Point2f(cross_x, cross_y));
			if (sqrt((cross_x - center.x)*(cross_x - center.x) + (cross_y - center.y)*(cross_y - center.y)) < radius - 5
				&& sqrt((cross_x - center.x)*(cross_x - center.x) + (cross_y - center.y)*(cross_y - center.y)) > radius*3 / 4.0)
			{
				hex_corners.push_back(Point2f(cross_x, cross_y));//六角顶点
				double k0 = 0.5*(k1 + k2);
				if (cross_x < center0.x && k0 < 0)
					k0 = -1.0 / k0;
				k_value.push_back(k0);
			}
		}
	}
	//Mat finalRst = drawcorner.clone();
	for (auto it = hex_corners.begin(); it != hex_corners.end(); ++it)//画出所有候选角点
	{
		circle(drawcorner, Point(it->x, it->y), 5, Scalar(0), 1);
	}
	imshow("hex_corner", drawcorner);
	imwrite("pic/hexcorner.jpg", drawcorner);
	waitKey();

	//-----------------distance constrain to pick out 3 convex points	
	vector<Point2f> hexVec;
	for (auto it = hex_corners.begin(); it != hex_corners.end(); ++it)
	{
		hexVec.push_back(Point2f(it->x, it->y));
	}
	vector<int> votes(hexVec.size(), 0);
	for (int i = 0; i < hexVec.size(); i++)
	{
		for (int j = 0; j < hexVec.size(); j++)
		{
			if (i != j)
			{
				double dist = sqrt((hexVec[i].x - hexVec[j].x)*(hexVec[i].x - hexVec[j].x) + (hexVec[i].y - hexVec[j].y)*(hexVec[i].y - hexVec[j].y));
				if (abs(dist - st_dst[0]) < dist_T || abs(dist - st_dst[1]) < dist_T || abs(dist - st_dst[2]) < dist_T)
					continue;
				votes[j]++;
			}
		}
	}
	int convex_cnt = 0;
	double min_votes = 200;
	vector<Point2f> three_convex;
	vector<int> vex_ind;
	while (convex_cnt < 3 && votes.size() > 0)
	{
		int min_index = 0;
		min_votes = 200;
		for (int i = 0; i < votes.size(); i++)
		{
			if (votes[i] < min_votes)
			{
				min_votes = votes[i];
				min_index = i;
			}
		}
		votes[min_index] = 200;//票数置位无穷大
		three_convex.push_back(hexVec[min_index]);
		vex_ind.push_back(min_index);
		convex_cnt++;
		min_votes = 100;
	}

	Mat threePoints(srcImg);
	for (size_t i = 0; i < min((int)circles2.size(), 1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(threePoints, center, 2, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(threePoints, center, radius, Scalar(0), 2, 8, 0);
	}

	for (int i = 0; i < hexVec.size(); i++)
	{
		circle(threePoints, Point(three_convex[i].x, three_convex[i].y), 5, Scalar(255), 1);
		line(threePoints, Point(three_convex[i % 3].x, three_convex[i % 3].y), Point(three_convex[(i + 1) % 3].x, three_convex[(i + 1) % 3].y), Scalar(0), 1);
		//画出角平分线,实验表明，此方法不稳定，三个角点之间误差太大
		/*Point p1(three_convex[i].x, three_convex[i].y);
		Point p2;
		p2.x = three_convex[i].x + 2 * (center.x - three_convex[i].x);
		p2.y = three_convex[i].y + (p2.x-p1.x) * k_value[vex_ind[i]];
		line(threePoints, p1, p2, Scalar(0), 1);*/
	}

	imshow("3Points", threePoints);
	imwrite("pic/three_corners.jpg", threePoints);
	waitKey();

	vector<Point2f> trLines;
	double k1 = -1.0 / ((three_convex[0].y - three_convex[1].y) / (three_convex[0].x - three_convex[1].x));
	double b1 = 0.5*(three_convex[0].y + three_convex[1].y) - k1*(0.5*(three_convex[0].x + three_convex[1].x));
	double k2 = -1.0 / ((three_convex[0].y - three_convex[2].y) / (three_convex[0].x - three_convex[2].x));
	double b2 = 0.5*(three_convex[0].y + three_convex[2].y) - k2*(0.5*(three_convex[0].x + three_convex[2].x));
	double k3 = -1.0 / ((three_convex[1].y - three_convex[2].y) / (three_convex[1].x - three_convex[2].x));
	double b3 = 0.5*(three_convex[1].y + three_convex[2].y) - k3*(0.5*(three_convex[1].x + three_convex[2].x));

	double cross1_x = (b2 - b1) / (k1 - k2);
	double cross1_y = k1 * cross1_x + b1;

	double cross2_x = (b3 - b1) / (k1 - k3);
	double cross2_y = k1 * cross2_x + b1;
	//cross2_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross2_y = k1 * cross2_x + b1;

	double cross3_x = (b2 - b3) / (k3 - k2);
	double cross3_y = k2 * cross3_x + b2;
	//cross3_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross3_y = k2*cross2_x + b2;

	trLines.push_back(Point2f(cross1_x, cross1_y));
	trLines.push_back(Point2f(cross2_x, cross2_y));
	trLines.push_back(Point2f(cross3_x, cross3_y));
	double centX = (cross1_x + cross2_x + cross3_x) / 3.0;
	double centY = (cross1_y + cross2_y + cross3_y) / 3.0;

	double err = sqrt((centX - center.x)*(centX - center.x) + (centY - center.y)*(centY - center.y));
	double edgeLen = sqrt((three_convex[0].x - three_convex[1].x)*(three_convex[0].x - three_convex[1].x)
		+ (three_convex[0].y - three_convex[1].y)*(three_convex[0].y - three_convex[1].y));
	cout << "the distance between circle center and hexagon center is " << err << endl;
	cout << "hexagon edge length is " << edgeLen << endl;
	double err_r = err / edgeLen;
	cout << "the relative error is " << err_r << endl;
	//circle(threePoints, Point(cross1_x, cross1_y), 1, Scalar(0), 1);
	//circle(threePoints, Point(cross2_x, cross2_y), 1, Scalar(0), 1);
	//circle(threePoints, Point(cross3_x, cross3_y), 1, Scalar(0), 1);
	//line(threePoints, Point(cross2_x, centY - 80), Point(cross2_x, centY + 50), Scalar(0));
	line(threePoints, Point(centX - 50, k1*(centX - 50) + b1), Point(centX + 50, k1*(centX + 50) + b1), Scalar(0));
	line(threePoints, Point(centX - 50, k2*(centX - 50) + b2), Point(centX + 50, k2*(centX + 50) + b2), Scalar(0));
	line(threePoints, Point(centX - 50, k3*(centX - 50) + b3), Point(centX + 50, k3*(centX + 50) + b3), Scalar(0));

	circle(threePoints, Point(centX, centY), 2, Scalar(255), -1);


	//namedWindow("finalCenter", 0);
	imshow("finalCenter", threePoints);
	imwrite("pic/finalCenter.jpg", threePoints);
	waitKey();
}
