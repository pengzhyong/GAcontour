#include <iostream>
#include <ctime>
#include "postProcess.h"
#include "GA.h"
#include "NCRF.h"
#include "bendGabar.h"
#include "circleFilter.h"
#include "myCanny.h"
using namespace std;

float resolution = 1;
float canny_t = 80;
float circle_t = 5;
float rmin = 50;
float rmax = 60;
Mat syntheticFastener(float c, float radius = 100, float radius_hex = 70);
Mat syntheticFastenerContour(float c, float radius = 100, float radius_hex = 70);
void MHL(Mat cntImg, Mat srcImg);

int main()
{
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

	Mat src = imread("C:\\Users\\pzy\\Desktop\\小论文\\螺母\\分类\\背景\\01_看图王.jpg");
	//Mat src = imread("C:\\Users\\pzy\\Desktop\\小论文\\螺母\\lena.jpg");
	cvtColor(src, src, COLOR_BGR2GRAY);
	pyrDown(src, src);
	
	Mat s1 = src.clone();
	Mat dst = myspace::Mycanny(s1, 0.1, 0.3);
	imshow("myCanny result", dst); waitKey();
	Mat canImg;
	Canny(src, canImg, 80, 240);
	imshow("canImg", canImg); waitKey(0);

	drawRNCRF();

	//drawButterflyNCRF();
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

	srcImg1.convertTo(srcImg1, CV_32F, 1 / 255.0);
	srcImg2.convertTo(srcImg2, CV_32F, 1 / 255.0);
	srcImg3.convertTo(srcImg3, CV_32F, 1 / 255.0);

	gtImg1.convertTo(gtImg1, CV_32F, 1 / 255.0);
	gtImg2.convertTo(gtImg2, CV_32F, 1 / 255.0);
	gtImg3.convertTo(gtImg3, CV_32F, 1 / 255.0);

	//Mat circleImg(srcImg3.size(), srcImg3.type(), Scalar(0));
	//circle(circleImg, Point2f(srcImg3.rows / 2, srcImg3.cols / 2), 100, Scalar(1));
	///*circleImg.forEach<float>([](float& p, const int* pos)
	//	->void {float rd = rand() % 2; if (rd < 0.001) p = 1.0; });*/
	//for (int r = 0; r < circleImg.rows; r++)
	//{  
	//	for (int c = 0; c < circleImg.cols; c++)
	//	{
	//		float rd = rand() % 100 * 0.01;
	//		if (rd < 0.3)
	//			circleImg.at<float>(r, c) = 1.0;
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
	//	float rho = lines0[i][0];
	//	float theta = lines0[i][1];
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
	lmImg.convertTo(lmImg, CV_32F, 1 / 255.0);
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
	vector<float> lines_len;
	for (auto i : lines_std)
		lines_len.push_back((i[0] - i[2])*(i[0] - i[2]) + (i[1] - i[3])*(i[1] - i[3]));
	sort(lines_len.begin(), lines_len.end());
	float mantain_rate = 0.35;//线段中保留比例
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
	vector<float> lines_k;
	for (auto i : lines_new)
	{
		float slope = atan((i[1] - i[3]) / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;
	const float T = 5.0 / 180.0 * CV_PI;
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
		float len = sqrt((lines_new[i][0] - lines_new[i][2])*(lines_new[i][0] - lines_new[i][2]) + (lines_new[i][1] - lines_new[i][3])*(lines_new[i][1] - lines_new[i][3]));
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
		float rho = lines[i][0];
		float theta = lines[i][1];
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
	vector<float> lines_k;
	for (auto i : lines)
	{

		float slope = atan((i[1] - i[3]) * 1.0 / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;
	const float T = 15.0 / 180.0 * CV_PI;

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
		float len = sqrt((lines[i][0] - lines[i][2])*(lines[i][0] - lines[i][2]) + (lines[i][1] - lines[i][3])*(lines[i][1] - lines[i][3]));
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
	map<float, Point2f> hex_corners;
	const float segdistT = 40;//线段距离阈值，低于此阈值才被认为是真正的六角螺母边
	for (int i = 0; i < hexagon.size(); i++)
	{
		float slope1 = atan((hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]));
		for (int j = 0; j < hexagon.size(); j++)
		{
			if (j == i) continue;
			float slope2 = atan((hexagon[j][1] - hexagon[j][3]) * 1.0 / (hexagon[j][0] - hexagon[j][2]));
			if (abs(abs(slope2 - slope1) - 1.0 * CV_PI / 3.0) < T 
				|| abs(abs(slope2 - slope1) - 2.0 * CV_PI / 3.0) < T)
			{
				float dist1, dist2, dist3, dist4;
				dist1 = sqrt((hexagon[i][0] - hexagon[j][0])*(hexagon[i][0] - hexagon[j][0])
					+ (hexagon[i][1] - hexagon[j][1])*(hexagon[i][1] - hexagon[j][1]));
				dist2 = sqrt((hexagon[i][0] - hexagon[j][2])*(hexagon[i][0] - hexagon[j][2])
					+ (hexagon[i][1] - hexagon[j][3])*(hexagon[i][1] - hexagon[j][3]));
				dist3 = sqrt((hexagon[i][2] - hexagon[j][0])*(hexagon[i][2] - hexagon[j][0])
					+ (hexagon[i][3] - hexagon[j][1])*(hexagon[i][3] - hexagon[j][1]));
				dist4 = sqrt((hexagon[i][2] - hexagon[j][2])*(hexagon[i][2] - hexagon[j][2])
					+ (hexagon[i][3] - hexagon[j][3])*(hexagon[i][3] - hexagon[j][3]));
				float segdist = min(min(dist1, dist2), min(dist3, dist4));//定义线段之间的距离为两线段端点之间的最短距离
				
  				if (1)//segdist < segdistT)
				{
					//line(drawcorner, Point(hexagon[i][0], hexagon[i][1]), Point(hexagon[i][2], hexagon[i][3]), Scalar(0, 255, 0));
					line(drawcorner, Point(hexagon[j][0], hexagon[j][1]), Point(hexagon[j][2], hexagon[j][3]), Scalar(0));

					float k1, k2, b1, b2;
					k1 = (hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]);
					k2 = (hexagon[j][1] - hexagon[j][3]) * 1.0 / (hexagon[j][0] - hexagon[j][2]);
					b1 = hexagon[i][1] - k1 * hexagon[i][0];
					b2 = hexagon[j][1] - k2 * hexagon[j][0];
					float cross_x = (b2 - b1) / (k1 - k2);
					float cross_y = k2 * cross_x + b2;
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
		circle(finalRst, Point(it->second.x, it->second.y), 3, Scalar(255),1);
	}
	imshow("hex_corner", finalRst);
	imwrite("pic/hexcorner.jpg", finalRst);
	waitKey();

	//-----------------distance constrain to pick out 3 convex points
	float len = 20.0;
	len = (rmax + rmin) * 0.909 / 2.0;
	float st_dst[3] = { len, 1.732*len, 2.0*len };
	float dist_T =3;
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
				float dist = sqrt((hexVec[i].x - hexVec[j].x)*(hexVec[i].x - hexVec[j].x) + (hexVec[i].y - hexVec[j].y)*(hexVec[i].y - hexVec[j].y));
				if (abs(dist - st_dst[0]) < dist_T || abs(dist - st_dst[1]) < dist_T || abs(dist - st_dst[2]) < dist_T)
					continue;
				votes[j]++;
			}
		}
	}
	int convex_cnt = 0;
	float min_votes = 100;
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
		circle(threePoints, Point(three_convex[i].x, three_convex[i].y), 3, Scalar(255), 1);
		line(threePoints, Point(three_convex[i % 3].x, three_convex[i % 3].y), Point(three_convex[(i + 1) % 3].x, three_convex[(i + 1) % 3].y), Scalar(0), 1);
		
	}

	imshow("3Points", threePoints);
	imwrite("pic/three_corners.jpg", threePoints);
	waitKey();

	vector<Point2f> trLines;
	float k1 = -1.0 / ((three_convex[0].y - three_convex[1].y) / (three_convex[0].x - three_convex[1].x));
	float b1 = 0.5*(three_convex[0].y + three_convex[1].y) - k1*(0.5*(three_convex[0].x + three_convex[1].x));
	float k2 = -1.0 / ((three_convex[0].y - three_convex[2].y) / (three_convex[0].x - three_convex[2].x));
	float b2 = 0.5*(three_convex[0].y + three_convex[2].y) - k2*(0.5*(three_convex[0].x + three_convex[2].x));
	float k3 = -1.0 / ((three_convex[1].y - three_convex[2].y) / (three_convex[1].x - three_convex[2].x));
	float b3 = 0.5*(three_convex[1].y + three_convex[2].y) - k3*(0.5*(three_convex[1].x + three_convex[2].x));

 	float cross1_x = (b2 - b1) / (k1 - k2);
	float cross1_y = k1 * cross1_x + b1;

	float cross2_x = (b3 - b1) / (k1 - k3);
	float cross2_y = k1 * cross2_x + b1;
	//cross2_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross2_y = k1 * cross2_x + b1;

	float cross3_x = (b2 - b3) / (k3 - k2);
	float cross3_y = k2 * cross3_x + b2;
	//cross3_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross3_y = k2*cross2_x + b2;

	trLines.push_back(Point2f(cross1_x, cross1_y));
	trLines.push_back(Point2f(cross2_x, cross2_y));
	trLines.push_back(Point2f(cross3_x, cross3_y));
	float centX = (cross1_x + cross2_x + cross3_x) / 3.0;
	float centY = (cross1_y + cross2_y + cross3_y) / 3.0;

	float err = sqrt((centX - center.x)*(centX - center.x) + (centY - center.y)*(centY - center.y));
	float edgeLen = sqrt((three_convex[0].x - three_convex[1].x)*(three_convex[0].x - three_convex[1].x) 
		+ (three_convex[0].y - three_convex[1].y)*(three_convex[0].y - three_convex[1].y));
	cout << "the distance between circle center and hexagon center is " << err << endl;
	cout << "hexagon edge length is " << edgeLen << endl;
	float err_r = err / edgeLen;
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

Mat syntheticFastener(float c, float radius, float radius_hex)
{
	Mat fastener(500, 500, CV_8U, Scalar(255));
	int centx = 250;
	int centy = 250;
	circle(fastener, Point(centx, centy), radius, Scalar(100), -1);
	
	Point hex[6];
	float rad = 60.0 / 180.0 * CV_PI;
	float dx = radius_hex * cos(rad);
	float dy = radius_hex * sin(rad);
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
		->void {float rd = rand() % 1000 * 0.001; if (rd < c) p = rand() % 255; });
	return fastener;
}
Mat syntheticFastenerContour(float c, float radius, float radius_hex)
{
	Mat fastener(500, 500, CV_8U, Scalar(255));
	int centx = 250;
	int centy = 250;
	circle(fastener, Point(centx, centy), radius, Scalar(100), 1);

	Point hex[6];
	float rad = 60.0 / 180.0 * CV_PI;
	float dx = radius_hex * cos(rad);
	float dy = radius_hex * sin(rad);
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
		->void {float rd = rand() % 1000 * 0.001; if (rd < c) p = rand() % 255; });
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
	vector<float> lines_len;
	for (auto i : lines_std)
		lines_len.push_back((i[0] - i[2])*(i[0] - i[2]) + (i[1] - i[3])*(i[1] - i[3]));
	sort(lines_len.begin(), lines_len.end());
	float mantain_rate = 0.35;//线段中保留比例
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
	vector<float> lines_k;
	for (auto i : lines_new)
	{
		float slope = atan((i[1] - i[3]) / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;
	const float T = 5.0 / 180.0 * CV_PI;
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
		float len = sqrt((lines_new[i][0] - lines_new[i][2])*(lines_new[i][0] - lines_new[i][2]) + (lines_new[i][1] - lines_new[i][3])*(lines_new[i][1] - lines_new[i][3]));
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
	for (size_t i = 0; i < min((int)circles2.size(), 1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(srcImg, center, 3, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(srcImg, center, radius, Scalar(0), 3, 8, 0);
	}
	imshow("hough circle2", srcImg);
	imwrite("pic/ncrfcircle_0.jpg", srcImg);;
	waitKey();

	//----------霍夫概率直线检测--------
	cntImg.forEach<uchar>([](uchar& p, const int* pos)
		->void {if (p > 0)p = 0; else p = 255; });
	vector<Vec4i> lines;
	HoughLinesP(cntImg, lines, 1, 5 * CV_PI / 180, 25, 13, 10);
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
	vector<float> lines_k;//保存所有直线斜率
	for (auto i : lines)
	{
		float slope = atan((i[1] - i[3]) * 1.0 / (i[0] - i[2]));
		lines_k.push_back(slope);
	}
	vector<Vec4f> hexagon;

	const float T = 15.0 / 180.0 * CV_PI;
	float len = 20.0;
	float st_dst[3] = { len, 1.732*len, 2.0*len };
	float dist_T = 3;

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
			if (sqrt((lines[i][0] - center.x)*(lines[i][0] - center.x) + (lines[i][1] - center.y)*(lines[i][1] - center.y)) >= radius - 5
				|| sqrt((lines[i][0] - center.x)*(lines[i][0] - center.x) + (lines[i][1] - center.y)*(lines[i][1] - center.y)) < radius*2.0 / 3.0
				|| sqrt((lines[i][2] - center.x)*(lines[i][2] - center.x) + (lines[i][3] - center.y)*(lines[i][3] - center.y)) >= radius - 5
				|| sqrt((lines[i][2] - center.x)*(lines[i][2] - center.x) + (lines[i][3] - center.y)*(lines[i][3] - center.y)) < radius*2.0 / 3.0)
				continue;
			hexagon.push_back(lines[i]);//候选角点
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

	Mat drawcorner(hexMat.size(), hexMat.type(), Scalar(0, 0, 0));
	//Mat drawcorner = hexMat.clone();
	cntImg.forEach<uchar>([](uchar& p, const int* pos)
		->void {if (p > 0)p = 0; else p = 255; });
	drawcorner = srcImg.clone();

	for (size_t i = 0; i < min((int)circles2.size(), 1); i++)
	{
		Point center(cvRound(circles2[i][0]), cvRound(circles2[i][1]));
		int radius = cvRound(circles2[i][2]);
		// draw the circle center
		circle(drawcorner, center, 2, Scalar(0), -1, 8, 0);
		// draw the circle outline
		circle(drawcorner, center, radius, Scalar(0), 2, 8, 0);
	}
	//-----------选出候选角点-------------
	vector<Point2f> hex_corners;
	for (int i = 0; i < hexagon.size(); i++)
	{
		float slope1 = atan((hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]));
		for (int j = 0; j < hexagon.size(); j++)
		{
			if (j == i) continue;
			//line(drawcorner, Point(hexagon[i][0], hexagon[i][1]), Point(hexagon[i][2], hexagon[i][3]), Scalar(0, 255, 0));
			line(drawcorner, Point(hexagon[j][0], hexagon[j][1]), Point(hexagon[j][2], hexagon[j][3]), Scalar(0));
			float k1, k2, b1, b2;
			k1 = (hexagon[i][1] - hexagon[i][3]) * 1.0 / (hexagon[i][0] - hexagon[i][2]);
			k2 = (hexagon[j][1] - hexagon[j][3]) * 1.0 / (hexagon[j][0] - hexagon[j][2]);
			b1 = hexagon[i][1] - k1 * hexagon[i][0];
			b2 = hexagon[j][1] - k2 * hexagon[j][0];
			float cross_x = (b2 - b1) / (k1 - k2);
			float cross_y = k2 * cross_x + b2;
			//hex_corners.insert(segdist, Point2f(cross_x, cross_y));
			if (sqrt((cross_x - center.x)*(cross_x - center.x) + (cross_y - center.y)*(cross_y - center.y)) < radius - 5
				&& sqrt((cross_x - center.x)*(cross_x - center.x) + (cross_y - center.y)*(cross_y - center.y)) > radius*3.22 / 4.0)
			{
				hex_corners.push_back(Point2f(cross_x, cross_y));//六角顶点
			}
		}
	}
	//Mat finalRst = drawcorner.clone();
	for (auto it = hex_corners.begin(); it != hex_corners.end(); ++it)//画出所有候选角点
	{
		circle(drawcorner, Point(it->x, it->y), 3, Scalar(255), 1);
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
				float dist = sqrt((hexVec[i].x - hexVec[j].x)*(hexVec[i].x - hexVec[j].x) + (hexVec[i].y - hexVec[j].y)*(hexVec[i].y - hexVec[j].y));
				if (abs(dist - st_dst[0]) < dist_T || abs(dist - st_dst[1]) < dist_T || abs(dist - st_dst[2]) < dist_T)
					continue;
				votes[j]++;
			}
		}
	}
	int convex_cnt = 0;
	float min_votes = 100;
	vector<Point2f> three_convex;
	while (convex_cnt < 3 && votes.size() > 0)
	{
		int min_index = 0;
		for (int i = 0; i < votes.size(); i++)
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
		circle(threePoints, Point(three_convex[i].x, three_convex[i].y), 3, Scalar(255), 1);
		line(threePoints, Point(three_convex[i % 3].x, three_convex[i % 3].y), Point(three_convex[(i + 1) % 3].x, three_convex[(i + 1) % 3].y), Scalar(0), 1);

	}

	imshow("3Points", threePoints);
	imwrite("pic/three_corners.jpg", threePoints);
	waitKey();

	vector<Point2f> trLines;
	float k1 = -1.0 / ((three_convex[0].y - three_convex[1].y) / (three_convex[0].x - three_convex[1].x));
	float b1 = 0.5*(three_convex[0].y + three_convex[1].y) - k1*(0.5*(three_convex[0].x + three_convex[1].x));
	float k2 = -1.0 / ((three_convex[0].y - three_convex[2].y) / (three_convex[0].x - three_convex[2].x));
	float b2 = 0.5*(three_convex[0].y + three_convex[2].y) - k2*(0.5*(three_convex[0].x + three_convex[2].x));
	float k3 = -1.0 / ((three_convex[1].y - three_convex[2].y) / (three_convex[1].x - three_convex[2].x));
	float b3 = 0.5*(three_convex[1].y + three_convex[2].y) - k3*(0.5*(three_convex[1].x + three_convex[2].x));

	float cross1_x = (b2 - b1) / (k1 - k2);
	float cross1_y = k1 * cross1_x + b1;

	float cross2_x = (b3 - b1) / (k1 - k3);
	float cross2_y = k1 * cross2_x + b1;
	//cross2_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross2_y = k1 * cross2_x + b1;

	float cross3_x = (b2 - b3) / (k3 - k2);
	float cross3_y = k2 * cross3_x + b2;
	//cross3_x = 0.5*(three_convex[1].x + three_convex[2].x);
	//cross3_y = k2*cross2_x + b2;

	trLines.push_back(Point2f(cross1_x, cross1_y));
	trLines.push_back(Point2f(cross2_x, cross2_y));
	trLines.push_back(Point2f(cross3_x, cross3_y));
	float centX = (cross1_x + cross2_x + cross3_x) / 3.0;
	float centY = (cross1_y + cross2_y + cross3_y) / 3.0;

	float err = sqrt((centX - center.x)*(centX - center.x) + (centY - center.y)*(centY - center.y));
	float edgeLen = sqrt((three_convex[0].x - three_convex[1].x)*(three_convex[0].x - three_convex[1].x)
		+ (three_convex[0].y - three_convex[1].y)*(three_convex[0].y - three_convex[1].y));
	cout << "the distance between circle center and hexagon center is " << err << endl;
	cout << "hexagon edge length is " << edgeLen << endl;
	float err_r = err / edgeLen;
	cout << "the relative error is " << err_r << endl;
	//circle(threePoints, Point(cross1_x, cross1_y), 1, Scalar(0), 1);
	//circle(threePoints, Point(cross2_x, cross2_y), 1, Scalar(0), 1);
	//circle(threePoints, Point(cross3_x, cross3_y), 1, Scalar(0), 1);
	line(threePoints, Point(cross2_x, centY - 80), Point(cross2_x, centY + 50), Scalar(0));
	line(threePoints, Point(centX - 50, k1*(centX - 50) + b1), Point(centX + 50, k1*(centX + 50) + b1), Scalar(0));
	line(threePoints, Point(centX - 50, k2*(centX - 50) + b2), Point(centX + 50, k2*(centX + 50) + b2), Scalar(0));
	circle(threePoints, Point(centX, centY), 2, Scalar(255), -1);


	//namedWindow("finalCenter", 0);
	imshow("finalCenter", threePoints);
	imwrite("pic/finalCenter.jpg", threePoints);
	waitKey();
}
