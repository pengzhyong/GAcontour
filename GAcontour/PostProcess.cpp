#include "postProcess.h"
#include <algorithm>

PostProgress::PostProgress()
{
}

PostProgress::~PostProgress()
{
}

void PostProgress::MySobel(Mat& src, Mat& dst, Mat& dstx, Mat& dsty)
{
	Mat SobKerx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1)/8;
	Mat SobKery = (Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1)/8;
	//src.copyTo(dst);
	//src.copyTo(dstx);
	//src.copyTo(dsty);


	for (int i = 1; i<src.rows - 1; i++)
	{
		uchar* ptr = src.ptr<uchar>(i);
		float* ptrDst = dst.ptr<float>(i);
		float* ptrDstx = dstx.ptr<float>(i);
		float* ptrDsty = dsty.ptr<float>(i);

		for (int j = 1; j<src.cols - 1; j++)
		{
			
			int average = 0;
			//乘上一个局部对比度系数，对比度越小，系数 越大
			double intenAve = 0;//局部均值
			intenAve = src.at<uchar>(i - 1, j - 1) + src.at<uchar>(i - 1, j) + src.at<uchar>(i - 1, j + 1)
				+ src.at<uchar>(i, j - 1) + src.at<uchar>(i, j) + src.at<uchar>(i + 1, j + 1)
				+ src.at<uchar>(i + 1, j - 1) + src.at<uchar>(i + 1, j) + src.at<uchar>(i + 1, j + 1);
			intenAve /= 9.0;
			double var = 0;//方差
			var = (src.at<uchar>(i - 1, j - 1) - intenAve)*(src.at<uchar>(i - 1, j - 1) - intenAve)
				+ (src.at<uchar>(i - 1, j) - intenAve)*(src.at<uchar>(i - 1, j) - intenAve)
				+ (src.at<uchar>(i - 1, j + 1) - intenAve)*(src.at<uchar>(i - 1, j + 1) - intenAve)
				+ (src.at<uchar>(i, j - 1) - intenAve)*(src.at<uchar>(i, j - 1) - intenAve)
				+ (src.at<uchar>(i, j) - intenAve)*(src.at<uchar>(i, j) - intenAve)
				+ (src.at<uchar>(i, j + 1) - intenAve)*(src.at<uchar>(i, j + 1) - intenAve)
				+ (src.at<uchar>(i + 1, j - 1) - intenAve)*(src.at<uchar>(i + 1, j - 1) - intenAve)
				+ (src.at<uchar>(i + 1, j) - intenAve)*(src.at<uchar>(i + 1, j) - intenAve)
				+ (src.at<uchar>(i + 1, j + 1) - intenAve)*(src.at<uchar>(i + 1, j + 1) - intenAve);
			var /= 9.0;
			var /= 200;
			double contrE = 1.0 / var;//系数
			double rx = 0;
			double ry = 0;
			rx = SobKerx.at<float>(0, 0)*(src.at<uchar>(i - 1, j - 1) - average) + SobKerx.at<float>(0, 1)*(src.at<uchar>(i - 1, j) - average) + SobKerx.at<float>(0, 2)*(src.at<uchar>(i - 1, j + 1) - average)
				+ SobKerx.at<float>(1, 0)*(src.at<uchar>(i, j - 1) - average) + SobKerx.at<float>(1, 1)*(src.at<uchar>(i, j) - average) + SobKerx.at<float>(1, 2)*(src.at<uchar>(i, j + 1) - average)
				+ SobKerx.at<float>(2, 0)*(src.at<uchar>(i + 1, j - 1) - average) + SobKerx.at<float>(2, 1)*(src.at<uchar>(i + 1, j) - average) + SobKerx.at<float>(2, 2)*(src.at<uchar>(i + 1, j + 1) - average);
			ry = SobKery.at<float>(0, 0)*(src.at<uchar>(i - 1, j - 1) - average) + SobKery.at<float>(0, 1)*(src.at<uchar>(i - 1, j) - average) + SobKery.at<float>(0, 2)*(src.at<uchar>(i - 1, j + 1) - average)
				+ SobKery.at<float>(1, 0)*(src.at<uchar>(i, j - 1) - average) + SobKery.at<float>(1, 1)*(src.at<uchar>(i, j) - average) + SobKery.at<float>(1, 2)*(src.at<uchar>(i, j + 1) - average)
				+ SobKery.at<float>(2, 0)*(src.at<uchar>(i + 1, j - 1) - average) + SobKery.at<float>(2, 1)*(src.at<uchar>(i + 1, j) - average) + SobKery.at<float>(2, 2)*(src.at<uchar>(i + 1, j + 1) - average);
			//rx *= contrE;
			//ry *= contrE;
			/*if (var < 0.01)
				ptrDst[j] = (sqrt(rx*rx + ry*ry)*1.0)*(1.0 + 60 * exp(-var));
			else if (var > 0.03)
				ptrDst[j] = (sqrt(rx*rx + ry*ry)*1.0)*(1.0 + 3 / (1+exp(-var)));
			else*/
			ptrDst[j] = sqrt(rx*rx + ry*ry);
			ptrDstx[j] = rx;// abs(rx);
			ptrDsty[j] = ry;// abs(ry);

			//ptrDst[j] = var;

			//std::cout << dst.at<float>(i,j) << std::endl;
		}
	}
	//std::cout << dstx << std::endl;

	//dst.convertTo(dst, CV_8U);
	//std::cout << dst << std::endl;
	//normalize(dst, dst, 0, 1, NORM_MINMAX);
	
}

void PostProgress::HTC(Mat & src, Mat & dst, int thHigh, int thLow)
{
	int th = thHigh;
	int tl = thLow;
	
	dst = src.clone();
	//dst.convertTo(dst, CV_8U);
	imwrite("mysob.jpg", dst);
	waitKey(0);

	for (int r = 0; r < dst.rows; r++)
	{
		for (int c = 0; c < dst.cols; c++)
		{
			if (dst.at<uchar>(r, c) >= th)
			{
				dst.at<uchar>(r, c) = 255;
			}
			if (dst.at<uchar>(r, c) < tl)
			{
				dst.at<uchar>(r, c) = 0;
			}
		}
	}
	Mat visitedMat(dst.size(), dst.type(), Scalar(0));

	for (int i = 0; i < dst.rows; i++)
	{
		for (int j = 0; j < dst.cols; j++)
		{
			if (dst.at<uchar>(i, j) >= th && visitedMat.at<uchar>(i, j) == 0)
			{

				int flag = 0;
				DFS(Point(i, j), dst, visitedMat, th, &flag);
			}
		}
	}
	threshold(dst, dst, th - 1, 255, THRESH_BINARY);
	imshow("dst1", dst);
	waitKey(0);
}

void PostProgress::DFS(const Point point, Mat & img, Mat & visited, int th, int * flag)
{
	if (point.x<0 || point.x>img.rows - 1 ||
		point.y<0 || point.y>img.cols - 1)
		return;
	if (visited.at<uchar>(point.x, point.y) == 1 || img.at<uchar>(point.x, point.y) == 0)
	{
		return;
	}
	visited.at<uchar>(point.x, point.y) = 1;
	img.at<uchar>(point.x, point.y) = 255;
	for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++)
		{
			if (i != 0 || j != 0)
			{
				DFS(Point(point.x + i, point.y + j), img, visited, th, flag);
			}
		}
	}
}

void PostProgress::RamerFunc(double* xdata, double* ydata, int beginpoint, int endpoint, int* ResultArray, int &Index, double th)
{
	double Prethreshold = th;
	double threshold = 0;
	int i, m, k;
	double x1, x2, y1, y2, d0, d;
	int markpoint;
	markpoint = 0;
	double xdata1[1000] = { 0 };
	double xdata2[1000] = { 0 };
	double ydata1[1000] = { 0 };
	double ydata2[1000] = { 0 };

	int fornum;
	fornum = endpoint - beginpoint - 1;
	for (i = beginpoint + 1; i <= beginpoint + fornum; i++)
	{
		x1 = xdata[beginpoint]; y1 = ydata[beginpoint];
		x2 = xdata[endpoint]; y2 = ydata[endpoint];
		d0 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
		d = fabs(((x2 - x1)*(ydata[i] - y1) - (y2 - y1)*(xdata[i] - x1)) / d0);
		if (d > threshold)
		{
			markpoint = i;
			threshold = d;
		}
	}
	////////算出两个点之间的最大点和最大值之后，保存最大值的标号///////////////////
	//ResultArray[Index] = markpoint;
	//Index++;
	for (k = beginpoint; k <= markpoint; k++)
	{
		xdata1[k] = xdata[k];
		ydata1[k] = ydata[k];
	}

	for (m = markpoint; m <= endpoint; m++)
	{
		xdata2[m] = xdata[m];
		ydata2[m] = ydata[m];
	}

	if (threshold > Prethreshold)
	{
		RamerFunc(xdata1, ydata1, beginpoint, markpoint, ResultArray, Index, th);
		RamerFunc(xdata2, ydata2, markpoint, endpoint, ResultArray, Index, th);
	}
	else
	{
		//ResultArray[Index - 1] = beginpoint;
		ResultArray[Index] = endpoint;
		Index++;
		if (Index > 300)//防止数组越界
		{
			return;
		}
	}
}

void PostProgress::SortContours(vector<vector<Point> >& inCont, vector<vector<Point> >& outCont)
{
	int lenThresh = 10;
	for (int i = 0; i < inCont.size(); i++)
	{
		vector<vector<Point> > vCnt;
		splitContours(inCont[i], vCnt);

		//for debuger
		int sizeSum = 0;
		for (int m = 0; m < vCnt.size(); m++)
		{
			sizeSum += vCnt[m].size();
		}
		std::cout << inCont[i].size() << "	" << vCnt.size() <<" "<< sizeSum << endl;
		//去除短轮廓，短轮廓一般是毛刺
		for (auto j = vCnt.begin(); j != vCnt.end();)
		{
			if ((*j).size() < lenThresh)
			{
				j = vCnt.erase(j);
			}
			else
				j++;
		}
		//重新连接轮廓
		for (auto j = vCnt.begin(); j != vCnt.end() ; j++)
		{
			for (auto k = j + 1; k != vCnt.end(); k++)
			{
				//如果两条轮廓起点相同，则合并轮廓
				/*if ((*j)[0] == (*k)[0])
				{
					reverse((*j).begin(), (*j).begin());
					(*j).insert((*j).end(), (*k).begin(), (*k).end());
				}*/
			}
		}
		outCont.insert(outCont.end(), vCnt.begin(), vCnt.end());
	}

	//for (int i = 0; i < inCont.size(); i++)
	//{
	//	//===============remove repeat points
	//	vector<Point> temp0;
	//	//cout << conts[i] << endl;
	//	temp0.push_back(inCont[i][0]);
	//	for (int j = 1; j < inCont[i].size(); j++)
	//	{
	//		bool flag = 1;
	//		for (int k = 0; k < temp0.size(); k++)
	//		{
	//			if (inCont[i][j].x == temp0[k].x && inCont[i][j].y == temp0[k].y)
	//			{
	//				flag = 0;
	//				break;
	//			}
	//		}
	//		if (flag == 1)
	//		{
	//			temp0.push_back(inCont[i][j]);
	//		}
	//	}
	//	inCont[i] = temp0;


	//	//int* isVisited = new int[inCont[i].size()];
	//	int isVisited[1000] = { 0 };
	//	std::cout << inCont[i].size();
 //		for (int i=0;i<inCont[i].size();i++)
	//	{
	//		isVisited[i] = 0;
	//	}
	//	int beginIdx = 0;
	//	Point beginP = findEndPoints(inCont[i], beginIdx);
	//	isVisited[beginIdx] = 1;

	//	vector<Point> temp;
	//	temp.push_back(beginP);
	//	bool flag = 1;
	//	Point crossPoint = Point(0, 0);
	//	int idxV = 0;
	//	while (flag)
	//	{
	//		flag = 0; 
	//		int fourNergbFlag = 0;
	//		
	//		int cntNextP = 1;
	//		Point tmpPoint = temp[temp.size() - 1];
	//		
	//		for (int j = 0; j < inCont[i].size(); j++)
	//		{
	//			if (isVisited[j] != 1 && isNeigbour(tmpPoint, inCont[i][j], 1))
	//			{
	//				
	//				if (cntNextP == 1)
	//				{
	//					isVisited[j] = 1;
	//					temp.push_back(inCont[i][j]);
	//					flag = 1;
	//					fourNergbFlag = 1;//存在4邻域

	//				}
	//				else
	//				{
	//					crossPoint = inCont[i][j];
	//					idxV = j;
	//				}
	//				cntNextP++;
	//				//break;
	//			}
	//		}
	//		//如果不存在4邻域，则开始寻找对角邻域，保证4邻域优先
	//		if (fourNergbFlag == 0)
	//		{

	//			for (int j = 0; j < inCont[i].size(); j++)
	//			{
	//				if (isVisited[j] != 1 && isNeigbour(tmpPoint, inCont[i][j], 0))
	//				{
	//					
	//					if (cntNextP == 1)
	//					{
	//						isVisited[j] = 1;
	//						temp.push_back(inCont[i][j]);
	//						flag = 1;
	//					}
	//					else
	//					{
	//						crossPoint = inCont[i][j];
	//						idxV = j;
	//					}
	//					cntNextP++;
	//					//break;
	//				}
	//			}
	//		}
	//		
	//	}
	//	//在轮廓存在交叉点的时候，找完一条边之后接着找另外一条边
	//	if (temp.size() < inCont[i].size())
	//	{
	//		temp.push_back(crossPoint);
	//		isVisited[idxV] = 1;
	//		flag = 1;
	//		while (flag)
	//		{
	//			flag = 0;
	//			int fourNergbFlag = 0;

	//			int cntNextP = 1;
	//			Point tmpPoint = temp[temp.size() - 1];

	//			for (int j = 0; j < inCont[i].size(); j++)
	//			{
	//				if (isVisited[j] != 1 && isNeigbour(tmpPoint, inCont[i][j], 1))
	//				{

	//					if (cntNextP == 1)
	//					{
	//						isVisited[j] = 1;
	//						temp.push_back(inCont[i][j]);
	//						flag = 1;
	//						fourNergbFlag = 1;//存在4邻域

	//					}
	//					else
	//					{
	//						crossPoint = inCont[i][j];
	//						flag = 0;
	//						idxV = j;
	//					}
	//					cntNextP++;
	//					//break;
	//				}
	//			}
	//			//如果不存在4邻域，则开始寻找对角邻域，保证4邻域优先
	//			if (fourNergbFlag == 0)
	//			{

	//				for (int j = 0; j < inCont[i].size(); j++)
	//				{
	//					if (isVisited[j] != 1 && isNeigbour(tmpPoint, inCont[i][j], 0))
	//					{

	//						if (cntNextP == 1)
	//						{
	//							isVisited[j] = 1;
	//							temp.push_back(inCont[i][j]);
	//							flag = 1;
	//						}
	//						else
	//						{
	//							crossPoint = inCont[i][j];
	//							flag = 0;
	//							idxV = j;
	//						}
	//						cntNextP++;
	//						//break;
	//					}
	//				}
	//			}

	//		}
	//	}
	//	

	//	std::cout <<" "<< temp.size() << std::endl;
	//	inCont[i] = temp;
	//	
	//}
}

void PostProgress::FindEndPoint(Mat & srcImg, vector<Point>& vEndPoints)
{
	//!!行列位置存为坐标是注意，存坐标时X轴为列，Y轴为行，所以要先存列再存行
	for (int r = 1; r < srcImg.rows - 1; r++)
	{
		uchar* prevPtr = srcImg.ptr<uchar>(r - 1);
		uchar* ptr = srcImg.ptr<uchar>(r);
		uchar* nextPtr = srcImg.ptr<uchar>(r + 1);
		for (int c = 1; c < srcImg.cols - 1; c++)
		{
			if (ptr[c]!=0)
			{
				//只有一个相邻点时，端点共有8中模式；有两个相邻点时，端点共有8中模式，共16中端点模式
				int neigborNums = 0;
				vector<Point> vNeigbP;
				for (int a = 0; a < 3; a++)
				{
					if (prevPtr[c - 1 + a]!=0)
					{
						neigborNums++;
						vNeigbP.push_back(Point(c + a - 1, r - 1));
					}
				}
				for (int a = 0; a < 3; a++)
				{
					if (ptr[c - 1 + a] != 0 && a != 1)
					{
						neigborNums++;
						vNeigbP.push_back(Point(c + a - 1, r));
					}
				}
				for (int a = 0; a < 3; a++)
				{
					if (nextPtr[c - 1 + a] != 0)
					{
						neigborNums++;
						vNeigbP.push_back(Point(c + a - 1, r + 1));
					}
				}

				if (neigborNums == 1)
				{
					vEndPoints.push_back(Point(c, r));
				}
				if (neigborNums == 2)
				{
					if ((abs(vNeigbP[0].x - vNeigbP[1].x) == 1 && (vNeigbP[0].y - vNeigbP[1].y) == 0) || (abs(vNeigbP[0].y - vNeigbP[1].y) == 1 && (vNeigbP[0].x - vNeigbP[1].x) == 0))
					{
						vEndPoints.push_back(Point(c, r));
					}
				}
				if (neigborNums == 3)
				{
					if ((vNeigbP[0].x== vNeigbP[1].x && vNeigbP[0].x == vNeigbP[2].x) || (vNeigbP[0].y == vNeigbP[1].y && vNeigbP[0].y == vNeigbP[2].y))
					{
						ptr[c] = 0;
					}
				}
				if (neigborNums == 0)
				{
					ptr[c] = 0;
				}
			}
		}
	}
}

void PostProgress::FindCrossPoint(Mat & srcImg, vector<Point>& vCrossPoints)
{
	for (int i = 1; i < srcImg.rows - 1; i++)
	{
		uchar* ptr = srcImg.ptr<uchar>(i);
		for (int j = 1; j < srcImg.cols - 1; j++)
		{
			if (ptr[j] != 0 && IsCrossPont(srcImg, Point(j, i)))
			{
				vCrossPoints.push_back(Point(j, i));
			}
		}
	}
}

bool PostProgress::IsCrossPont(const Mat & srcImg, Point pt)
{
	int r = pt.y;
	int c = pt.x;
	const uchar* prePtr = srcImg.ptr<uchar>(r-1);
	const uchar* ptr = srcImg.ptr<uchar>(r);
	const uchar* nextPtr = srcImg.ptr<uchar>(r+1);
	if (ptr[c-1] != 0 && ptr[c+1] != 0 && (prePtr[c] != 0 || nextPtr[c] != 0))//==== - =====
	{
		return true;
	}
	if (prePtr[c] != 0 && nextPtr[c] != 0 && (ptr[c-1] != 0 || ptr[c+1] != 0))//==== | =====
	{
		return true;
	}
	if (prePtr[c + 1] != 0 && nextPtr[c - 1] != 0 && (prePtr[c-1] != 0 || nextPtr[c+1] != 0))//==== / =====
	{
		return true;
	}
	if (prePtr[c - 1] != 0 && nextPtr[c + 1] != 0 && (prePtr[c+1] != 0 || nextPtr[c-1] != 0))//==== \ =====
	{
		return true;
	}
	//其他模式	==== Y ===== 满足任意两个相邻点之间1范数都大于1
	if (ptr[c + 1] != 0 && prePtr[c - 1] != 0 && nextPtr[c - 1] != 0)
		return true;
	if (prePtr[c + 1] != 0 && ptr[c - 1] != 0 && nextPtr[c] != 0)
		return true;
	if (prePtr[c] != 0 && nextPtr[c - 1] != 0 && nextPtr[c + 1] != 0)
		return true;
	if (prePtr[c - 1] != 0 && nextPtr[c] != 0 && ptr[c + 1] != 0)
		return true;
	if (ptr[c - 1] != 0 && prePtr[c + 1] != 0 && nextPtr[c + 1] != 0)
		return true;
	if (nextPtr[c - 1] != 0 && prePtr[c] != 0 && ptr[c + 1] != 0)
		return true;
	if (nextPtr[c] != 0 && prePtr[c - 1] != 0 && prePtr[c + 1] != 0)
		return true;
	if (nextPtr[c + 1] != 0 && ptr[c - 1] != 0 && prePtr[c] != 0)
		return true;

	return false;
}

void PostProgress::RemoveBurr(Mat & srcImg, vector<Point>& vEndPoints, vector<Point>& vCrossPoints, int lenThresh)
{
	int lengThresh = lenThresh;//length threshhold
	for (int i = 0; i < vEndPoints.size(); i++)
	{
		srcImg.at<uchar>(vEndPoints[i].y, vEndPoints[i].x) = 1;//端点赋1
	}
	for (int i = 0; i < vCrossPoints.size(); i++)
	{
		srcImg.at<uchar>(vCrossPoints[i].y, vCrossPoints[i].x) = 2;//交叉点赋2
	}
	vector<int> delIdx;
	Mat isVisited(srcImg.size(), CV_8U, Scalar(0));
	for (int i=0;i<vEndPoints.size();i++)
	{
		int pNums = 0;
		vector<Point> vTmpPoints;
		vTmpPoints.push_back(vEndPoints[i]);
		isVisited.at<uchar>(vEndPoints[i].y, vEndPoints[i].x) = 1;
		for (int j=0;j<lengThresh;j++)
		{
			pNums++;
			int r = vTmpPoints[vTmpPoints.size() - 1].y;
			int c = vTmpPoints[vTmpPoints.size() - 1].x;
			//如果追踪到了边界，则追踪结束
			if (r == 0 || r == srcImg.rows -1  || c == 0 || c == srcImg.cols - 1)
			{
				break;
			}
			uchar* prevPtr = srcImg.ptr<uchar>(r-1);
			uchar* ptr = srcImg.ptr<uchar>(r);
			uchar* nextPtr = srcImg.ptr<uchar>(r+1);
			bool bTraced = 0;
			if (ptr[c + 1] !=0 && !isVisited.at<uchar>(r,c+1))
			{
				//如果碰到交叉点就结束跟踪
				if (ptr[c + 1] == 2 || ptr[c + 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c+1, r));
				isVisited.at<uchar>(r, c + 1) = 1;
			}
			else if (prevPtr[c] != 0 && !isVisited.at<uchar>(r-1,c))
			{
				if (prevPtr[c] == 2 || prevPtr[c] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c, r-1));
				isVisited.at<uchar>(r - 1, c) = 1;
			}
			else if (ptr[c - 1] != 0 && !isVisited.at<uchar>(r, c - 1))
			{
				if (ptr[c - 1] == 1 || ptr[c - 1] == 2)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c-1, r));
				isVisited.at<uchar>(r, c - 1) = 1;
			}
			else if (nextPtr[c] != 0 && !isVisited.at<uchar>(r+1,c))
			{
				if (nextPtr[c] == 2 || nextPtr[c] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c, r+1));
				isVisited.at<uchar>(r+1,c) = 1;
			}
			//========
			else if (prevPtr[c + 1] != 0 && !isVisited.at<uchar>(r-1, c + 1))
			{
				if (prevPtr[c + 1] == 2 || prevPtr[c + 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c+1, r-1));
				isVisited.at<uchar>(r - 1, c + 1) = 1;
			}
			else if (prevPtr[c - 1] != 0 && !isVisited.at<uchar>(r-1,c - 1))
			{
				if (prevPtr[c - 1] == 2 || prevPtr[c - 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c - 1, r - 1));
				isVisited.at<uchar>(r - 1, c - 1) = 1;
			}
			else if (nextPtr[c - 1] != 0 && !isVisited.at<uchar>(r+1,c-1))
			{
				if (nextPtr[c - 1] == 2 || nextPtr[c - 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c-1, r+1));
				isVisited.at<uchar>(r + 1, c - 1) = 1;
			}
			else if (nextPtr[c + 1] != 0 && !isVisited.at<uchar>(r+1, c + 1))
			{
				if (nextPtr[c + 1] == 2 || nextPtr[c + 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c + 1, r + 1));
				isVisited.at<uchar>(r + 1, c + 1) = 1;
			}
			if (!bTraced)
			{
				break;
			}
			
		}
		if (pNums < lengThresh)
		{
			for (int k=0;k<vTmpPoints.size();k++)
			{
				srcImg.at<uchar>(vTmpPoints[k].y, vTmpPoints[k].x) = 0;
			}
			delIdx.push_back(i);
		}
	 }
	for (int i = delIdx.size() - 1; i >= 0; i--)
	{
		vEndPoints.erase(vEndPoints.begin() + delIdx[i]);//删除向量中的点时，从后往前删，避免索引变化带来的影响
	}
	for (int i = 0; i < vCrossPoints.size(); i++)
	{
		srcImg.at<uchar>(vCrossPoints[i].y, vCrossPoints[i].x) = 255;//交叉点还原
	}
}

void PostProgress::RemoveBurr(Mat & srcImg, int th)
{
	vector<Point> vEndPoints, vCrossPoints;
	FindEndPoint(srcImg, vEndPoints);
	FindCrossPoint(srcImg, vCrossPoints);
	RemoveBurr(srcImg, vEndPoints, vCrossPoints, th);
}

void PostProgress::EdgeTrack(Mat & srcImg, vector<vector<Point>>& vvPoints)
{
	vector<Point> vEndPoints, vCrossPoints;
	FindEndPoint(srcImg, vEndPoints);
	FindCrossPoint(srcImg, vCrossPoints);
	for (int i = 0; i < vEndPoints.size(); i++)
	{
		srcImg.at<uchar>(vEndPoints[i].y, vEndPoints[i].x) = 1;//端点赋1
	}
	for (int i = 0; i < vCrossPoints.size(); i++)
	{
		srcImg.at<uchar>(vCrossPoints[i].y, vCrossPoints[i].x) = 2;//交叉点赋2
	}
	Mat isVisited(srcImg.size(), CV_8U, Scalar(0));//0表示未被追踪过，1表示被追踪过
	
	//追踪有非环形轮廓
	for (int i = 0; i < vEndPoints.size(); i++)
	{
		int pNums = 0;
		vector<Point> vTmpPoints;
		vTmpPoints.push_back(vEndPoints[i]);
		isVisited.at<uchar>(vEndPoints[i].y, vEndPoints[i].x) = 1;
		while(1)
		{
			pNums++;
			int r = vTmpPoints[vTmpPoints.size() - 1].y;
			int c = vTmpPoints[vTmpPoints.size() - 1].x;
			//如果追踪到了边界，则追踪结束
			if (r == 0 || r == srcImg.rows - 1 || c == 0 || c == srcImg.cols - 1)
			{
				break;
			}
			uchar* prevPtr = srcImg.ptr<uchar>(r - 1);
			uchar* ptr = srcImg.ptr<uchar>(r);
			uchar* nextPtr = srcImg.ptr<uchar>(r + 1);
			bool bTraced = 0;
			if (ptr[c + 1] != 0 && !isVisited.at<uchar>(r, c + 1))
			{
				//如果碰到交叉点就结束跟踪
				if (ptr[c + 1] == 2 || ptr[c + 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c + 1, r));
				isVisited.at<uchar>(r, c + 1) = 1;
			}
			else if (prevPtr[c] != 0 && !isVisited.at<uchar>(r - 1, c))
			{
				if (prevPtr[c] == 2 || prevPtr[c] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c, r - 1));
				isVisited.at<uchar>(r - 1, c) = 1;
			}
			else if (ptr[c - 1] != 0 && !isVisited.at<uchar>(r, c - 1))
			{
				if (ptr[c - 1] == 1 || ptr[c - 1] == 2)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c - 1, r));
				isVisited.at<uchar>(r, c - 1) = 1;
			}
			else if (nextPtr[c] != 0 && !isVisited.at<uchar>(r + 1, c))
			{
				if (nextPtr[c] == 2 || nextPtr[c] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c, r + 1));
				isVisited.at<uchar>(r + 1, c) = 1;
			}
			//========
			else if (prevPtr[c + 1] != 0 && !isVisited.at<uchar>(r - 1, c + 1))
			{
				if (prevPtr[c + 1] == 2 || prevPtr[c + 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c + 1, r - 1));
				isVisited.at<uchar>(r - 1, c + 1) = 1;
			}
			else if (prevPtr[c - 1] != 0 && !isVisited.at<uchar>(r - 1, c - 1))
			{
				if (prevPtr[c - 1] == 2 || prevPtr[c - 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c - 1, r - 1));
				isVisited.at<uchar>(r - 1, c - 1) = 1;
			}
			else if (nextPtr[c - 1] != 0 && !isVisited.at<uchar>(r + 1, c - 1))
			{
				if (nextPtr[c - 1] == 2 || nextPtr[c - 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c - 1, r + 1));
				isVisited.at<uchar>(r + 1, c - 1) = 1;
			}
			else if (nextPtr[c + 1] != 0 && !isVisited.at<uchar>(r + 1, c + 1))
			{
				if (nextPtr[c + 1] == 2 || nextPtr[c + 1] == 1)
					break;
				bTraced = 1;
				vTmpPoints.push_back(Point(c + 1, r + 1));
				isVisited.at<uchar>(r + 1, c + 1) = 1;
			}
			if (!bTraced)
			{
				break;
			}
			assert(pNums < 10000);
		}
		if (vTmpPoints.size() > 1)
			vvPoints.push_back(vTmpPoints);
	}

	//追踪环形轮廓
	for (int r=0;r<srcImg.rows;r++)
	{
		uchar* ptr = srcImg.ptr<uchar>(r);
		for (int c=0;c<srcImg.cols;c++)
		{
			if (!isVisited.at<uchar>(r, c))
			{
				int pNums = 0;
				vector<Point> vTmpPoints;
				vTmpPoints.push_back(Point(c,r));
				isVisited.at<uchar>(r,c) = 1;
				while (1)
				{
					pNums++;
					int r = vTmpPoints[vTmpPoints.size() - 1].y;
					int c = vTmpPoints[vTmpPoints.size() - 1].x;
					//如果追踪到了边界，则追踪结束
					if (r == 0 || r == srcImg.rows - 1 || c == 0 || c == srcImg.cols - 1)
					{
						break;
					}
					uchar* prevPtr = srcImg.ptr<uchar>(r - 1);
					uchar* ptr = srcImg.ptr<uchar>(r);
					uchar* nextPtr = srcImg.ptr<uchar>(r + 1);
					bool bTraced = 0;
					if (ptr[c + 1] != 0 && !isVisited.at<uchar>(r, c + 1))
					{
						//如果碰到交叉点就结束跟踪
						if (ptr[c + 1] == 2 || ptr[c + 1] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c + 1, r));
						isVisited.at<uchar>(r, c + 1) = 1;
					}
					else if (prevPtr[c] != 0 && !isVisited.at<uchar>(r - 1, c))
					{
						if (prevPtr[c] == 2 || prevPtr[c] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c, r - 1));
						isVisited.at<uchar>(r - 1, c) = 1;
					}
					else if (ptr[c - 1] != 0 && !isVisited.at<uchar>(r, c - 1))
					{
						if (ptr[c - 1] == 1 || ptr[c - 1] == 2)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c - 1, r));
						isVisited.at<uchar>(r, c - 1) = 1;
					}
					else if (nextPtr[c] != 0 && !isVisited.at<uchar>(r + 1, c))
					{
						if (nextPtr[c] == 2 || nextPtr[c] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c, r + 1));
						isVisited.at<uchar>(r + 1, c) = 1;
					}
					//========
					else if (prevPtr[c + 1] != 0 && !isVisited.at<uchar>(r - 1, c + 1))
					{
						if (prevPtr[c + 1] == 2 || prevPtr[c + 1] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c + 1, r - 1));
						isVisited.at<uchar>(r - 1, c + 1) = 1;
					}
					else if (prevPtr[c - 1] != 0 && !isVisited.at<uchar>(r - 1, c - 1))
					{
						if (prevPtr[c - 1] == 2 || prevPtr[c - 1] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c - 1, r - 1));
						isVisited.at<uchar>(r - 1, c - 1) = 1;
					}
					else if (nextPtr[c - 1] != 0 && !isVisited.at<uchar>(r + 1, c - 1))
					{
						if (nextPtr[c - 1] == 2 || nextPtr[c - 1] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c - 1, r + 1));
						isVisited.at<uchar>(r + 1, c - 1) = 1;
					}
					else if (nextPtr[c + 1] != 0 && !isVisited.at<uchar>(r + 1, c + 1))
					{
						if (nextPtr[c + 1] == 2 || nextPtr[c + 1] == 1)
							break;
						bTraced = 1;
						vTmpPoints.push_back(Point(c + 1, r + 1));
						isVisited.at<uchar>(r + 1, c + 1) = 1;
					}
					if (!bTraced)
					{
						break;
					}
					assert(pNums < 10000);
				}
				if (vTmpPoints.size() > 1)
					vvPoints.push_back(vTmpPoints);
			}
		}
	}
}

void PostProgress::FittingDenosing(vector<vector<Point>>& srcVec, vector<vector<Point>>& dstVec, int lineArr[], double curR, double disT)
{
	PostProgress postPro;
	double curRadio = curR;//总点数/线段数
	for (auto iter1 = srcVec.begin(); iter1 != srcVec.end(); iter1++)
	{
		double xdata[3000] = { 0 };
		double ydata[3000] = { 0 };
		int pNums = 0;
		for (auto iter2 = iter1->begin(); iter2 < iter1->end(); iter2++)
		{
			xdata[pNums] = iter2->x;
			ydata[pNums] = iter2->y;
			pNums++;
		}
		//int resArr[300] = { 0 };
		lineArr[0] = 0;//保存线段端点索引下标的数组
		int ind = 1;//中间变量，下标
		double th = disT;//直线拟合距离阈值
		postPro.RamerFunc(xdata, ydata, 0, pNums - 1, lineArr, ind, th);
		if (pNums / (ind-1) > curRadio && pNums > 20)
		{
			dstVec.push_back(*iter1);
		}
	}
}

void PostProgress::Vec2Mat(vector<vector<Point>>& vvPoint, Mat & dstImg)
{
	for (auto it = vvPoint.begin(); it != vvPoint.end(); it++)
	{
		for (auto itp = (*it).begin(); itp != (*it).end(); itp++)
		{
			dstImg.at<uchar>((*itp).y, (*itp).x) = 255;
		}
	}
}

Point PostProgress::findEndPoints(vector<Point>& vp, int& beginIdx)
{
	int minx = vp[0].x;
	int miny = vp[0].y;

	for (int i = 1; i < vp.size(); i++)
	{
		if (vp[i].x < minx)
		{
			minx = vp[i].x;
			miny = vp[i].y;
			beginIdx = i;
		}
	}
	return Point(minx, miny);
}

bool PostProgress::isNeigbour(Point& a, Point& b, int isFourNeig)
{
	if (a.x == b.x && a.y == b.y)
	{
		return 0;
	}
	if (isFourNeig)
	{
		if ((abs(a.x - b.x) == 0 && abs(a.y - b.y) ==1) || (abs(a.y - b.y) == 0 && abs(a.x - b.x) == 1))//直接相邻
		{
			return 1;
		}
	}
	
	else 
	{
		if (abs(a.x - b.x) < 2 && abs(a.y - b.y) < 2)//对角相邻
		{
			return 1;
		}
	}
	return 0;
}

bool PostProgress::isNeigbour(const Point & a, const Point & b)
{
	if (a.x == b.x && a.y == b.y)
	{
		return 0;
	}
	if (abs(a.x - b.x) < 2 && abs(a.y - b.y) < 2)//直接相邻
	{
		return 1;
	}
	return 0;
}

int PostProgress::distance(Point& a, Point& b)
{
	return abs(a.x - b.x) + abs(a.y - b.y);
}

void PostProgress::splitContours(vector<Point>& srcCont, vector<vector<Point> >& dstConts)
{
	removeRepeat(srcCont);
	vector<int> crossPoint;
	FindCross(srcCont, crossPoint);
	if (crossPoint.size() == 0)
	{
		dstConts.push_back(srcCont);
		return;
	}
	int isVisited[1000] = { 0 };
	for (int i = 0; i < crossPoint.size(); i++)
	{
		isVisited[crossPoint[i]] = 1;
	}

	for (int i = 1; i < crossPoint.size(); i += 4)
	{
		vector<Point > tmpPoints1;
		tmpPoints1.push_back(srcCont[crossPoint[i - 1]]);
		getSingleCnt(crossPoint[i], srcCont, isVisited, tmpPoints1);
		dstConts.push_back(tmpPoints1);

		vector<Point > tmpPoints2;
		tmpPoints2.push_back(srcCont[crossPoint[i - 1]]);
		getSingleCnt(crossPoint[i + 1], srcCont, isVisited, tmpPoints2);
		dstConts.push_back(tmpPoints2); 

		vector<Point > tmpPoints3;
		tmpPoints3.push_back(srcCont[crossPoint[i - 1]]);//需要把交叉点本身包含进去
		getSingleCnt(crossPoint[i + 2], srcCont, isVisited, tmpPoints3);
		dstConts.push_back(tmpPoints3);
	}

	////int* isVisited = new int[inCont[i].size()];
	//int isVisited[1000] = { 0 };
	//std::cout << srcCont.size();
	//int beginIdx = 0;
	//Point beginP = findEndPoints(srcCont, beginIdx);
	//isVisited[beginIdx] = 1;
	//
	//bool flag = 1;
	//Point crossPoint = Point(0, 0);
	//int idxV = 0;
	//Point neigbPointsArr[10] = { (0,0) };
	//int crossNums = 0;
	//vector<Point> temp;
	//temp.push_back(beginP);
	//while (flag)
	//{
	//	flag = 0;
	//	int cntNextP = 1;
	//	Point tmpPoint = temp[temp.size() - 1];
	//	crossNums = 0;
	//	for (int j = 0; j < srcCont.size(); j++)
	//	{
	//		if (isVisited[j] != 1 && isNeigbour(tmpPoint, srcCont[j]))
	//		{
	//			isVisited[j] = 1;
	//			neigbPointsArr[crossNums] = srcCont[j];
	//			crossNums++;
	//			crossPoint = srcCont[j];
	//			idxV = j;
	//			cntNextP++;
	//			//break;
	//		}
	//	}
	//	bool isTurn = 0;
	//	if ((crossNums == 2) && ((neigbPointsArr[0].x - neigbPointsArr[1].x)*(neigbPointsArr[0].y - neigbPointsArr[1].y)) == 0)
	//	{
	//		isTurn = 1;
	//		//排列相邻点的顺序，是的4邻域点排在8邻域点前面，可以优先连接4邻域点
	//		if ((neigbPointsArr[0].x - tmpPoint.x)*(neigbPointsArr[0].y - tmpPoint.y) != 0)
	//		{
	//			Point tempP;
	//			tempP = neigbPointsArr[0];
	//			neigbPointsArr[0] = neigbPointsArr[1];
	//			neigbPointsArr[1] = tempP;
	//		}
	//	}
	//	if (crossNums == 1 || isTurn)
	//	{
	//		temp.push_back(neigbPointsArr[0]);
	//		continue;
	//	}
	//	else
	//		break;
	//}
	////在轮廓存在交叉点的时候，找完一条边之后接着找另外一条边
	////if (temp.size() < inCont[i].size())
	////{
	////	temp.push_back(crossPoint);
	////	isVisited[idxV] = 1;
	////	flag = 1;
	////	while (flag)
	////	{
	////		flag = 0;
	////		int fourNergbFlag = 0;
	////		int cntNextP = 1;
	////		Point tmpPoint = temp[temp.size() - 1];
	////		for (int j = 0; j < inCont[i].size(); j++)
	////		{
	////			if (isVisited[j] != 1 && isNeigbour(tmpPoint, inCont[i][j], 1))
	////			{
	////				if (cntNextP == 1)
	////				{
	////					isVisited[j] = 1;
	////					temp.push_back(inCont[i][j]);
	////					flag = 1;
	////					fourNergbFlag = 1;//存在4邻域
	////				}
	////				else
	////				{
	////					crossPoint = inCont[i][j];
	////					flag = 0;
	////					idxV = j;
	////				}
	////				cntNextP++;
	////				//break;
	////			}
	////		}
	////		//如果不存在4邻域，则开始寻找对角邻域，保证4邻域优先
	////		if (fourNergbFlag == 0)
	////		{
	////			for (int j = 0; j < inCont[i].size(); j++)
	////			{
	////				if (isVisited[j] != 1 && isNeigbour(tmpPoint, inCont[i][j], 0))
	////				{
	////					if (cntNextP == 1)
	////					{
	////						isVisited[j] = 1;
	////						temp.push_back(inCont[i][j]);
	////						flag = 1;
	////					}
	////					else
	////					{
	////						crossPoint = inCont[i][j];
	////						flag = 0;
	////						idxV = j;
	////					}
	////					cntNextP++;
	////					//break;
	////				}
	////			}
	////		}
	////	}
	////}

}

void PostProgress::removeRepeat(vector<Point>& srcCont)
{
	//===============remove repeat points
	vector<Point> temp0;
	//cout << conts[i] << endl;
	temp0.push_back(srcCont[0]);
	for (int j = 1; j < srcCont.size(); j++)
	{
		bool flag = 1;
		for (int k = 0; k < temp0.size(); k++)
		{
			if (srcCont[j].x == temp0[k].x && srcCont[j].y == temp0[k].y)
			{
				flag = 0;
				break;
			}
		}
		if (flag == 1)
		{
			temp0.push_back(srcCont[j]);
		}
	}
	srcCont = temp0;
}

void PostProgress::FindCross(const vector<Point>& inCnt, vector<int >& crossIdx)
{
	int pNums = inCnt.size();
	for (int i = 0; i < pNums; i++)
	{
		vector<int> tmpIdx;
		int idx = 0;
		//找出所有相邻点
		for (int j = 0; j < pNums; j++)
		{
			if (isNeigbour(inCnt[i], inCnt[j]))
			{
				tmpIdx.push_back(idx);
			}
			idx++;
		}
		//为了简单起见，暂时只考虑三点相邻情况，如果3点中两两之前的距离大于1，则为交叉点
		if (tmpIdx.size() >= 3)
		{
			int flag = 1;
			for (int k = 0; k < tmpIdx.size(); k++)
			{
				int deltaPosx = inCnt[tmpIdx[k]].x - inCnt[i].x;
				int deltaPosy = inCnt[tmpIdx[k]].y - inCnt[i].y;

				if (deltaPosx == 1 && deltaPosy ==0)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].x - inCnt[i].x) == -1 && (inCnt[tmpIdx[m]].y - inCnt[i].y) == 0)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if ((inCnt[tmpIdx[n]].x - inCnt[i].x) == 0 && abs(inCnt[tmpIdx[n]].y - inCnt[i].y) == 1)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;
									break;
								}
							}
						}
					}
				}
				else if (deltaPosx == -1 && deltaPosy == 0)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].x - inCnt[i].x) == 1 && (inCnt[tmpIdx[m]].y - inCnt[i].y) == 0)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if ((inCnt[tmpIdx[n]].x - inCnt[i].x) == 0 && abs(inCnt[tmpIdx[n]].y - inCnt[i].y) == 1)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;
									break;
								}
							}
						}
					}
				}
				else if (deltaPosy == 1 && deltaPosx == 0)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].y - inCnt[i].y) == -1 && (inCnt[tmpIdx[m]].x - inCnt[i].x) == 0)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if (abs(inCnt[tmpIdx[n]].x - inCnt[i].x) == 1 && abs(inCnt[tmpIdx[n]].y - inCnt[i].y) == 0)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;

									break;
								}
							}
						}
					}
				}
				else if (deltaPosy == -1 && deltaPosx == 0)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].y - inCnt[i].y) == 1 && (inCnt[tmpIdx[m]].x - inCnt[i].x) == 0)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if (abs(inCnt[tmpIdx[n]].x - inCnt[i].x) == 1 && abs(inCnt[tmpIdx[n]].y - inCnt[i].y) == 0)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;

									break;
								}
							}
						}
					}
				}
				else if (deltaPosx == 1 && deltaPosy == 1)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].x - inCnt[i].x) == -1 && (inCnt[tmpIdx[m]].y - inCnt[i].y) == -1)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if ((inCnt[tmpIdx[n]].x - inCnt[i].x)*(inCnt[tmpIdx[n]].y - inCnt[i].y) == -1)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;

									break;
								}
							}
						}
					}
				}
				else if (deltaPosx == -1 && deltaPosy == -1)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].x - inCnt[i].x) == 1 && (inCnt[tmpIdx[m]].y - inCnt[i].y) == 1)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if ((inCnt[tmpIdx[n]].x - inCnt[i].x)*(inCnt[tmpIdx[n]].y - inCnt[i].y) == -1)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;

									break;
								}
							}
						}
					}
				}
				else if (deltaPosx == 1 && deltaPosy == -1)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].x - inCnt[i].x) == -1 && (inCnt[tmpIdx[m]].y - inCnt[i].y) == 1)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if ((inCnt[tmpIdx[n]].x - inCnt[i].x)*(inCnt[tmpIdx[n]].y - inCnt[i].y) == 1)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;

									break;
								}
							}
						}
					}
				}
				else if (deltaPosx == -1 && deltaPosy == 1)
				{
					for (int m = 0; m < tmpIdx.size(); m++)
					{
						if ((inCnt[tmpIdx[m]].x - inCnt[i].x) == 1 && (inCnt[tmpIdx[m]].y - inCnt[i].y) == -1)
						{
							for (int n = 0; n < tmpIdx.size(); n++)
							{
								if ((inCnt[tmpIdx[n]].x - inCnt[i].x)*(inCnt[tmpIdx[n]].y - inCnt[i].y) == 1)
								{
									crossIdx.push_back(i);
									crossIdx.push_back(tmpIdx[k]);
									crossIdx.push_back(tmpIdx[m]);
									crossIdx.push_back(tmpIdx[n]);
									flag = 0;

									break;
								}
							}
						}
					}
				}
				if (flag == 0)
					break;
			}



			//int norm1 = abs(inCnt[tmpIdx[0]].x - inCnt[tmpIdx[1]].x) + abs(inCnt[tmpIdx[0]].y - inCnt[tmpIdx[1]].y);
			//int norm2 = abs(inCnt[tmpIdx[0]].x - inCnt[tmpIdx[2]].x) + abs(inCnt[tmpIdx[0]].y - inCnt[tmpIdx[2]].y);
			//int norm3 = abs(inCnt[tmpIdx[1]].x - inCnt[tmpIdx[2]].x) + abs(inCnt[tmpIdx[1]].y - inCnt[tmpIdx[2]].y);
			//if (norm1 > 1 && norm2 > 1 && norm3 > 1)
			//{
			//	//存储交叉点和3个相邻的点
			//	crossIdx.push_back(i);
			//	crossIdx.push_back(tmpIdx[0]);
			//	crossIdx.push_back(tmpIdx[1]);
			//	crossIdx.push_back(tmpIdx[2]);
			//}
		}
	}
}

void PostProgress::getSingleCnt(int beginPointIdx, vector<Point>& inCnt, int * isVisited, vector<Point>& outCnt)
{
	outCnt.push_back(inCnt[beginPointIdx]);
	//if (isVisited[beginPointIdx])
	//	return;//如果点之前被追踪过，那么直接返回，以免一条轮廓追踪两次
	isVisited[beginPointIdx] = 1;
	bool isNotEnd = 1;
	while (isNotEnd)
	{
		isNotEnd = 0;
		for (int i = 0; i < inCnt.size(); i++)
		{
			if (isVisited[i] == 0 && isNeigbour(outCnt[outCnt.size() - 1],inCnt[i]))
			{
				outCnt.push_back(inCnt[i]);
				isVisited[i] = 1;
				isNotEnd = 1;
			}
		}
	}
}

void NMS(Mat& srcx, Mat& srcy, Mat& dst)
{
	Mat mag(srcx.size(), CV_32F, Scalar(0));//保存梯度大小值
	Mat pha(srcx.size(), CV_32F, Scalar(0));

	//srcx.convertTo(srcx, CV_32F);
	//srcy.convertTo(srcy, CV_32F);

	for (int r = 1; r<srcx.rows - 1; r++)
	{
		for (int c = 1; c<srcx.cols - 1; c++)
		{
			mag.at<float>(r, c) = sqrt(srcx.at<float>(r, c)*srcx.at<float>(r, c) + srcy.at<float>(r, c)*srcy.at<float>(r, c));
			pha.at<float>(r, c) = atan2(srcy.at<float>(r, c), srcx.at<float>(r, c));

		}
	}
	//cartToPolar(srcx, srcy, mag, pha);


	//std::cout << pha << std::endl;
	//dst.convertTo(dst, CV_32F, 1.0/255);
	//dst = mag.clone();
	//normalize(mag, mag, 0, 1, NORM_MINMAX);
	//imshow("dst in nms", mag);
	//waitKey(0);
	//dst.convertTo(dst, CV_32F);
	/*normalize(dst, dst, 0, 1, NORM_MINMAX);
	imshow("dst in nms", dst);
	waitKey(0);*/

	for (int r = 1; r < srcx.rows - 1; r++)
	{
		for (int c = 1; c < srcx.cols - 1; c++)
		{
			float angle = pha.at<float>(r, c);
			if (angle < 0)
				angle += 2 * CV_PI;
			if ((angle >= 1.875*CV_PI || angle < 0.125*CV_PI) || (angle >= 0.875 * CV_PI && angle < 1.125*CV_PI))//第一个条件要用||,处在圆首尾交接处
			{
				float v0 = mag.at<float>(r, c);
				float v1 = mag.at<float>(r, c - 1);
				float v2 = mag.at<float>(r, c + 1);
				if (v0 < v1 || v0 < v2)
				{
					dst.at<float>(r, c) = 0;
				}

				//条件放宽
				/*uchar v0 = mag.at<float>(r, c);
				uchar v1 = mag.at<float>(r, c + 1);
				uchar v2 = mag.at<float>(r-1, c+1);
				uchar v3 = mag.at<float>(r - 1, c);
				uchar v4 = mag.at<float>(r-1, c-1);
				uchar v5 = mag.at<float>(r, c-1);
				uchar v6 = mag.at<float>(r + 1, c-1);
				uchar v7 = mag.at<float>(r + 1, c);
				uchar v8 = mag.at<float>(r + 1, c+1);

				if (!(v0>v1&&v0>v5) && !(v0>v2&&v0>v6) && !(v0>v4&&v0>v8))
				{
				continue;
				}
				dst.at<float>(r, c) = 0;*/
			}
			if ((angle >= 0.125*CV_PI && angle < 0.375*CV_PI) || (angle >= 1.125 * CV_PI && angle < 1.375*CV_PI))
			{
				float v0 = mag.at<float>(r, c);
				float v1 = mag.at<float>(r + 1, c - 1);
				float v2 = mag.at<float>(r - 1, c + 1);
				if (v0 < v1 || v0 < v2)
				{
					dst.at<float>(r, c) = 0;
				}

				/*uchar v0 = mag.at<float>(r, c);
				uchar v1 = mag.at<float>(r, c + 1);
				uchar v2 = mag.at<float>(r - 1, c + 1);
				uchar v3 = mag.at<float>(r - 1, c);
				uchar v4 = mag.at<float>(r - 1, c - 1);
				uchar v5 = mag.at<float>(r, c - 1);
				uchar v6 = mag.at<float>(r + 1, c - 1);
				uchar v7 = mag.at<float>(r + 1, c);
				uchar v8 = mag.at<float>(r + 1, c + 1);

				if (!(v0 > v3&&v0 > v7) && !(v0 > v2&&v0 > v6) && !(v0 > v5&&v0 > v1))
				{
				continue;
				}
				dst.at<float>(r, c) = 0;*/
			}
			if ((angle >= 0.375*CV_PI && angle < 0.625*CV_PI) || (angle >= 1.375 * CV_PI && angle < 1.625*CV_PI))
			{
				float v0 = mag.at<float>(r, c);
				float v1 = mag.at<float>(r - 1, c);
				float v2 = mag.at<float>(r + 1, c);
				if (v0 < v1 || v0 < v2)
				{
					dst.at<float>(r, c) = 0;
				}

				/*uchar v0 = mag.at<float>(r, c);
				uchar v1 = mag.at<float>(r, c + 1);
				uchar v2 = mag.at<float>(r - 1, c + 1);
				uchar v3 = mag.at<float>(r - 1, c);
				uchar v4 = mag.at<float>(r - 1, c - 1);
				uchar v5 = mag.at<float>(r, c - 1);
				uchar v6 = mag.at<float>(r + 1, c - 1);
				uchar v7 = mag.at<float>(r + 1, c);
				uchar v8 = mag.at<float>(r + 1, c + 1);

				if (!(v0 > v1&&v0 > v5) && !(v0 > v3&&v0 > v7) && !(v0 > v4&&v0 > v8))
				{
				continue;
				}*/
				dst.at<float>(r, c) = 0;
			}
			if ((angle >= 0.625*CV_PI && angle < 0.875*CV_PI) || (angle >= 1.625 * CV_PI && angle < 1.875*CV_PI))
			{
				float v0 = mag.at<float>(r, c);
				float v1 = mag.at<float>(r - 1, c - 1);
				float v2 = mag.at<float>(r + 1, c + 1);
				if (v0 < v1 || v0 < v2)
				{

					dst.at<float>(r, c) = 0;
				}

				/*uchar v0 = mag.at<float>(r, c);
				uchar v1 = mag.at<float>(r, c + 1);
				uchar v2 = mag.at<float>(r - 1, c + 1);
				uchar v3 = mag.at<float>(r - 1, c);
				uchar v4 = mag.at<float>(r - 1, c - 1);
				uchar v5 = mag.at<float>(r, c - 1);
				uchar v6 = mag.at<float>(r + 1, c - 1);
				uchar v7 = mag.at<float>(r + 1, c);
				uchar v8 = mag.at<float>(r + 1, c + 1);

				if (!(v0 > v3&&v0 > v7) && !(v0 > v2&&v0 > v6) && !(v0 > v4&&v0 > v8))
				{
				continue;
				}
				dst.at<float>(r, c) = 0;*/
			}
		}
	}
	//std::cout << dst << std::endl;
	//dst.convertTo(dst, CV_8U);
}

void NMS(const Mat& src, Mat& dst)
{
	Mat diffx(src.size(), CV_32F);
	Mat diffy(src.size(), CV_32F);
	Sobel(src, diffx, CV_32F, 1, 0);
	Sobel(src, diffy, CV_32F, 0, 1);
	NMS(diffx, diffy, dst);//调用重载函数
}