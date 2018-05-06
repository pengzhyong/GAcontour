#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
using namespace std;
using namespace cv;
class PostProgress
{
public:
	PostProgress();
	~PostProgress();
	//非极大值抑制

	//滞后阈值连接
	void HTC(Mat& src, Mat& dst, int thHigh, int thLow);
	//深度优先搜索，用于滞后阈值连接
	void DFS(const Point point, Mat& img, Mat& visited, int th, int* flag);
	//Sobel算子
	void MySobel(Mat& src, Mat& dst, Mat& dstx, Mat& dsty);
	//对轮廓进行多边形逼近，通过轮廓点数与多边形数目的比值来判定轮廓是否是噪声轮廓，依据轮廓的低曲率特性
	void RamerFunc(double* xdata, double* ydata, int beginpoint, int endpoint, int* ResultArray, int &Index, double th);
	//对轮廓进行排序，这一步应在多边形逼近之前进行，如果轮廓为进行排序，可能会导致逼近出现错误的结果
	void SortContours(vector<vector<Point> >& inCont, vector<vector<Point> >& outCont);
	//找出轮廓中的交叉点,返回交叉点的索引号；在所有模式中，只有T形模式的结构才是交叉点，共有8中T形模式
	void FindCross(const vector<Point >& inCnt, vector<int >& crossIdx);
	
	//================以下函数是在图像基础上直接进行
	//找到图像中的端点，为去毛刺做准备
	void FindEndPoint(Mat& srcImg, vector<Point>& vEndPoints);
	void FindCrossPoint(Mat& srcImg, vector<Point>& vCrossPoints);
	bool IsCrossPont(const Mat& srcImg, Point pt);
	void RemoveBurr(Mat& srcImg, vector<Point>& vEndPoint, vector<Point>& vCrossPoint, int lenThresh = 20);
	void RemoveBurr(Mat& srcImg, int th);//移除毛刺是，对于环状噪声无法去除，这是 因为环状轮廓找不到端点和交叉点
	void EdgeTrack(Mat& srcImg, vector<vector<Point> >& vvPoints);
	void FittingDenosing(vector<vector<Point> >& srcVec, vector<vector<Point> >& dstVec, int lineArr[], double curRadio = 20, double disTh = 3);
	void Vec2Mat(vector<vector<Point> >& vvPoint, Mat& dstImg);
private:
	Point findEndPoints(vector<Point>&, int& );
	bool isNeigbour(Point & a, Point & b, int isFourNeig);
	bool isNeigbour(const Point & a, const Point & b);
	int distance(Point&, Point&);
	void splitContours(vector<Point >& srcConts, vector<vector<Point > >& dstCnt);//将含有多条交叉线的轮廓进行分解，使得每一条线轮廓没有分支;采用递归方式
	void removeRepeat(vector<Point>&);
	//找出没有交叉点的轮廓，轮廓从交叉点的一个邻域点开始跟踪，直到走到端点或者下一个交叉点
	void getSingleCnt(int beginPointIdx, vector<Point >& inCnt, int* isVisited, vector<Point >& outCnt);
};

void NMS(Mat& srcx, Mat& srcy, Mat& dst);
void NMS(const Mat& src, Mat& dst);