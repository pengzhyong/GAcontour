#include "SkeletonThin.h"



SkeletonThin::SkeletonThin()
{
}


SkeletonThin::~SkeletonThin()
{
}

void SkeletonThin::skeleton_Erode(Mat& src, Mat&  dst)
{

	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	//用三种不同的核进行实验，实验表明十字形核效果略好
	Mat kernel1 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat kernel2 = getStructuringElement(MORPH_CROSS, Size(3, 3));
	Mat kernel3 = (Mat_<uchar>(3, 3) << 1, 0, 1, 0, 1, 0, 1, 0, 1);
	Mat kernel4 = (Mat_<uchar>(2, 2) << 1, 1, 1, 1);//对于偶数行列的核，锚点不能取不到核的中心点，使得最后的骨架不在图像中心。
	double max;
	int k;
	Mat open_k, Sa[100];
	dst = Scalar(0, 0, 0);
	src.copyTo(open_k);
	for (k = 0; k < 100; k++)
	{
		erode(src, src, kernel2);
		morphologyEx(src, open_k, CV_MOP_OPEN, kernel2);
		Sa[k] = src - open_k;
		dst = dst + Sa[k];
		minMaxLoc(src, 0, &max);
		if (max == 0)break;
	}
}
void SkeletonThin::skeleton_Zhang(Mat& src, Mat& dst)
{
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	vector<Point> step1, step2;
	int i, j, n;
	int width, height;
	width = src.cols - 1;
	//之所以减1，是方便处理8邻域，防止越界
	height = src.rows - 1;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	//n表示迭代次数
	for (n = 0; ; n++)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1] > 0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1] > 0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1] > 0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step] > 0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1] > 0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1] > 0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1] > 0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step] > 0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7 && A1 == 1)
					{
						if ((p2 == 0 || p4 == 0 || p6 == 0) && (p4 == 0 || p6 == 0 || p8 == 0)) //p2*p4*p6=0 && p4*p6*p8==0
						{
							//dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							step1.push_back(Point(i, j));//---------------------
							ifEnd = true;
						}
					}
				}
			}
		}

		//将已经标记的像素置0
		vector<Point>::iterator it1;
		for (it1 = step1.begin(); it1 != step1.end(); it1++)
		{
			dst.at<uchar>(it1->x, it1->y) = 0;
		}

		//第二步
		dst.copyTo(tmpimg);
		img = tmpimg.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				A1 = 0;
				uchar* p = img + j;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1] > 0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1] > 0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1] > 0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step] > 0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1] > 0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1] > 0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1] > 0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step] > 0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7 && A1 == 1)
					{
						if ((p2 == 0 || p4 == 0 || p8 == 0) && (p2 == 0 || p6 == 0 || p8 == 0)) //p2*p4*p8=0 && p2*p6*p8==0
						{
							//dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							step2.push_back(Point(i, j));
							ifEnd = true;
						}
					}
				}
			}
		}

		//将标记的像素点置0
		vector<Point>::iterator it2;
		for (it2 = step2.begin(); it2 != step2.end(); it2++)
		{
			dst.at<uchar>(it2->x, it2->y) = 0;
		}

		//如果两个子迭代已经没有可以细化的像素了，则退出迭代
		if (!ifEnd)
		{
			printf("intera=%d\n", n);
			break;
		}
	}

}
void SkeletonThin::skeleton_Hidtch(Mat& src, Mat& dst)
{
	//http://cgm.cs.mcgill.ca/~godfried/teaching/projects97/azar/skeleton.html#algorithm
	//算法有问题，得不到想要的效果
	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j;
	int width, height;
	//之所以减2，是方便处理8邻域，防止越界
	width = src.cols - 2;
	height = src.rows - 2;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	int A1;
	cv::Mat tmpimg;
	while (1)
	{
		dst.copyTo(tmpimg);
		ifEnd = false;
		img = tmpimg.data + step;
		vector<Point> flag;
		for (i = 2; i < height; i++)
		{
			img += step;
			for (j = 2; j < width; j++)
			{
				uchar* p = img + j;
				A1 = 0;
				if (p[0] > 0)
				{
					if (p[-step] == 0 && p[-step + 1] > 0) //p2,p3 01模式
					{
						A1++;
					}
					if (p[-step + 1] == 0 && p[1] > 0) //p3,p4 01模式
					{
						A1++;
					}
					if (p[1] == 0 && p[step + 1] > 0) //p4,p5 01模式
					{
						A1++;
					}
					if (p[step + 1] == 0 && p[step] > 0) //p5,p6 01模式
					{
						A1++;
					}
					if (p[step] == 0 && p[step - 1] > 0) //p6,p7 01模式
					{
						A1++;
					}
					if (p[step - 1] == 0 && p[-1] > 0) //p7,p8 01模式
					{
						A1++;
					}
					if (p[-1] == 0 && p[-step - 1] > 0) //p8,p9 01模式
					{
						A1++;
					}
					if (p[-step - 1] == 0 && p[-step] > 0) //p9,p2 01模式
					{
						A1++;
					}
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					//计算AP2,AP4
					int A2, A4;
					A2 = 0;
					//if(p[-step]>0)
					{
						if (p[-2 * step] == 0 && p[-2 * step + 1] > 0) A2++;
						if (p[-2 * step + 1] == 0 && p[-step + 1] > 0) A2++;
						if (p[-step + 1] == 0 && p[1] > 0) A2++;
						if (p[1] == 0 && p[0] > 0) A2++;
						if (p[0] == 0 && p[-1] > 0) A2++;
						if (p[-1] == 0 && p[-step - 1] > 0) A2++;
						if (p[-step - 1] == 0 && p[-2 * step - 1] > 0) A2++;
						if (p[-2 * step - 1] == 0 && p[-2 * step] > 0) A2++;
					}


					A4 = 0;
					//if(p[1]>0)
					{
						if (p[-step + 1] == 0 && p[-step + 2] > 0) A4++;
						if (p[-step + 2] == 0 && p[2] > 0) A4++;
						if (p[2] == 0 && p[step + 2] > 0) A4++;
						if (p[step + 2] == 0 && p[step + 1] > 0) A4++;
						if (p[step + 1] == 0 && p[step] > 0) A4++;
						if (p[step] == 0 && p[0] > 0) A4++;
						if (p[0] == 0 && p[-step] > 0) A4++;
						if (p[-step] == 0 && p[-step + 1] > 0) A4++;
					}


					//printf("p2=%d p3=%d p4=%d p5=%d p6=%d p7=%d p8=%d p9=%d\n", p2, p3, p4, p5, p6,p7, p8, p9);
					//printf("A1=%d A2=%d A4=%d\n", A1, A2, A4);
					if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) > 1 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) < 7 && A1 == 1)
					{
						if (((p2 == 0 || p4 == 0 || p8 == 0) || A2 != 1) && ((p2 == 0 || p4 == 0 || p6 == 0) || A4 != 1))
						{
							//dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
							flag.push_back(Point(i, j));
							ifEnd = true;
							//printf("\n");

							//PrintMat(dst);
						}
					}
				}
			}
		}
		vector<Point>::iterator it;
		for (it = flag.begin(); it != flag.end(); it++)
		{
			dst.at<uchar>(it->x, it->y) = 0;
		}
		//printf("\n");
		//PrintMat(dst);
		//PrintMat(dst);
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}
}
void SkeletonThin::skeleton_Rosenfeld(cv::Mat& src, cv::Mat& dst)
{

	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	int i, j, n;
	int width, height;
	//之所以减1，是方便处理8邻域，防止越界
	width = src.cols - 1;
	height = src.rows - 1;
	int step = src.step;
	int  p2, p3, p4, p5, p6, p7, p8, p9;
	uchar* img;
	bool ifEnd;
	cv::Mat tmpimg;
	int dir[4] = { -step, step, 1, -1 };

	while (1)
	{
		//分四个子迭代过程，分别对应北，南，东，西四个边界点的情况
		vector<Point> flag;
		ifEnd = false;
		for (n = 0; n < 4; n++)
		{
			dst.copyTo(tmpimg);
			img = tmpimg.data;
			for (i = 1; i < height; i++)
			{
				img += step;
				for (j = 1; j < width; j++)
				{
					uchar* p = img + j;
					//如果p点是背景点或者且为方向边界点，依次为北南东西，继续循环
					if (p[0] == 0 || p[dir[n]] > 0) continue;
					p2 = p[-step] > 0 ? 1 : 0;
					p3 = p[-step + 1] > 0 ? 1 : 0;
					p4 = p[1] > 0 ? 1 : 0;
					p5 = p[step + 1] > 0 ? 1 : 0;
					p6 = p[step] > 0 ? 1 : 0;
					p7 = p[step - 1] > 0 ? 1 : 0;
					p8 = p[-1] > 0 ? 1 : 0;
					p9 = p[-step - 1] > 0 ? 1 : 0;
					//8 simple判定
					int is8simple = 1;
					if (p2 == 0 && p6 == 0)
					{
						if ((p9 == 1 || p8 == 1 || p7 == 1) && (p3 == 1 || p4 == 1 || p5 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p8 == 0)
					{
						if ((p9 == 1 || p2 == 1 || p3 == 1) && (p5 == 1 || p6 == 1 || p7 == 1))
							is8simple = 0;
					}
					if (p8 == 0 && p2 == 0)
					{
						if (p9 == 1 && (p3 == 1 || p4 == 1 || p5 == 1 || p6 == 1 || p7 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p2 == 0)
					{
						if (p3 == 1 && (p5 == 1 || p6 == 1 || p7 == 1 || p8 == 1 || p9 == 1))
							is8simple = 0;
					}
					if (p8 == 0 && p6 == 0)
					{
						if (p7 == 1 && (p3 == 9 || p2 == 1 || p3 == 1 || p4 == 1 || p5 == 1))
							is8simple = 0;
					}
					if (p4 == 0 && p6 == 0)
					{
						if (p5 == 1 && (p7 == 1 || p8 == 1 || p9 == 1 || p2 == 1 || p3 == 1))
							is8simple = 0;
					}
					int adjsum;
					adjsum = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
					//判断是否是邻接点或孤立点,0,1分别对于那个孤立点和端点
					if (adjsum != 1 && adjsum != 0 && is8simple == 1)
					{
						//dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
						flag.push_back(Point(i, j));
						ifEnd = true;
					}

				}
			}
		}

		vector<Point>::iterator it;
		for (it = flag.begin(); it != flag.end(); it++)
		{
			dst.at<uchar>(it->x, it->y) = 0;
		}
		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}

}
//Pavlidis不能执行
void SkeletonThin::skeleton_Pavlidis(cv::Mat& src, cv::Mat& dst)
{

	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	char erase, n[8];
	unsigned char bdr1, bdr2, bdr4, bdr5;
	short k, b;
	unsigned long i, j;


	int width, height;
	width = dst.cols;
	height = dst.rows;

	//把不能于0的值转化为1，便于后面处理
	for (i = 0; i < height; i++)
	{
		for (j = 0; j < width; j++)
		{
			if (dst.at<uchar>(i, j) != 0)
			{
				dst.at<uchar>(i, j) = 1;
			}
			//图像边框像素值为0
			if (i == 0 || i == (height - 1) || j == 0 || j == (width - 1))
				dst.at<uchar>(i, j) = 0;
		}
	}

	erase = 1;
	width = width - 1;
	height = height - 1;
	uchar* img;
	int step = dst.step;
	while (erase)
	{

		img = dst.data;
		//第一个循环，取得前景轮廓，轮廓用2表示
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;


				if (p[0] != 1)
					continue;

				n[0] = p[1];
				n[1] = p[-step + 1];
				n[2] = p[-step];
				n[3] = p[-step - 1];
				n[4] = p[-1];
				n[5] = p[step - 1];
				n[6] = p[step];
				n[7] = p[step + 1];

				//bdr1是2进制表示的p0...p6p7排列，10000011,p0=1,p6=p7=1
				bdr1 = 0;
				for (k = 0; k < 8; k++)
				{
					if (n[k] >= 1)
						bdr1 |= 0x80 >> k;
				}
				//内部点,p0, p2, p4, p6都是为1, 非边界点，所以继续循环
				//0xaa 10101010
				//  0   1   0   
				//  1         1
				//   0   1    0

				if ((bdr1 & 0xaa) == 0xaa)
					continue;
				//不是内部点，则是边界点，对于边界点，我们标记为2，是轮廓
				p[0] = 2;

				b = 0;

				for (k = 0; k <= 7; k++)
				{
					b += bdr1&(0x80 >> k);
				}
				//在边界点中，等于1，则是端点，等于0，则是孤立点，此时标记3
				if (b <= 1)
					p[0] = 3;

				//此条件说明p点是中间点，如果移去会引起断裂
				// 0x70        0x7         0x88      0xc1        0x1c      0x22      0x82     0x1      0xa0     0x40     0x28    0x10       0xa      0x4
				// 0 0 0     0  1  1     1  0   0    0   0   0    1  1  0    0   0   1  0  0  1  0 0 0    0  0  0   0 0 0    1  0  0   0  0  0  1  0  1   0 1 0
				// 1    0     0      1     0       0    0        1    1      0    0        0  0      0  0    1    0      0   0    0    0      0   1      0  0     0    0    0
				// 1 1 0     0  0  0     0  0   1    0   1   1    0  0  0    1   0   1  0  0  1  0 0 0    1  0  1   0 1 0    1  0  0   0  0  0  0  0  0   0 0 0
				if ((bdr1 & 0x70) != 0 && (bdr1 & 0x7) != 0 && (bdr1 & 0x88) == 0)
					p[0] = 3;
				else if ((bdr1 && 0xc1) != 0 && (bdr1 & 0x1c) != 0 && (bdr1 & 0x22) == 0)
					p[0] = 3;
				else if ((bdr1 & 0x82) == 0 && (bdr1 & 0x1) != 0)
					p[0] = 3;
				else if ((bdr1 & 0xa0) == 0 && (bdr1 & 0x40) != 0)
					p[0] = 3;
				else if ((bdr1 & 0x28) == 0 && (bdr1 & 0x10) != 0)
					p[0] = 3;
				else if ((bdr1 & 0xa) == 0 && (bdr1 & 0x4) != 0)
					p[0] = 3;

			}
		}
		//printf("------------------------------\n");
		//PrintMat(dst);
		img = dst.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;

				if (p[0] == 0)
					continue;

				n[0] = p[1];
				n[1] = p[-step + 1];
				n[2] = p[-step];
				n[3] = p[-step - 1];
				n[4] = p[-1];
				n[5] = p[step - 1];
				n[6] = p[step];
				n[7] = p[step + 1];

				bdr1 = bdr2 = 0;

				//bdr1是2进制表示的当前点p的8邻域连通情况，hdr2是当前点周围轮廓点的连接情况
				for (k = 0; k <= 7; k++)
				{
					if (n[k] >= 1)
						bdr1 |= 0x80 >> k;
					if (n[k] >= 2)
						bdr2 |= 0x80 >> k;
				}

				//相等，就是周围全是值为2的像素，继续
				if (bdr1 == bdr2)
				{
					p[0] = 4;
					continue;
				}

				//p0不为2，继续
				if (p[0] != 2) continue;
				//=4都是不可删除的轮廓点
				//     0x80       0xa     0x40        0x1      0x30   0x6
				//   0 0 0      1  0 1    0  0  0    0  0  0   0 0 0   0 1 1
				//   0    0      0     0    0      0    0      1   1    0   0    0
				//   0 0 1      0  0 0    0  1  0    0  0  0   1 0 0   0 0 0

				if (
					(bdr2 & 0x80) != 0 && (bdr1 & 0xa) == 0 &&
					//    ((bdr1&0x40)!=0 &&(bdr1&0x1)!=0 ||     ((bdr1&0x40)!=0 ||(bdr1 & 0x1)!=0) &&(bdr1&0x30)!=0 &&(bdr1&0x6)!=0 )
					(((bdr1 & 0x40) != 0 || (bdr1 & 0x1) != 0) && (bdr1 & 0x30) != 0 && (bdr1 & 0x6) != 0)
					)
				{
					p[0] = 4;
				}
				//
				else if ((bdr2 & 0x20) != 0 && (bdr1 & 0x2) == 0 &&
					//((bdr1&0x10)!=0 && (bdr1&0x40)!=0 || ((bdr1&0x10)!=0 || (bdr1&0x40)!=0) &&    (bdr1&0xc)!=0 && (bdr1&0x81)!=0)
					(((bdr1 & 0x10) != 0 || (bdr1 & 0x40) != 0) && (bdr1 & 0xc) != 0 && (bdr1 & 0x81) != 0)
					)
				{
					p[0] = 4;
				}

				else if ((bdr2 & 0x8) != 0 && (bdr1 & 0x80) == 0 &&
					//((bdr1&0x4)!=0 && (bdr1&0x10)!=0 || ((bdr1&0x4)!=0 || (bdr1&0x10)!=0) &&(bdr1&0x3)!=0 && (bdr1&0x60)!=0)
					(((bdr1 & 0x4) != 0 || (bdr1 & 0x10) != 0) && (bdr1 & 0x3) != 0 && (bdr1 & 0x60) != 0)
					)
				{
					p[0] = 4;
				}

				else if ((bdr2 & 0x2) != 0 && (bdr1 & 0x20) == 0 &&
					//((bdr1&0x1)!=0 && (bdr1&0x4)!=0 ||((bdr1&0x1)!=0 || (bdr1&0x4)!=0) &&(bdr1&0xc0)!=0 && (bdr1&0x18)!=0)
					(((bdr1 & 0x1) != 0 || (bdr1 & 0x4) != 0) && (bdr1 & 0xc0) != 0 && (bdr1 & 0x18) != 0)
					)
				{
					p[0] = 4;
				}
			}
		}
		//printf("------------------------------\n");
		//PrintMat(dst);
		img = dst.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;

				if (p[0] != 2)
					continue;


				n[0] = p[1];
				n[1] = p[-step + 1];
				n[2] = p[-step];
				n[3] = p[-step - 1];
				n[4] = p[-1];
				n[5] = p[step - 1];
				n[6] = p[step];
				n[7] = p[step + 1];

				bdr4 = bdr5 = 0;
				for (k = 0; k <= 7; k++)
				{
					if (n[k] >= 4)
						bdr4 |= 0x80 >> k;
					if (n[k] >= 5)
						bdr5 |= 0x80 >> k;
				}
				//值为4和5的像素
				if ((bdr4 & 0x8) == 0)
				{
					p[0] = 5;
					continue;
				}
				if ((bdr4 & 0x20) == 0 && bdr5 == 0)
				{
					p[0] = 5;
					continue;
				}

			}
		}
		erase = 0;
		//printf("------------------------------\n");
		//PrintMat(dst);
		img = dst.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;
				if (p[0] == 2 || p[0] == 5)
				{
					erase = 1;
					p[0] = 0;
				}
			}
		}
	}

}
//查找表不完善，结果不理想
void SkeletonThin::skeleton_LUT(cv::Mat& src, cv::Mat& dst)
{

	if (src.type() != CV_8UC1)
	{
		printf("只能处理二值或灰度图像\n");
		return;
	}
	//非原地操作时候，copy src到dst
	if (dst.data != src.data)
	{
		src.copyTo(dst);
	}

	//    P0 P1 P2
	//    P7    P3
	//    P6 P5 P4
	unsigned char deletemark[256] = {
		0,0,0,0,0,0,0,1,    0,0,1,1,0,0,1,1,
		0,0,0,0,0,0,0,0,    0,0,1,1,1,0,1,1,
		0,0,0,0,0,0,0,0,    1,0,0,0,1,0,1,1,
		0,0,0,0,0,0,0,0,    1,0,1,1,1,0,1,1,
		0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,    0,0,0,0,0,0,0,0,
		0,0,0,0,0,0,0,0,    1,0,0,0,1,0,1,1,
		1,0,0,0,0,0,0,0,    1,0,1,1,1,0,1,1,
		0,0,1,1,0,0,1,1,    0,0,0,1,0,0,1,1,
		0,0,0,0,0,0,0,0,    0,0,0,1,0,0,1,1,
		1,1,0,1,0,0,0,1,    0,0,0,0,0,0,0,0,
		1,1,0,1,0,0,0,1,    1,1,0,0,1,0,0,0,
		0,1,1,1,0,0,1,1,    0,0,0,1,0,0,1,1,
		0,0,0,0,0,0,0,0,    0,0,0,0,0,1,1,1,
		1,1,1,1,0,0,1,1,    1,1,0,0,1,1,0,0,
		1,1,1,1,0,0,1,1,    1,1,0,0,1,1,0,0
	};//索引
	int i, j;
	int width, height;
	//之所以减1，是方便处理8邻域，防止越界
	width = src.cols - 1;
	height = src.rows - 1;
	int step = src.step;
	int  p0, p1, p2, p3, p4, p5, p6, p7;
	uchar* img;
	bool ifEnd;
	bool border = false; //交换删除的次序，防止从一边细化
	while (1)
	{
		vector<Point> flag;
		border = !border;
		img = dst.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;
				//如果p点是背景点,继续循环
				if (p[0] == 0) continue;
				p0 = p[-step - 1] > 0 ? 1 : 0;
				p1 = p[-step] > 0 ? 1 : 0;
				p2 = p[-step + 1] > 0 ? 1 : 0;
				p3 = p[1] > 0 ? 1 : 0;
				p4 = p[step + 1] > 0 ? 1 : 0;
				p5 = p[step] > 0 ? 1 : 0;
				p6 = p[step - 1] > 0 ? 1 : 0;
				p7 = p[-1] > 0 ? 1 : 0;

				//如果sum等于0，则不是内部点，是轮廓点，设置其像素值为2
				int sum;
				sum = p0 & p1 & p2 & p3 & p4 & p5 & p6 & p7;

				//判断是否是邻接点或孤立点,0,1分别对于那个孤立点和端点
				if (sum == 0)
				{
					//dst.at<uchar>(i, j) = 4; //满足删除条件，设置当前像素为0
					flag.push_back(Point(i, j));
				}

			}
		}
		vector<Point>::iterator it;
		for (it = flag.begin(); it != flag.end(); it++)
		{
			dst.at<uchar>(it->x, it->y) = 0;
		}
		//执行删除操作
		ifEnd = false;

		img = dst.data;
		for (i = 1; i < height; i++)
		{
			img += step;
			for (j = 1; j < width; j++)
			{
				uchar* p = img + j;
				//如果p点是背景点,继续循环
				if (p[0] != 4) continue;
				p0 = p[-step - 1] > 0 ? 1 : 0;
				p1 = p[-step] > 0 ? 1 : 0;
				p2 = p[-step + 1] > 0 ? 1 : 0;
				p3 = p[1] > 0 ? 1 : 0;
				p4 = p[step + 1] > 0 ? 1 : 0;
				p5 = p[step] > 0 ? 1 : 0;
				p6 = p[step - 1] > 0 ? 1 : 0;
				p7 = p[-1] > 0 ? 1 : 0;

				p1 = p1 << 1;
				p2 = p2 << 2;
				p3 = p3 << 3;
				p4 = p4 << 4;
				p5 = p5 << 5;
				p6 = p6 << 6;
				p7 = p7 << 7;

				//求的8邻域在索引表中的索引
				int sum;
				sum = p0 | p1 | p2 | p3 | p4 | p5 | p6 | p7;

				//判断是否是邻接点或孤立点,0,1分别对于那个孤立点和端点
				if (deletemark[sum] == 1)
				{
					dst.at<uchar>(i, j) = 0; //满足删除条件，设置当前像素为0
					ifEnd = true;
				}

			}
		}

		//printf("\n");
		//PrintMat(dst);
		//printf("\n");

		//已经没有可以细化的像素了，则退出迭代
		if (!ifEnd) break;
	}
}
