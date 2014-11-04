#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "harrisDetector.h"

using namespace std;
using namespace cv;

int main()
{
	// 读取图像
	Mat image = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church01.jpg", 0);
	if (!image.data)
		return 0;

	// 显示图像
	namedWindow("Original Image");
	imshow("Original Image", image);

	// 检测 Harris 角点
	Mat cornerStrength;
	cornerHarris(image, cornerStrength,
		3,     // 相邻像素尺寸
		3,     // 滤波器的孔径大小
		0.01); // Harris 参数

	// 角点强度的阈值
	Mat harrisCorners;
	double thresh = 0.0001;
	threshold(cornerStrength, harrisCorners, thresh, 255, THRESH_BINARY_INV);

	// 显示角点图
	namedWindow("Harris Corner Map");
	imshow("Harris Corner Map", harrisCorners);

	// 创建 Harris 检测对象
	HarrisDetector harris;
	// 计算 Harris 值 
	harris.detect(image);
	// 检测 Harris 角点
	vector<Point> pts;
	harris.getCorners(pts, 0.01);
	// 绘制 Harris 角点
	harris.drawOnImage(image, pts);

	// 显示角点
	namedWindow("Harris Corners");
	imshow("Harris Corners", image);

	// 读取图像
	image = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church01.jpg", 0);

	// 计算适合追踪的优质特征
	vector<Point2f> corners;
	goodFeaturesToTrack(image, corners,
		500,	// 返回的最大特征点数目
		0.01,	// 质量等级
		10);	// 两点之间的最小运行距离

	// 遍历所有角点
	vector<Point2f>::const_iterator it = corners.begin();
	while (it != corners.end()) 
	{
		// 在每一个角点位置画一个圆
		circle(image, *it, 3, Scalar(255, 255, 255), 2);
		++it;
	}

	// 显示角点
	namedWindow("Good Features to Track");
	imshow("Good Features to Track", image);

	// 读取图像
	image = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church01.jpg", 0);

	// 关键点容器
	vector<KeyPoint> keypoints;
	// 计算适合追踪的优质特征 
	GoodFeaturesToTrackDetector gftt(
		500,	// 返回的最大特征点数目
		0.01,	// 质量等级
		10);	// 两点之间的最小运行距离
	// 用特征检测法进行关键点检测
	gftt.detect(image, keypoints);

	drawKeypoints(image,		// 原始图像
		keypoints,					// 关键点容器
		image,						// 结果图像
		Scalar(255, 255, 255),	// 关键点颜色
		DrawMatchesFlags::DRAW_OVER_OUTIMG); // 绘制标志

	// 显示角点
	namedWindow("Good Features to Track Detector");
	imshow("Good Features to Track Detector", image);

	//keypoints.clear();

	waitKey();
	return 0;
}