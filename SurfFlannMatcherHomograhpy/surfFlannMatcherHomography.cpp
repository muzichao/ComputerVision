#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/features2d.hpp"

// warpPerspective 头文件
#include<opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat image1 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/parliament1.bmp", 0);
	Mat image2 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/parliament2.bmp", 0);
	if (!image1.data || !image2.data)
	{
		cout << " --(!) Error reading images " << endl; return -1;
	}

	//-- Step 1: 用 SURF 法检测特征关键点
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	vector<KeyPoint> keypoints1, keypoints2;
	detector.detect(image1, keypoints1);
	detector.detect(image2, keypoints2);

	//-- Step 2: 提取特征描述子
	SurfDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(image1, keypoints1, descriptors1);
	extractor.compute(image2, keypoints2, descriptors2);

	//-- Step 3: 用 FLANN 法匹配特征描述子
	FlannBasedMatcher matcher;
	vector< DMatch > matches;
	matcher.match(descriptors1, descriptors2, matches);

	//-- 快速计算特征点之间的最大和最小距离
	double max_dist = 0; double min_dist = 100;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//-- 绘制优质匹配，比如距离小于2倍最小距离的匹配，或者当最小距离非常
	//-- 小的时候，使用一个小的任意值（0.02）
	//-- PS.- 也可以使用 radiusMatch .
	vector< DMatch > good_matches;
	for (int i = 0; i < descriptors1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}

	//-- 仅绘制优质匹配
	Mat imagematches;
	drawMatches(image1, keypoints1, image2, keypoints2,
		good_matches,
		imagematches,
		Scalar::all(-1),
		Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//-- 显示检测到的匹配
	imshow("Good Matches", imagematches);
	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d -- Keypoint 2: %d \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}

	// 把 keypoints 转换为 Point2f
	vector<Point2f> points1, points2;
	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		// /获得左侧 keypoints 的位置
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(Point2f(x, y));
		// 获得右侧 keypoints 的位置
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(Point2f(x, y));
	}
	cout << points1.size() << " " << points2.size() << endl;

	// 找到两幅图像之间的单应矩阵
	vector<uchar> inliers(points1.size(), 0);
	Mat homography = findHomography(
		Mat(points1), Mat(points2), // 对应的点集
		inliers,    // 输出的正确值
		CV_RANSAC,  // RANSAC 法
		1.);        // 到反投影点的最大距离

	// 歪曲图1 到 图2
	Mat result;
	warpPerspective(image1, // 输入图像
		result,         // 输出图像
		homography,     // homography
		Size(2 * image1.cols, image1.rows)); // 输出图像大小

	// 复制图1到整幅图像的前半部份
	Mat half(result, Rect(0, 0, image2.cols, image2.rows));
	image2.copyTo(half);// 复制图2到图1的ROI区域

	// 显示歪曲的图像
	namedWindow("After warping");
	imshow("After warping", result);

	waitKey(0);

	return 0;
}

