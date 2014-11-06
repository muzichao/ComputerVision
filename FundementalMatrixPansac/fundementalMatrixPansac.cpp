#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

// SurfDescriptorExtractor，另外需要链接opencv_nonfree249d.lib文件
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>

// BruteForceMatcher，另外需要链接opencv_legacy249d.lib文件
#include<opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

int main()
{
	// 读取输入图像
	Mat image1 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church01.jpg", 0);
	Mat image2 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church03.jpg", 0);
	if (!image1.data || !image2.data)
		return 0;

	// 显示输入图像
	namedWindow("Right Image");
	imshow("Right Image", image1);
	namedWindow("Left Image");
	imshow("Left Image", image2);

	// 关键点容器
	vector<KeyPoint> keypoints1;
	vector<KeyPoint> keypoints2;

	// 构造 surf 特征检测器
	SurfFeatureDetector surf(2500);

	// 检测 surf 特征
	surf.detect(image1, keypoints1);
	surf.detect(image2, keypoints2);

	cout << "Number of SURF points (1): " << keypoints1.size() << endl;
	cout << "Number of SURF points (2): " << keypoints2.size() << endl;

	// 绘制keypoints
	Mat imageKP;
	drawKeypoints(image1, keypoints1, imageKP, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	namedWindow("Right SURF Features");
	imshow("Right SURF Features", imageKP);

	drawKeypoints(image2, keypoints2, imageKP, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	namedWindow("Left SURF Features");
	imshow("Left SURF Features", imageKP);

	// 构造 surf 特征描述子提取器
	SurfDescriptorExtractor surfDesc;

	// 提取 surf 特征描述子
	Mat descriptors1, descriptors2;
	surfDesc.compute(image1, keypoints1, descriptors1);
	surfDesc.compute(image2, keypoints2, descriptors2);

	cout << "descriptor matrix size: " << descriptors1.rows << " by " << descriptors1.cols << endl;

	// 构造匹配器
	BruteForceMatcher<L2<float>> matcher;

	// 匹配两幅图像的描述子
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);

	cout << "Number of matched points: " << matches.size() << endl;

	
	nth_element(matches.begin(),    // 初始位置
	matches.begin() + matches.size()/4, // 中间位置
	matches.end());     // 结束位置
	// 删除中间位置之后的所有值
	matches.erase(matches.begin() + matches.size()/4, matches.end());
	

	// 绘制选择的匹配
	Mat imageMatches;
	drawMatches(image1, keypoints1,  // 第一幅图及其关键点
		image2, keypoints2,  // 第二幅图及其关键点
		//selMatches,           // the matches
		matches,            // 匹配结果
		imageMatches,       // 生成的图像
		Scalar(255, 255, 255)); // 直线颜色

	namedWindow("Matches");
	imshow("Matches", imageMatches);

	// 把一个 keypoints 向量转换为两个 Point2f 向量
	vector<Point2f> points1, points2;
	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		// 得到左侧关键点的位置
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(Point2f(x, y));

		// 得到右侧关键点的位置
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(Point2f(x, y));
	}
	cout << points1.size() << " " << points2.size() << endl;

	// 用 RANSAC 计算 F 矩阵
	vector<uchar> inliers(points1.size(), 0);
	Mat fundemental = findFundamentalMat(
		Mat(points1), Mat(points2), // 匹配点
		inliers,      // 匹配状态 (inlier ou outlier)
		CV_FM_RANSAC, // RANSAC 法
		1,            // 到极线的距离
		0.98);        // 置信概率

	// 读取图像
	image1 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church01.jpg", 0);
	image2 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church03.jpg", 0);

	// 将一个 keypoints 向量转换为两个 int 向量
	vector<int> pointIndexes1;
	vector<int> pointIndexes2;


	// 转换 keypoints 类型为 Point2f
	vector<Point2f> selPoints1, selPoints2;
	KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
	KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

	// 绘制少量点的极线
	vector<Vec3f> lines1;
	computeCorrespondEpilines(Mat(selPoints1), 1, fundemental, lines1);
	for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
	{
		line(image2, Point(0, -(*it)[2] / (*it)[1]),
			Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
			Scalar(255, 255, 255));
	}

	vector<Vec3f> lines2;
	computeCorrespondEpilines(Mat(selPoints2), 2, fundemental, lines2);
	for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it)
	{
		line(image1, Point(0, -(*it)[2] / (*it)[1]),
			Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
			Scalar(255, 255, 255));
	}

	// 绘制 inlier 点
	vector<Point2f> points1In, points2In;

	vector<Point2f>::const_iterator itPts = points1.begin();
	vector<uchar>::const_iterator itIn = inliers.begin();
	while (itPts != points1.end())
	{
		// 在每一个 inlier 点上绘制一个圆
		if (*itIn)
		{
			circle(image1, *itPts, 3, Scalar(255, 255, 255), 2);
			points1In.push_back(*itPts);
		}
		++itPts;
		++itIn;
	}

	itPts = points2.begin();
	itIn = inliers.begin();
	while (itPts != points2.end())
	{
		// 在每一个 inlier 点上绘制一个圆
		if (*itIn)
		{
			circle(image2, *itPts, 3, Scalar(255, 255, 255), 2);
			points2In.push_back(*itPts);
		}
		++itPts;
		++itIn;
	}

	// 显示图像和点
	namedWindow("Right Image Epilines (RANSAC)");
	imshow("Right Image Epilines (RANSAC)", image1);
	namedWindow("Left Image Epilines (RANSAC)");
	imshow("Left Image Epilines (RANSAC)", image2);


	waitKey();
	return 0;
}
