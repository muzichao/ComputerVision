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

	// 选择少量匹配结果 
	vector<DMatch> selMatches;
	/*
	keypoints1.push_back(KeyPoint(342.,615.,2));
	keypoints2.push_back(KeyPoint(410.,600.,2));
	selMatches.push_back(DMatch(keypoints1.size()-1,keypoints2.size()-1,0)); // street light bulb
	selMatches.push_back(matches[6]);  // right tower
	selMatches.push_back(matches[60]);  // left bottom window
	selMatches.push_back(matches[139]);
	selMatches.push_back(matches[141]);  // middle window
	selMatches.push_back(matches[213]);
	selMatches.push_back(matches[273]);

	int kk=0;
	while (kk<matches.size()) 
	{
	cout<<kk<<endl;
	selMatches.push_back(matches[kk++]);
	selMatches.pop_back();
	waitKey();
	}
	*/

	cout << matches.size() << endl;

	/* between church01 and church03 */
	selMatches.push_back(matches[14]);
	selMatches.push_back(matches[16]);
	selMatches.push_back(matches[141]);
	selMatches.push_back(matches[146]);
	selMatches.push_back(matches[235]);
	selMatches.push_back(matches[238]);
	selMatches.push_back(matches[273]); //vector subscript out of range 274

	// 绘制选择的匹配
	Mat imageMatches;
	drawMatches(image1, keypoints1,  // 第一幅图及其关键点
		image2, keypoints2,  // 第二幅图及其关键点
		//selMatches,			// the matches
		matches,			// 匹配结果
		imageMatches,		// 生成的图像
		Scalar(255, 255, 255)); // 直线颜色

	namedWindow("Matches");
	imshow("Matches", imageMatches);

	// 将一个 keypoints 向量转换为两个 Point2f 向量
	vector<int> pointIndexes1;
	vector<int> pointIndexes2;
	for (vector<DMatch>::const_iterator it = selMatches.begin(); it != selMatches.end(); ++it) 
	{
		// 得到选择的匹配点的索引
		pointIndexes1.push_back(it->queryIdx);
		pointIndexes2.push_back(it->trainIdx);
	}

	// 转换 keypoints 类型为 Point2f
	vector<Point2f> selPoints1, selPoints2;
	KeyPoint::convert(keypoints1, selPoints1, pointIndexes1);
	KeyPoint::convert(keypoints2, selPoints2, pointIndexes2);

	// check by drawing the points 
	vector<Point2f>::const_iterator it = selPoints1.begin();
	while (it != selPoints1.end()) 
	{
		// 在每一个角点位置画一个圈
		circle(image1, *it, 3, Scalar(255, 255, 255), 2);
		++it;
	}

	it = selPoints2.begin();
	while (it != selPoints2.end()) 
	{
		// 在每一个角点位置画一个圈
		circle(image2, *it, 3, Scalar(255, 255, 255), 2);
		++it;
	}

	// 从7个匹配中计算F矩阵
	Mat fundemental = findFundamentalMat(
		Mat(selPoints1), // 图1中的点
		Mat(selPoints2), // 图2中的点
		CV_FM_7POINT);   // 使用7个点的方法

	cout << "F-Matrix size= " << fundemental.rows << "," << fundemental.cols << endl;

	// 在右图中绘制对应的极线
	vector<Vec3f> lines1;
	computeCorrespondEpilines(
		Mat(selPoints1), // 图像点
		1,               // 图1（也可以是2）
		fundemental,     // F矩阵
		lines1);         // 一组极线

	// 对于所有极线
	for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it) 
	{
		// 绘制第一列与最后一列之间的极线
		line(image2, Point(0, -(*it)[2] / (*it)[1]),
			Point(image2.cols, -((*it)[2] + (*it)[0] * image2.cols) / (*it)[1]),
			Scalar(255, 255, 255));
	}

	// 在左图中绘制对应的极线
	vector<Vec3f> lines2;
	computeCorrespondEpilines(Mat(selPoints2), 2, fundemental, lines2);

	for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it) 
	{
		// 绘制第一列与最后一列之间的极线
		line(image1, Point(0, -(*it)[2] / (*it)[1]),
			Point(image1.cols, -((*it)[2] + (*it)[0] * image1.cols) / (*it)[1]),
			Scalar(255, 255, 255));
	}

	// 显示图像以及图像中的点和极线
	namedWindow("Right Image Epilines");
	imshow("Right Image Epilines", image1);
	namedWindow("Left Image Epilines");
	imshow("Left Image Epilines", image2);

	waitKey();
	return 0;
}