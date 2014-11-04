#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

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

	// 关键点容器
	vector<KeyPoint> keypoints;

	MserFeatureDetector mser;
	mser.detect(image, keypoints);

	Mat featureImage;
	// 绘制特征点，加上尺度和方向信息
	drawKeypoints(image,        // 原始图像
		keypoints,                  // 关键点容器
		featureImage,               // 生成图像
		Scalar(255, 255, 255),  // 特征点颜色
		DrawMatchesFlags::DRAW_RICH_KEYPOINTS); // 绘制标志

	// 显示角点
	namedWindow("MSER Features");
	imshow("MSER Features", featureImage);

	waitKey();
	return 0;
}
