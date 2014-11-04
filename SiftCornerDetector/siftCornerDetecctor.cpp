#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

// SurfDescriptorExtractor，另外需要链接opencv_nonfree249d.lib文件
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>

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

	keypoints.clear();
	// 创建surf特征检测对象
	SiftFeatureDetector sift(
		0.03,  // 特征阈值
		10.);  // 用于降低直线敏感度的阈值

	// 检测surf特征
	sift.detect(image, keypoints);
	Mat featureImage;
	drawKeypoints(image, keypoints, featureImage, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// 显示角点
	namedWindow("SIFT Features");
	imshow("SIFT Features", featureImage);

	waitKey();
	return 0;
}
