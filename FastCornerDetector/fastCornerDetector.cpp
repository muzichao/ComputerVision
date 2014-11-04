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
	{
		return 0;
	}

	// 显示图像
	namedWindow("Original Image");
	imshow("Original Image", image);

	// 关键点容器
	vector<KeyPoint> keypoints;
	FastFeatureDetector fast(40);
	fast.detect(image, keypoints);

	drawKeypoints(image, keypoints, image, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_OVER_OUTIMG);

	// 显示角点
	namedWindow("FAST Features");
	imshow("FAST Features", image);

	waitKey();
	return 0;
}
