#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

// SurfDescriptorExtractor，另外需要链接opencv_nonfree249d.lib文件
#include<opencv2/nonfree/features2d.hpp>
#include<opencv2/nonfree/nonfree.hpp>

// BruteForceMatcher，另外需要链接opencv_legacy249d.lib文件
#include<opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;
int main()
{
    // 读入图像
    Mat image1 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church01.jpg", 0);
    Mat image2 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/church02.jpg", 0);
    if (!image1.data || !image2.data)
        return 0;

    // 显示图像
    namedWindow("Right Image");
    imshow("Right Image", image1);
    namedWindow("Left Image");
    imshow("Left Image", image2);

    // 关键点容器
    vector<KeyPoint> keypoints1;
    vector<KeyPoint> keypoints2;

    // 创建surf特征检测对象
    SurfFeatureDetector surf(3000);

    // 检测surf特征
    surf.detect(image1, keypoints1);
    surf.detect(image2, keypoints2);

    cout << "Number of SURF points (1): " << keypoints1.size() << endl;
    cout << "Number of SURF points (2): " << keypoints2.size() << endl;

    // 绘制关键点
    Mat imageKP;
    drawKeypoints(image1, keypoints1, imageKP, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	// 显示关键点图像
    namedWindow("Right SURF Features");
    imshow("Right SURF Features", imageKP);

    drawKeypoints(image2, keypoints2, imageKP, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    namedWindow("Left SURF Features");
    imshow("Left SURF Features", imageKP);

    // 构造surf描述子提取器
    SurfDescriptorExtractor surfDesc;

    // 提取surf描述子
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

	// 第n个元素放在第n个位置，之前都是小于第n个元素，之后都大于
    nth_element(matches.begin(),    // 初始位置
                     matches.begin() + 24, // 排序元素的位置
                     matches.end());     // 终止位置


    // 移除第25位之后所有的元素
    matches.erase(matches.begin() + 25, matches.end());

    Mat imageMatches;
    drawMatches(image1, keypoints1,  // 第一幅图像及其特征点
                    image2, keypoints2,  // 第二幅图像及其特征点
                    matches,            // 匹配结果
                    imageMatches,       // 生成的图像
                    Scalar(255, 255, 255)); // 直线的颜色

	// 显示匹配图像
    namedWindow("Matches");
    imshow("Matches", imageMatches);

    waitKey();
    return 0;
}
