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

#include "matcher.h"

using namespace std;
using namespace cv;

int main()
{
    // 输入图像
    Mat image1 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/parliament1.bmp", 0);
    Mat image2 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/parliament2.bmp", 0);
    if (!image1.data || !image2.data)
        return 0;

    // 显示图像
    namedWindow("Image 1");
    imshow("Image 1", image1);
    namedWindow("Image 2");
    imshow("Image 2", image2);

    // 准备匹配器
    RobustMatcher rmatcher;
    rmatcher.setConfidenceLevel(0.98);
    rmatcher.setMinDistanceToEpipolar(1.0);
    rmatcher.setRatio(0.65f);
    Ptr<FeatureDetector> pfd = new SurfFeatureDetector(10);
    rmatcher.setFeatureDetector(pfd);

    // 匹配两幅图像
    vector<DMatch> matches;
    vector<KeyPoint> keypoints1, keypoints2;
    Mat fundemental = rmatcher.match(image1, image2, matches, keypoints1, keypoints2);

    // 绘制匹配结果
    Mat imageMatches;
    drawMatches(image1, keypoints1,  // 第一幅图像及其关键点
                image2, keypoints2,  // 第二幅图像及其关键点
                matches,            // 匹配结果
                imageMatches,       // 生成的图像
                Scalar(255, 255, 255)); // 直线颜色
    namedWindow("Matches");
    imshow("Matches", imageMatches);

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

    // 绘制 inlier 点
    vector<Point2f>::const_iterator itPts = points1.begin();
    vector<uchar>::const_iterator itIn = inliers.begin();
    while (itPts != points1.end())
    {
        // 在每个 inlier 位置画圆
        if (*itIn)
            circle(image1, *itPts, 3, Scalar(255, 255, 255), 2);

        ++itPts;
        ++itIn;
    }

    itPts = points2.begin();
    itIn = inliers.begin();
    while (itPts != points2.end())
    {
        // 在每个 inlier 位置画圆
        if (*itIn)
            circle(image2, *itPts, 3, Scalar(255, 255, 255), 2);

        ++itPts;
        ++itIn;
    }

    // 显示带点图像
    namedWindow("Image 1 Homography Points");
    imshow("Image 1 Homography Points", image1);
    namedWindow("Image 2 Homography Points");
    imshow("Image 2 Homography Points", image2);

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

    waitKey();
    return 0;
}
