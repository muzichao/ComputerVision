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
    // 读取输入图像
    Mat image1 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/canal1.jpg", 0);
    Mat image2 = imread("E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/canal2.jpg", 0);
    if (!image1.data || !image2.data)
        return 0;

    // 显示输入图像
    namedWindow("Right Image");
    imshow("Right Image", image1);
    namedWindow("Left Image");
    imshow("Left Image", image2);

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
    drawMatches(image1, keypoints1, // 第一幅图像及其关键点
                image2, keypoints2, // 第二幅图像及其关键点
                matches,            // 匹配结果
                imageMatches,       // 生成的图像
                Scalar(255, 255, 255)); // 直线颜色
    namedWindow("Matches");
    imshow("Matches", imageMatches);

    // 把 keypoints 转换为 Point2f
    vector<Point2f> points1, points2;

    for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
    {
        // 获得左侧 keypoints 的位置
        float x = keypoints1[it->queryIdx].pt.x;
        float y = keypoints1[it->queryIdx].pt.y;
        points1.push_back(Point2f(x, y));
        circle(image1, Point(x, y), 3, Scalar(255, 255, 255), 3);
        // 获得右侧 keypoints 的位置
        x = keypoints2[it->trainIdx].pt.x;
        y = keypoints2[it->trainIdx].pt.y;
        circle(image2, Point(x, y), 3, Scalar(255, 255, 255), 3);
        points2.push_back(Point2f(x, y));
    }

    // 绘制极线
    vector<Vec3f> lines1;
    computeCorrespondEpilines(Mat(points1), 1, fundemental, lines1);

    for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end(); ++it)
    {
        line(image2, Point(0, -(*it)[2] / (*it)[1]),
             Point(image2.cols, -((*it)[2] + (*it)[0]*image2.cols) / (*it)[1]),
             Scalar(255, 255, 255));
    }

    vector<Vec3f> lines2;
    computeCorrespondEpilines(Mat(points2), 2, fundemental, lines2);

    for (vector<Vec3f>::const_iterator it = lines2.begin(); it != lines2.end(); ++it)
    {
        line(image1, Point(0, -(*it)[2] / (*it)[1]),
             Point(image1.cols, -((*it)[2] + (*it)[0]*image1.cols) / (*it)[1]),
             Scalar(255, 255, 255));
    }

    // 显示带极线的图像
    namedWindow("Right Image Epilines (RANSAC)");
    imshow("Right Image Epilines (RANSAC)", image1);
    namedWindow("Left Image Epilines (RANSAC)");
    imshow("Left Image Epilines (RANSAC)", image2);

    waitKey();
    return 0;
}
