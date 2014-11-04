#ifndef CAMERACALIBRATOR_H
#define CAMERACALIBRATOR_H

#include <vector>
#include <iostream>

#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/highgui/highgui.hpp>

class CameraCalibrator
{
    // 输入点
	// 位于世界座标的点
    std::vector<std::vector<cv::Point3f>> objectPoints;

	// 像素座标点
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // 输出矩阵
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    // 标定方式
    int flag;

    // 用于图像去畸变
    cv::Mat map1, map2;
    bool mustInitUndistort; // 是否重新进行去畸变

public:
    CameraCalibrator() : flag(0), mustInitUndistort(true) {};

    // 打开棋盘图像并提取角点
    int addChessboardPoints(const std::vector<std::string> &filelist, cv::Size &boardSize);

    // 增加场景点和对应的图像点
    void addPoints(const std::vector<cv::Point2f> &imageCorners, const std::vector<cv::Point3f> &objectCorners);

    // 相机标定
    double calibrate(cv::Size &imageSize);

    // 设置相机标定方式
    void setCalibrationFlag(bool radial8CoeffEnabled = false, bool tangentialParamEnabled = false);

    // 标定后去除图像中的畸变
    cv::Mat CameraCalibrator::remap(const cv::Mat &image);

    // Getters
    cv::Mat getCameraMatrix()
    {
        return cameraMatrix;
    }
    cv::Mat getDistCoeffs()
    {
        return distCoeffs;
    }
};

#endif // CAMERACALIBRATOR_H
