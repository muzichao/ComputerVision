#include <iostream>
#include <iomanip>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "CameraCalibrator.h"

using namespace std;
using namespace cv;

int main()
{
    namedWindow("Image");
    Mat image;
    vector<string> filelist; // 文件名列表

    // 产生棋盘图像文件名列表
    for (int i = 1; i <= 20; i++)
    {
        stringstream str;
		// 产生文件名字符串
        str << "E:/桌面资料/编程/openCV/opencv-2-cookbook-src-master/images/chessboards/chessboard" << setw(2) << setfill('0') << i << ".jpg";
        cout << str.str() << endl;

        filelist.push_back(str.str()); // 把字符串str.str()压入容器
        image = imread(str.str(), 0); // 显示文件
        imshow("Image", image);

        waitKey(100);
    }

    // 创建标定对象
    CameraCalibrator cameraCalibrator;

    // 棋盘角点（不包含边缘）
    Size boardSize(6, 4);

	// filelist : 文件名列表
	// boardSize : 棋盘角点个数
	// 打开棋盘图像并提取角点
    cameraCalibrator.addChessboardPoints(filelist, boardSize);

    // 相机标定
    //  cameraCalibrator.setCalibrationFlag(true,true);
    cameraCalibrator.calibrate(image.size());

    // 图像去畸变
    image = imread(filelist[6]);
    Mat uImage = cameraCalibrator.remap(image);

    // 显示相机矩阵
    Mat cameraMatrix = cameraCalibrator.getCameraMatrix();
    cout << " Camera intrinsic: " << cameraMatrix.rows << "x" << cameraMatrix.cols << endl;
    cout << cameraMatrix.at<double>(0, 0) << " " << cameraMatrix.at<double>(0, 1) << " " << cameraMatrix.at<double>(0, 2) << endl;
    cout << cameraMatrix.at<double>(1, 0) << " " << cameraMatrix.at<double>(1, 1) << " " << cameraMatrix.at<double>(1, 2) << endl;
    cout << cameraMatrix.at<double>(2, 0) << " " << cameraMatrix.at<double>(2, 1) << " " << cameraMatrix.at<double>(2, 2) << endl;

    imshow("Original Image", image);
    imshow("Undistorted Image", uImage);

    waitKey();
    return 0;
}
