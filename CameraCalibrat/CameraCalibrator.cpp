#include "CameraCalibrator.h"

// 打开棋盘图像并提取角点
int CameraCalibrator::addChessboardPoints(
    const std::vector<std::string> &filelist,
    cv::Size &boardSize)
{
    // 棋盘上的点的两种坐标
    std::vector<cv::Point2f> imageCorners;
    std::vector<cv::Point3f> objectCorners;

    // 3D场景中的点:
    // 在棋盘坐标系中初始化棋盘角点
    // 这些点位于 (X,Y,Z)= (i,j,0)
    for (int i = 0; i < boardSize.height; i++)
    {
        for (int j = 0; j < boardSize.width; j++)
        {
            objectCorners.push_back(cv::Point3f(i, j, 0.0f));
        }
    }

    // 2D图像中的点:
    cv::Mat image; // 用来保存棋盘图像
    int successes = 0;

    // 循环所有图片
    for (int i = 0; i < filelist.size(); i++)
    {
        // 打开图像
        image = cv::imread(filelist[i], 0);

        // 得到棋盘角点
        bool found = cv::findChessboardCorners(image, boardSize, imageCorners);

        // 获取亚像素精度
		// 利用迭代法提高精度
        cv::cornerSubPix(image, imageCorners,
                         cv::Size(5, 5), // 搜索窗口的一半大小
                         cv::Size(-1, -1), // 死区的一半大小，(-1, -1)表示没有死区
                         cv::TermCriteria(cv::TermCriteria::MAX_ITER +
                                          cv::TermCriteria::EPS,
                                          30,        // 最大迭代数目
                                          0.1));     // 最小精度

        // 如果角点数目满足要要求，那么将它加入数据
        if (imageCorners.size() == boardSize.area())
        {
            // 添加一个视角中的图像点及场景点
            addPoints(imageCorners, objectCorners);
            successes++;
        }

        // 绘制角点
		// found 已经找到角点
        cv::drawChessboardCorners(image, boardSize, imageCorners, found);

		// 显示
        cv::imshow("Corners on Chessboard", image);
        cv::waitKey(100);
    }

    return successes;
}

// 添加场景点与对应的图像点
void CameraCalibrator::addPoints(const std::vector<cv::Point2f> &imageCorners, const std::vector<cv::Point3f> &objectCorners)
{
    // 2D图像点
    imagePoints.push_back(imageCorners);

    // 对应3D场景中的点
    objectPoints.push_back(objectCorners);
}

// 进行标定，返回重投影误差
// 计算了相机内参矩阵(cameraMatrix)
// 计算了畸变系数(distCoeffs)
// 计算了旋转矩阵(rvecs)
// 计算了平移向量(tvecs)
double CameraCalibrator::calibrate(cv::Size &imageSize)
{
    // 必须重新进行去畸变
    mustInitUndistort = true;

    //输出旋转和平移
    std::vector<cv::Mat> rvecs, tvecs;

    // 开始标定
    return
        calibrateCamera(objectPoints, // 3D点
                        imagePoints,  // 图像点
                        imageSize,    // 图像尺寸
                        cameraMatrix, // 输出的相机矩阵
                        distCoeffs,   // 畸变系数
                        rvecs, tvecs, // 旋转和平移
                        flag);        // 额外选项
    //                  ,CV_CALIB_USE_INTRINSIC_GUESS);

}

// 标定后去除图像中的畸变
cv::Mat CameraCalibrator::remap(const cv::Mat &image)
{
    cv::Mat undistorted;

    if (mustInitUndistort)   // 每次标定只需要初始化一次
    {
        cv::initUndistortRectifyMap(
            cameraMatrix,  // calibrateCamera中计算得到的相机矩阵
            distCoeffs,    // calibrateCamera中计算得到的畸变矩阵
            cv::Mat(),     // 可选的校正矩阵(此处为空)
            cv::Mat(),     // 用于生成undistorted对象的相机矩阵
            cv::Size(640, 480), // image.size(), undistorted对象的尺寸
            CV_32FC1,      // 输出映射图像的类型
            map1, map2);   // x坐标和y坐标映射函数

        mustInitUndistort = false;
    }

    // 应用映射函数
    cv::remap(image, undistorted, map1, map2,
              cv::INTER_LINEAR); // 插值类型

    return undistorted;
}


// Set the calibration options
// 8radialCoeffEnabled should be true if 8 radial coefficients are required (5 is default)
// tangentialParamEnabled should be true if tangeantial distortion is present
void CameraCalibrator::setCalibrationFlag(bool radial8CoeffEnabled, bool tangentialParamEnabled)
{
    // Set the flag used in cv::calibrateCamera()
    flag = 0;
    if (!tangentialParamEnabled) flag += CV_CALIB_ZERO_TANGENT_DIST;
    if (radial8CoeffEnabled) flag += CV_CALIB_RATIONAL_MODEL;
}

