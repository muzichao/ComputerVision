#if !defined HARRISD
#define HARRISD

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

class HarrisDetector
{
private:

	// 表示角点强度的32位浮点图像
	cv::Mat cornerStrength;
	// 表示阈值化后角点强度的32位浮点图像
	cv::Mat cornerTh;
	// 局部极大值图像（内部）
	cv::Mat localMax;
	// 导数平滑的相邻像素的尺寸
	int neighbourhood;
	// 梯度计算的孔径大小
	int aperture;
	// Harris 参数
	double k;
	// 阈值计算的最大强度
	double maxStrength;
	// 计算得到的阈值（内部）
	double threshold;
	// 非极大值抑制的相邻像素的尺寸
	int nonMaxSize;
	// 非极大值抑制的核
	cv::Mat kernel;

public:

	HarrisDetector() : neighbourhood(3), aperture(3), k(0.1), maxStrength(0.0), threshold(0.01), nonMaxSize(3)
	{
		setLocalMaxWindowSize(nonMaxSize);
	}

	// 创建非极大值抑制的核
	void setLocalMaxWindowSize(int size)
	{
		nonMaxSize = size;
		kernel.create(nonMaxSize, nonMaxSize, CV_8U);
	}

	// 计算 Harris 角点 
	void detect(const cv::Mat& image)
	{
		// Harris 计算
		cv::cornerHarris(image, cornerStrength,
			neighbourhood,// 邻域大小
			aperture,     // 滤波器的孔径大小
			k);           // Harris 参数

		// 内部阈值计算
		double minStrength; // 未使用
		cv::minMaxLoc(cornerStrength, &minStrength, &maxStrength);

		// 局部极大值检测
		cv::Mat dilated;  // 临时图像
		cv::dilate(cornerStrength, dilated, cv::Mat());
		cv::compare(cornerStrength, dilated, localMax, cv::CMP_EQ);
	}

	// 由 Harris 值获取角点图
	cv::Mat getCornerMap(double qualityLevel)
	{
		cv::Mat cornerMap;

		// 对角点图像进行阈值化
		threshold = qualityLevel*maxStrength;
		cv::threshold(cornerStrength, cornerTh, threshold, 255, cv::THRESH_BINARY);

		// 转换为8位图像
		cornerTh.convertTo(cornerMap, CV_8U);

		// 非极大值抑制
		cv::bitwise_and(cornerMap, localMax, cornerMap);

		return cornerMap;
	}

	// 从计算得到的 Harris 值得到角点
	void getCorners(std::vector<cv::Point> &points, double qualityLevel)
	{
		// 得到角点映射
		cv::Mat cornerMap = getCornerMap(qualityLevel);
		// 得到角点
		getCorners(points, cornerMap);
	}

	// 由计算得到的角点映射获取特征点
	void getCorners(std::vector<cv::Point> &points, const cv::Mat& cornerMap)
	{
		// 遍历像素得到所有特征
		for (int y = 0; y < cornerMap.rows; y++) 
		{
			const uchar* rowPtr = cornerMap.ptr<uchar>(y);

			for (int x = 0; x < cornerMap.cols; x++) 
			{
				// 如果是特征点
				if (rowPtr[x]) 
				{
					points.push_back(cv::Point(x, y));
				}
			}
		}
	}

	// 在特征点的位置绘制圆
	void drawOnImage(cv::Mat &image, const std::vector<cv::Point> &points, cv::Scalar color = cv::Scalar(255, 255, 255), int radius = 3, int thickness = 2)
	{
		std::vector<cv::Point>::const_iterator it = points.begin();

		// 遍历所有特征点
		while (it != points.end()) 
		{
			// 绘制圆
			cv::circle(image, *it, radius, color, thickness);
			++it;
		}
	}
};

#endif
