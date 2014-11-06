#if !defined MATCHER
#define MATCHER

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

class RobustMatcher
{

private:

	// 指向特征检测器的指针
	cv::Ptr<cv::FeatureDetector> detector;
	// 指向描述子提取器的指针
	cv::Ptr<cv::DescriptorExtractor> extractor;
	float ratio; // 第一个以及第二个最近邻之间的最大比率
	bool refineF; // 是否改变F矩阵
	double distance; // 到极线的最小距离
	double confidence; // 置信等级（概率）

public:

	RobustMatcher() : ratio(0.65f), refineF(true), confidence(0.99), distance(3.0)
	{
		// SURF 为默认特征
		detector = new cv::SurfFeatureDetector();
		extractor = new cv::SurfDescriptorExtractor();
	}

	// 设置特征检测器
	void setFeatureDetector(cv::Ptr<cv::FeatureDetector> &detect)
	{
		detector = detect;
	}

	// 设置描述子提取器
	void setDescriptorExtractor(cv::Ptr<cv::DescriptorExtractor> &desc)
	{
		extractor = desc;
	}

	// 设置 RANSAC 中到极线的最小距离
	void setMinDistanceToEpipolar(double d)
	{
		distance = d;
	}

	// 设置 RANSAC 中的置信等级
	void setConfidenceLevel(double c)
	{
		confidence = c;
	}

	// 设置最近邻之间的比率
	void setRatio(float r)
	{
		ratio = r;
	}

	// 标记是否 F 矩阵需要重新计算
	void refineFundamental(bool flag)
	{
		refineF = flag;
	}

	// 移除 NN 比率大于阈值的匹配
	// 返回移除点的数目
	// (对应的项被清0，即尺寸将为0)
	int ratioTest(std::vector<std::vector<cv::DMatch>> &matches)
	{
		int removed = 0;

		// 遍历所有匹配
		for (std::vector<std::vector<cv::DMatch>>::iterator matchIterator = matches.begin();
			matchIterator != matches.end(); ++matchIterator)
		{
			// 如果识别两个最近邻
			if (matchIterator->size() > 1)
			{
				// 检测距离比率
				if ((*matchIterator)[0].distance / (*matchIterator)[1].distance > ratio)
				{
					matchIterator->clear(); // 移除匹配
					removed++;
				}

			}
			else     // 不包含两个最近邻
			{
				matchIterator->clear(); // 移除匹配
				removed++;
			}
		}
		return removed;
	}

	// 在 symmetrical 向量中插入对称匹配
	void symmetryTest(const std::vector<std::vector<cv::DMatch>> &matches1,
		const std::vector<std::vector<cv::DMatch>> &matches2,
		std::vector<cv::DMatch> &symMatches)
	{
		// 遍历图1 -> 图2的所有匹配
		for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches1.begin();
			matchIterator1 != matches1.end(); ++matchIterator1)
		{
			if (matchIterator1->size() < 2) // 忽略被删除的匹配
				continue;
			// 遍历图2 -> 图1的所有匹配
			for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 = matches2.begin();
				matchIterator2 != matches2.end(); ++matchIterator2)
			{
				if (matchIterator2->size() < 2) // 忽略被删除的匹配
					continue;
				// 对称性测试
				if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&
					(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx)
				{
					// 增加对称的匹配
					symMatches.push_back(cv::DMatch((*matchIterator1)[0].queryIdx,
						(*matchIterator1)[0].trainIdx,
						(*matchIterator1)[0].distance));
					break; // 图1 -> 图2的下一个匹配
				}
			}
		}
	}

	// 基于 RANSAC 识别优质匹配
	// 返回基础矩阵
	cv::Mat ransacTest(const std::vector<cv::DMatch> &matches,
		const std::vector<cv::KeyPoint> &keypoints1,
		const std::vector<cv::KeyPoint> &keypoints2,
		std::vector<cv::DMatch> &outMatches)
	{
		// 把 keypoints 转换为 Point2f
		std::vector<cv::Point2f> points1, points2;
		for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it)
		{
			// 得到左边特征值的坐标
			float x = keypoints1[it->queryIdx].pt.x;
			float y = keypoints1[it->queryIdx].pt.y;
			points1.push_back(cv::Point2f(x, y));
			// 得到右边特征值的坐标
			x = keypoints2[it->trainIdx].pt.x;
			y = keypoints2[it->trainIdx].pt.y;
			points2.push_back(cv::Point2f(x, y));
		}

		// 基于 RANSAC 计算F矩阵
		std::vector<uchar> inliers(points1.size(), 0);
		cv::Mat fundemental = cv::findFundamentalMat(
			cv::Mat(points1), cv::Mat(points2), // 匹配点
			inliers,      // 匹配状态 (inlier ou outlier)
			CV_FM_RANSAC, // RANSAC 方法
			distance,     // 到极线的距离
			confidence);  // 置信概率

		// 提取通过的匹配
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		// 遍历所有匹配
		for (; itIn != inliers.end(); ++itIn, ++itM)
		{
			if (*itIn)   // 为有限匹配
			{
				outMatches.push_back(*itM);
			}
		}

		std::cout << "Number of matched points (after cleaning): " << outMatches.size() << std::endl;

		if (refineF)
		{
			// F矩阵将使用所有接受的匹配重新计算

			// 把  KeyPoint 类型转换为 Point2f 类型
			points1.clear();
			points2.clear();

			for (std::vector<cv::DMatch>::const_iterator it = outMatches.begin();
				it != outMatches.end(); ++it)
			{
				// 得到左边特征点的坐标
				float x = keypoints1[it->queryIdx].pt.x;
				float y = keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x, y));
				// 得到右边特征点的坐标
				x = keypoints2[it->trainIdx].pt.x;
				y = keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x, y));
			}

			// 从所有接受的匹配中计算8点F
			fundemental = cv::findFundamentalMat(
				cv::Mat(points1), cv::Mat(points2), // 匹配点
				CV_FM_8POINT); // 8点匹配法
		}
		return fundemental;
	}

	// 使用对称性测试以及 PANSAC 匹配特征点
	// 返回基础矩阵
	cv::Mat match(cv::Mat &image1, cv::Mat &image2, // 输入图像
		std::vector<cv::DMatch> &matches, // 输出匹配及特征点
		std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2)
	{
		// 1a. 检测SURF特征
		detector->detect(image1, keypoints1);
		detector->detect(image2, keypoints2);

		std::cout << "Number of SURF points (1): " << keypoints1.size() << std::endl;
		std::cout << "Number of SURF points (2): " << keypoints2.size() << std::endl;

		// 1b. 提取SURF特征描述子
		cv::Mat descriptors1, descriptors2;
		extractor->compute(image1, keypoints1, descriptors1);
		extractor->compute(image2, keypoints2, descriptors2);

		std::cout << "descriptor matrix size: " << descriptors1.rows << " by " << descriptors1.cols << std::endl;

		// 2. 匹配两幅图像的描述子

		// 创建匹配器
		cv::BruteForceMatcher<cv::L2<float>> matcher;

		// 从图1 到 图2的k个最近邻（k=2）
		std::vector<std::vector<cv::DMatch>> matches1;
		matcher.knnMatch(descriptors1, descriptors2,
			matches1, // 匹配结果的向量（每项有两个值）
			2);       // 返回两个最近邻

		// 从图2 到 图1的k个最近邻（k=2）
		std::vector<std::vector<cv::DMatch>> matches2;
		matcher.knnMatch(descriptors2, descriptors1,
			matches2, // 匹配结果的向量（每项有两个值）
			2);       // 返回两个最近邻

		std::cout << "Number of matched points 1->2: " << matches1.size() << std::endl;
		std::cout << "Number of matched points 2->1: " << matches2.size() << std::endl;

		// 3. 移除NN比率大于阈值的匹配

		// 清理图1 -> 图2的匹配
		int removed = ratioTest(matches1);
		std::cout << "Number of matched points 1->2 (ratio test) : " << matches1.size() - removed << std::endl;
		// 清理图2 -> 图1的匹配
		removed = ratioTest(matches2);
		std::cout << "Number of matched points 1->2 (ratio test) : " << matches2.size() - removed << std::endl;

		// 4. 移除非对称的匹配
		std::vector<cv::DMatch> symMatches;
		symmetryTest(matches1, matches2, symMatches);

		std::cout << "Number of matched points (symmetry test): " << symMatches.size() << std::endl;

		// 5. 使用RANSAC进行最终验证
		cv::Mat fundemental = ransacTest(symMatches, keypoints1, keypoints2, matches);

		// 返回找到的基础矩阵
		return fundemental;
	}
};

#endif
