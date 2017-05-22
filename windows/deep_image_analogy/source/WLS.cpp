#include "WLS.h"
#include <stdio.h> 
#include <stdlib.h>
#include <Eigen/Sparse>

void WeightedLeastSquare(cv::Mat& resImg, const cv::Mat& img_guide, const cv::Mat& img_color,
	float alpha, float lamda)
{

	float epsilon = 0.0001f;

	cv::Mat grayImgF = cv::Mat::zeros(img_guide.size(), CV_32FC1);
	cv::cvtColor(img_guide, grayImgF, CV_BGR2GRAY);

	cv::Mat gradWeightX = cv::Mat::zeros(img_guide.size(), CV_32FC1);
	cv::Mat gradWeightY = cv::Mat::zeros(img_guide.size(), CV_32FC1);

#pragma omp parallel for
	for (int y = 0; y < img_guide.rows - 1; ++y)
	{
		for (int x = 0; x < img_guide.cols - 1; ++x)
		{
			if (x + 1 < img_guide.cols)
			{
				float gx = grayImgF.at<float>(y, x + 1) - grayImgF.at<float>(y, x);
				gradWeightX.at<float>(y, x) = lamda / (pow(abs(gx), alpha) + epsilon);
			}
			if (y + 1 < img_guide.rows)
			{
				float gy = grayImgF.at<float>(y + 1, x) - grayImgF.at<float>(y, x);
				gradWeightY.at<float>(y, x) = lamda / (pow(abs(gy), alpha) + epsilon);
			}
		}
	}
	//prepare
	int width = img_color.cols;
	int height = img_color.rows;
	int size = width * height;
	int n = width * height;

	//matrix	
	Eigen::SparseMatrix<float> A(n,n);
	Eigen::VectorXi nnzVec = Eigen::VectorXi::Constant(n, 4 + 1);
	Eigen::VectorXf bs[3], xs[3];
	A.reserve(nnzVec);
	bs[0].resize(n);
	bs[1].resize(n);
	bs[2].resize(n);

	xs[0].resize(n);
	xs[1].resize(n);
	xs[2].resize(n);


	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			float a[5];
			a[0] = a[1] = a[2] = a[3] = a[4] = 0.0f;

			int ii = y * width + x;
			if (y - 1 >= 0) // top
			{
				const float gyw = gradWeightY.at<float>(y - 1, x);
				a[2] += 1.0f * gyw;
				a[0] -= 1.0f * gyw;
				A.insert(ii, ii - width) = a[0];
			}
			if (x - 1 >= 0) // left
			{
				const float gxw = gradWeightX.at<float>(y, x - 1);
				a[2] += 1.0f * gxw;
				a[1] -= 1.0f * gxw;
				A.insert(ii, ii - 1) = a[1];
			}
			if (x + 1 < width) // right
			{
				const float gxw = gradWeightX.at<float>(y, x);
				a[2] += 1.0f * gxw;
				a[3] -= 1.0f * gxw;
				A.insert(ii, ii + 1) = a[3];
			}
			if (y + 1 < height) // bottom
			{
				const float gyw = gradWeightY.at<float>(y, x);
				a[2] += 1.0f * gyw;
				a[4] -= 1.0f * gyw;
				A.insert(ii, ii + width) = a[4];
			}

			// data term
			a[2] += 1.f;
			A.insert(ii, ii) = a[2];

			const cv::Vec3f& col = img_color.at<cv::Vec3f>(y, x);
			xs[0][ii] = 0.0f;
			xs[1][ii] = 0.0f;
			xs[2][ii] = 0.0f;
			bs[0][ii] = (float)col[0];			
			bs[1][ii] = (float)col[1];
			bs[2][ii] = (float)col[2];
		}
	}

#pragma omp parallel for
	for (int ch = 0; ch < 3; ++ch)
	{
		Eigen::SimplicialLLT<Eigen::SparseMatrix<float> > solver;
		solver.compute(A);
		xs[ch] = solver.solve(bs[ch]);
	}


	//paste	
	resImg = cv::Mat(height, width, CV_32FC3);

#pragma omp parallel for
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			resImg.at<cv::Vec3f>(y, x) = cv::Vec3f(
				xs[0][y * width + x],
				xs[1][y * width + x],
				xs[2][y * width + x]);

		}
	}

}
