#ifndef WLS_H
#define WLS_H

#include "opencv2/opencv.hpp"
#include <cusolverSp.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

void WeightedLeastSquare(cv::Mat& resImg, const cv::Mat& img0, const cv::Mat& img1,
	float alpha = 1.f, float lamda = 0.8f);


#endif