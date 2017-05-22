#pragma once

#include "time.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "stdio.h"
#include <cmath>
#include "math_constants.h"
#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

__host__ __device__ unsigned int XY_TO_INT(int x, int y);

__host__ __device__ int INT_TO_X(unsigned int v);

__host__ __device__ int INT_TO_Y(unsigned int v);

__global__ void blend(float *cmap, float* oldd, float* newd, float weight, int * params);

__global__ void patchmatch(float * a, float * b, float *a1, float *b1, unsigned int *ann, float *annd, int * params);

__global__ void initialAnn_kernel( unsigned int * ann, int * params);

__global__ void upSample_kernel( unsigned int * ann, unsigned int * ann_tmp, int * params, int aw_half, int ah_half);

__global__ void avg_vote(unsigned int * ann, float * pb, float * pc, int * params);

__host__ Mat reconstruct_dflow(Mat a, Mat b, unsigned int * ann, int patch_w);

__host__ Mat reconstruct_avg(Mat a, Mat b, unsigned int * ann, int patch_w);

__host__ __device__ int clamp(int x, int x_max, int x_min);

