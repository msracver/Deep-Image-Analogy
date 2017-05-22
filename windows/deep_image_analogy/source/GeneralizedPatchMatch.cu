

#include "GeneralizedPatchmatch.cuh"
#include "curand_kernel.h"

__host__ __device__ int clamp(int x, int x_max, int x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ unsigned int XY_TO_INT(int x, int y) {//r represent the number of 10 degree, x,y - 11 bits, max = 2047, r - max = 36, 6 bits
	return (((y) << 11) | (x));
}
__host__ __device__ int INT_TO_X(unsigned int v) {
	return (v)&((1 << 11) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
	return (v >> 11)&((1 << 11) - 1);
}

__host__ __device__ int cuMax(int a, int b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}
__host__ __device__ int cuMin(int a, int b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ float MycuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	 return curand_uniform(&state);

}
__device__ void InitcuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(i, 0, 0, &state);

}

__host__ Mat reconstruct_avg(Mat a, Mat b, unsigned int * ann, int patch_w) {

	Mat c;
	a.copyTo(c);
	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++)
		{
		
			float point_num = 0, *dist_tmp;
			
			dist_tmp = new float[3];

			for (int dc = 0; dc < 3; dc++)
			{
				dist_tmp[dc] = 0;
			}

			for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
				for (int dy = -patch_w / 2; dy <=patch_w / 2; dy++)
				{

					if ((ax + dx) < a.cols && (ax + dx) >= 0 && (ay + dy) < a.rows && (ay + dy) >= 0)
					{

						unsigned int vp = ann[(ay + dy)*a.cols + ax + dx];
						int xp = INT_TO_X(vp), yp = INT_TO_Y(vp);

						if ((xp - dx) < b.cols && (xp - dx) >= 0 && (yp - dy) < b.rows && (yp - dy) >= 0)//a patch that contain this pixel
						{

							for (int dc = 0; dc < 3; dc++)
							{
								dist_tmp[dc] += b.at<Vec3b>(yp - dy, xp - dx).val[dc];
							}

							point_num++;
						}
					}

				}

			}

			for (int dc = 0; dc < 3; dc++)
			{
				c.at<Vec3b>(ay, ax).val[dc] = dist_tmp[dc]/point_num;
			}
			
			delete[] dist_tmp;
		}
	}
	return c;
}

__host__ Mat reconstruct_dflow(Mat a, Mat b, unsigned int * ann, int patch_w) {
	Mat flow;
	a.copyTo(flow);
	for (int ay = 0; ay < a.rows; ay++) {
		for (int ax = 0; ax < a.cols; ax++)
		{
			unsigned int v = ann[ay*a.cols + ax];
			int xbest = INT_TO_X(v);
			int ybest = INT_TO_Y(v);
			flow.at<Vec3b>(ay, ax).val[0] = (uchar)(255 * ((float)(ax - xbest + b.cols - 1) / (2 * b.cols)));
			flow.at<Vec3b>(ay, ax).val[2] = (uchar)(0);
			flow.at<Vec3b>(ay, ax).val[1] = (uchar)(255 * ((float)(ay - ybest + b.rows - 1) / (2 * b.rows)));
		}
	}
	return flow;
}

__host__ __device__ float dist_compute(float * a, float * b, float * a1, float * b1, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int bx, int by, int patch_w, float cutoff = INT_MAX) {//this is the average number of all matched pixel
																																																		  //suppose patch_w is an odd number
	float pixel_sum = 0, pixel_no = 0, pixel_dist = 0;//number of pixels realy counted
	float pixel_sum1 = 0;
	int a_slice = a_rows*a_cols, b_slice = b_rows*b_cols;
	int a_pitch = a_cols, b_pitch = b_cols;
	float dp_tmp;

	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {

			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{
				if (channels == 3)
				{
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp = a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] - b[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
						pixel_sum += dp_tmp * dp_tmp;
						dp_tmp = a1[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] - b1[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
						pixel_sum1 += dp_tmp * dp_tmp;

					}
				}
				else
				{
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp = a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
						pixel_sum -= dp_tmp;
						dp_tmp = a1[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b1[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
						pixel_sum1 -= dp_tmp;
					}
				}
				

				pixel_no += 1;
			}
		}

	}


	pixel_dist = (pixel_sum + pixel_sum1) / pixel_no;
	if (pixel_dist >= cutoff) { return cutoff; }
	else {
		return pixel_dist;
	}
}

__host__ __device__ float dist(float * a, float * b, float *a1, float *b1, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int xp, int yp, int patch_w, float cutoff = INT_MAX) {

	return dist_compute(a, b, a1, b1,  channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, cutoff);

}

__device__ void improve_guess(float * a, float * b, float *a1, float *b1, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, float &dbest, int xp, int yp, int patch_w, float rr) {
	float d;
	d = dist(a, b, a1, b1, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w, dbest);
	if (d + rr < dbest) {
		xbest = xp;
		ybest = yp;
		dbest = d;
	}
}


__global__ void initialAnn_kernel(unsigned int * ann, int * params) {

	//just use 7 of 9 parameters
	int ah = params[1];
	int aw = params[2];


	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	if (ax < aw && ay < ah) {
		int bx = ax;
		int by = ay;
		ann[ay*aw + ax] = XY_TO_INT(bx, by);
	}
}

__global__ void upSample_kernel(unsigned int * ann, unsigned int * ann_tmp,int * params, int aw_half,int ah_half) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	
	
	float aw_ratio = (float)aw / (float)aw_half;
	float ah_ratio = (float)ah / (float)ah_half;
	int ax_half = (ax+0.5) / aw_ratio;
	int ay_half = (ay+0.5) / ah_ratio;
	ax_half = clamp(ax_half, aw_half - 1, 0);
	ay_half = clamp(ay_half, ah_half - 1, 0);
	

	if (ax < aw&&ay < ah) {

		unsigned int v_half = ann[ay_half*aw_half + ax_half];
		int bx_half = INT_TO_X(v_half);
		int by_half = INT_TO_Y(v_half);

		int bx = ax + (bx_half - ax_half)*aw_ratio + 0.5;
		int by = ay + (by_half - ay_half)*ah_ratio + 0.5;

		bx = clamp(bx, bw-1, 0);
		by = clamp(by, bh-1, 0);

		ann_tmp[ay*aw + ax] = XY_TO_INT(bx, by);
	}

}

__global__ void patchmatch(float * a, float * b, float *a1, float *b1, unsigned int *ann, float *annd, int * params) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	int b_rows = params[3];
	int b_cols = params[4];
	int patch_w = params[5];
	int pm_iters = params[6];
	int rs_max = params[7];


	if (ax < a_cols && ay < a_rows) {
	
		// for random number
		curandState state;
		InitcuRand(state);

		unsigned int v, vp;

		int xp, yp, xbest, ybest;

		int xmin, xmax, ymin, ymax;

		float dbest;
		v = ann[ay*a_cols + ax];
		xbest = INT_TO_X(v), ybest = INT_TO_Y(v);			
		annd[ay*a_cols + ax] = dist(a, b, a1, b1,  ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, patch_w);

		for (int iter = 0; iter < pm_iters; iter++) {

			/* Current (best) guess. */
			v = ann[ay*a_cols + ax];
			xbest = INT_TO_X(v), ybest = INT_TO_Y(v);			
			dbest = annd[ay*a_cols + ax];

			/* In each iteration, improve the NNF, by jumping flooding. */
			for (int jump = 8; jump > 0; jump /= 2) {

				/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
				if ((ax - jump) < a_cols && (ax - jump) >= 0)//left
				{
					vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b

					xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp, the center of the patch, which should be in the image

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}
				
				if ((ax + jump) < a_cols)//right
				{
					vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b

					xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay - jump) < a_rows && (ay - jump) >= 0)//up
				{
					vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b
					xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{

						//improve guess
						improve_guess(a, b, a1, b1, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay + jump) < a_rows)//down
				{
					vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b	
					xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, a1, b1,  ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = rs_max;
			if (rs_start > cuMax(b_cols, b_rows)) {
				rs_start = cuMax(b_cols, b_rows);
			}
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				/* Sampling window */
				xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
				ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
				xp = xmin + (int)(MycuRand(state)*(xmax - xmin)) % (xmax - xmin);
				yp = ymin + (int)(MycuRand(state)*(ymax - ymin)) % (ymax - ymin);

				//improve guess
				improve_guess(a, b, a1, b1,  ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN);

			}

			ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
			annd[ay*a_cols + ax] = dbest;
			__syncthreads();
		}
	}
}

__global__ void blend(float *cmap, float* oldd, float* newd, float weight,int * params)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	
	int slice_a = ah * aw;
	int pitch_a = aw;
	
	float thre = 0.05;
	
	if (ax < aw&& ay < ah)
	{
		float fa = cmap[ay*pitch_a + ax];
		if (fa < thre)
			fa = 0.0f;
		else fa = weight;
		for (int i = 0; i < ch; i++)
		{
			
			newd[i*slice_a + ay*pitch_a + ax] = oldd[i*slice_a + ay*pitch_a + ax]* fa + newd[i*slice_a + ay*pitch_a + ax] * (1.0-fa);
		}
	}
}

// ********** VOTE ***********

__global__ void avg_vote(unsigned int * ann, float * pb, float * pc, int * params) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	int count = 0;

	if (ax < aw&&ay < ah)
	{

		//set zero for all the channels at (ax,ay)
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] = 0;

		}

		//count the sum of all the possible value of (ax,ay)
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
			for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
			{

				if ((ax + dx) < aw && (ax + dx) >= 0 && (ay + dy) < ah && (ay + dy) >= 0)
				{
					unsigned int vp = ann[(ay + dy)*aw + ax + dx];
					
					int xp = INT_TO_X(vp);
					int yp = INT_TO_Y(vp);

					if ((xp - dx) < bw && (xp - dx) >= 0 && (yp - dy) < bh && (yp - dy) >= 0)
					{
						count++;
						for (int dc = 0; dc < ch; dc++)
						{
							pc[dc*slice_a + ay*pitch_a + ax] += pb[dc*slice_b + (yp - dy)*pitch_b + xp - dx];
						}
					}
				}

			}
		}

		//count average value
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] /= count;
		}

	}
}
