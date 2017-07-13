#include "caffe/util/math_functions.hpp"
#include "caffe/common.hpp"
#include "GeneralizedPatchMatch.cuh"
#include "DeepAnalogy.cuh"
#include "WLS.h"
#include "Deconv.h"

struct Parameters
{

	std::vector<std::string> layers; //which layers  used as content

	int patch_size0;
	int iter;

};


__host__ void norm(float* &dst, float* src, float* smooth, Dim dim){

	int count = dim.channel*dim.height*dim.width;
	float* x = src;
	float* x2;
	cudaMalloc(&x2, count*sizeof(float));
	caffe_gpu_mul(count, x, x, x2);

	//caculate dis
	float*sum;
	float* ones;
	cudaMalloc(&sum, dim.height*dim.width*sizeof(float));
	cudaMalloc(&ones, dim.channel*sizeof(float));
	caffe_gpu_set(dim.channel, 1.0f, ones);
	caffe_gpu_gemv(CblasTrans, dim.channel, dim.height*dim.width, 1.0f, x2, ones, 0.0f, sum);

	float *dis;
	cudaMalloc(&dis, dim.height*dim.width*sizeof(float));
	caffe_gpu_powx(dim.height*dim.width, sum, 0.5f, dis);

	if (smooth != NULL)
	{
		cudaMemcpy(smooth, sum, dim.height*dim.width*sizeof(float), cudaMemcpyDeviceToDevice);
		int index;
		float minv, maxv;
		cublasIsamin(Caffe::cublas_handle(), dim.height*dim.width, sum, 1, &index);
		cudaMemcpy(&minv, sum + index - 1, sizeof(float), cudaMemcpyDeviceToHost);
		cublasIsamax(Caffe::cublas_handle(), dim.height*dim.width, sum, 1, &index);
		cudaMemcpy(&maxv, sum + index - 1, sizeof(float), cudaMemcpyDeviceToHost);

		caffe_gpu_add_scalar(dim.height*dim.width, -minv, smooth);
		caffe_gpu_scal(dim.height*dim.width, 1.0f / (maxv - minv), smooth);
	}


	//norm
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, dim.channel, dim.width*dim.height, 1, 1.0f, ones, dis, 0.0f, x2);
	caffe_gpu_div(count, src, x2, dst);

	cudaFree(x2);
	cudaFree(ones);
	cudaFree(dis);
	cudaFree(sum);
}

DeepAnalogy::DeepAnalogy(){
	resizeRatio = 1;
	weightLevel = 3;
	photoTransfer = false;
	file_A = "";
	file_BP = "";
	path_output = "";
	path_model = "";


}

DeepAnalogy::~DeepAnalogy(){

}

void DeepAnalogy::SetRatio(float ratio){
	resizeRatio = ratio;
}
void DeepAnalogy::SetBlendWeight(int level){
	weightLevel = level;
}
void DeepAnalogy::UsePhotoTransfer(bool flag){
	photoTransfer = flag;
}
void DeepAnalogy::SetModel(string path){
	path_model =path;
}
void DeepAnalogy::SetA(string f_a){
	file_A = f_a;
}
void DeepAnalogy::SetBPrime(string f_bp){
	file_BP = f_bp;
}
void DeepAnalogy::SetOutputDir(string f_o){
	path_output = f_o;
}
void DeepAnalogy::SetGPU(int no){
	cudaSetDevice(no);
}
void DeepAnalogy::LoadInputs(){
	float ratio;
	Mat ori_AL = imread(file_A);
	Mat ori_BPL = imread(file_BP);
	if (ori_AL.empty() || ori_BPL.empty())
	{
		cout << "image cannot read!" << endl;
		waitKey();
		return;
	}

	ori_A_cols = ori_AL.cols;
	ori_A_rows = ori_AL.rows;
	ori_BP_cols = ori_BPL.cols;
	ori_BP_rows = ori_BPL.rows;

	if (ori_AL.rows > 700)
	{
		ratio = 700.f / ori_AL.rows;
		cv::resize(ori_AL, img_AL, Size(), ratio, ratio, INTER_CUBIC);
		ori_AL = img_AL.clone();

	}
	if (ori_AL.cols > 700)
	{
		ratio = 700.f / ori_AL.cols;
		cv::resize(ori_AL, img_AL, Size(), ratio, ratio, INTER_CUBIC);
		ori_AL = img_AL.clone();

	}

	if (ori_AL.rows < 200)
	{
		ratio = 200.f / ori_AL.rows;
		cv::resize(ori_AL, img_AL, Size(), ratio, ratio, INTER_CUBIC);
		ori_AL = img_AL.clone();

	}

	if (ori_AL.cols < 200)
	{
		ratio = 200.f / ori_AL.cols;
		cv::resize(ori_AL, img_AL, Size(), ratio, ratio, INTER_CUBIC);
		ori_AL = img_AL.clone();

	}

	if (ori_BPL.rows > 700)
	{
		ratio = 700.f / ori_BPL.rows;
		cv::resize(ori_BPL, img_BPL, Size(), ratio, ratio, INTER_CUBIC);
		ori_BPL = img_BPL.clone();

	}

	if (ori_BPL.cols > 700)
	{
		ratio = 700.f / ori_BPL.cols;
		cv::resize(ori_BPL, img_BPL, Size(), ratio, ratio, INTER_CUBIC);
		ori_BPL = img_BPL.clone();
	}
	if (ori_BPL.rows < 200)
	{
		ratio = 200.f / ori_BPL.rows;
		cv::resize(ori_BPL, img_BPL, Size(), ratio, ratio, INTER_CUBIC);
		ori_BPL = img_BPL.clone();

	}

	if (ori_BPL.cols < 200)
	{
		ratio = 200.f / ori_BPL.cols;
		cv::resize(ori_BPL, img_BPL, Size(), ratio, ratio, INTER_CUBIC);
		ori_BPL = img_BPL.clone();

	}

	if ((ori_AL.cols*ori_AL.rows) > 350000)
	{
		ratio = sqrt((float)(350000) / (float)(ori_AL.cols*ori_AL.rows));
		cv::resize(ori_AL, img_AL, Size(), ratio, ratio, INTER_CUBIC);
		ori_AL = img_AL.clone();

	}

	if ((ori_BPL.cols*ori_BPL.rows) > 350000)
	{
		ratio = sqrt((float)(350000) / (float)(ori_BPL.cols*ori_BPL.rows));
		cv::resize(ori_BPL, img_BPL, Size(), ratio, ratio, INTER_CUBIC);
		ori_BPL = img_BPL.clone();
	}


	int maxLateral, minLateral;
	maxLateral = max(max(ori_AL.rows, ori_AL.cols), max(ori_BPL.rows, ori_BPL.cols));
	minLateral = min(min(ori_AL.rows, ori_AL.cols), min(ori_BPL.rows, ori_BPL.cols));

	if (maxLateral > 700 || minLateral < 200)
	{
		cout << "The sizes of images are not permitted. (One side cannot be larger than 700 or smaller than 200 and the area should not be larger than 350000)" << endl;
		waitKey();
		return;
	}


	cur_A_cols = ori_AL.cols;
	cur_A_rows = ori_AL.rows;
	cur_BP_cols = ori_BPL.cols;
	cur_BP_rows = ori_BPL.rows;

	if (ori_A_cols != ori_AL.cols)
	{
		cout << "The input image A has been resized to " << cur_A_cols << " x " << cur_A_rows << ".\n";
	}

	if (ori_BP_cols != ori_BPL.cols)
	{
		cout << "The input image B prime has been resized to " << cur_BP_cols << " x " << cur_BP_rows << ".\n";
	}


	cv::resize(ori_AL, img_AL, Size(), (float)cur_A_cols / ori_AL.cols, (float)cur_A_rows / ori_AL.rows, INTER_CUBIC);
	cv::resize(ori_BPL, img_BPL, Size(), (float)cur_BP_cols / ori_BPL.cols, (float)cur_BP_rows / ori_BPL.rows, INTER_CUBIC);


}

void DeepAnalogy::ComputeAnn() {

	if (img_BPL.empty()||img_AL.empty())
	{
		waitKey();
		return;
	}

	const int param_size = 8;


	int ann_size_AB, ann_size_BA;//should be assigned later
	int *params_host, *params_device_AB, *params_device_BA;
	unsigned int *ann_device_AB, *ann_host_AB, *ann_device_BA, *ann_host_BA;
	float *annd_device_AB, *annd_host_AB, *annd_device_BA, *annd_host_BA;

	char fname[256];

	//set parameters
	Parameters params;
	params.layers.push_back("conv5_1");
	params.layers.push_back("conv4_1");
	params.layers.push_back("conv3_1");
	params.layers.push_back("conv2_1");
	params.layers.push_back("conv1_1");
	params.layers.push_back("data");

	std::vector<float> weight;
	weight.push_back(1.0);
	switch (weightLevel)
	{
	case 1:
		weight.push_back(0.7);
		weight.push_back(0.6);
		weight.push_back(0.5);
		weight.push_back(0.0);
		break;
	case 2:
		weight.push_back(0.8);
		weight.push_back(0.7);
		weight.push_back(0.6);
		weight.push_back(0.1);
		break;

	case 3:
		weight.push_back(0.9);
		weight.push_back(0.8);
		weight.push_back(0.7);
		weight.push_back(0.2);
		break;

	default:
		weight.push_back(0.9);
		weight.push_back(0.8);
		weight.push_back(0.7);
		weight.push_back(0.2);
		break;
	}

	weight.push_back(0.0);

	std::vector<int> sizes;
	sizes.push_back(3);
	sizes.push_back(3);
	sizes.push_back(3);
	sizes.push_back(5);
	sizes.push_back(5);
	sizes.push_back(3);

	params.iter = 10;

	//scale and enhance
    float ratio = resizeRatio;
	Mat img_BP, img_A;
	cv::resize(img_AL, img_A, Size(), ratio, ratio, INTER_CUBIC);
	cv::resize(img_BPL, img_BP, Size(), ratio, ratio, INTER_CUBIC);

	std::vector<int> range;
	if (img_A.cols > img_A.rows)
	{
		range.push_back(img_A.cols / 16);

	}
	else
	{
		range.push_back(img_A.rows / 16);

	}
	range.push_back(6);
	range.push_back(6);
	range.push_back(4);
	range.push_back(4);
	range.push_back(2);

	//load caffe
	::google::InitGoogleLogging("deepanalogy");
	string model_file = "vgg19/VGG_ILSVRC_19_layers_deploy.prototxt";
	string trained_file = "vgg19/VGG_ILSVRC_19_layers.caffemodel";

	Classifier classifier_A(path_model + model_file, path_model + trained_file);
	Classifier classifier_B(path_model + model_file, path_model + trained_file);


	std::vector<float *> data_A, data_AP;
	data_A.resize(params.layers.size());
	data_AP.resize(params.layers.size());
	std::vector<Dim> data_A_size;
	data_A_size.resize(params.layers.size());
	classifier_A.Predict(img_A, params.layers, data_AP, data_A, data_A_size);

	std::vector<float *> data_B, data_BP;
	data_B.resize(params.layers.size());
	data_BP.resize(params.layers.size());
	std::vector<Dim> data_B_size;
	data_B_size.resize(params.layers.size());
	classifier_B.Predict(img_BP, params.layers, data_B, data_BP, data_B_size);

	clock_t start, finish;
	double duration;
	start = clock();

	ann_size_AB = img_AL.cols*img_AL.rows;
	ann_size_BA = img_BPL.cols*img_BPL.rows;
	params_host = (int *)malloc(param_size * sizeof(int));
	ann_host_AB = (unsigned int *)malloc(ann_size_AB * sizeof(unsigned int));
	annd_host_AB = (float *)malloc(ann_size_AB * sizeof(float));
	ann_host_BA = (unsigned int *)malloc(ann_size_BA * sizeof(unsigned int));
	annd_host_BA = (float *)malloc(ann_size_BA * sizeof(float));

	cudaMalloc(&params_device_AB, param_size * sizeof(int));
	cudaMalloc(&params_device_BA, param_size * sizeof(int));
	cudaMalloc(&ann_device_AB, ann_size_AB * sizeof(unsigned int));
	cudaMalloc(&annd_device_AB, ann_size_AB * sizeof(float));
	cudaMalloc(&ann_device_BA, ann_size_BA * sizeof(unsigned int));
	cudaMalloc(&annd_device_BA, ann_size_BA * sizeof(float));

	int numlayer = params.layers.size();


	//feature match
	for (int curr_layer = 0; curr_layer < numlayer - 1; curr_layer++)//from 32 to 512
	{

		//set parameters
		params_host[0] = data_A_size[curr_layer].channel;//channels
		params_host[1] = data_A_size[curr_layer].height;
		params_host[2] = data_A_size[curr_layer].width;
		params_host[3] = data_B_size[curr_layer].height;
		params_host[4] = data_B_size[curr_layer].width;
		params_host[5] = sizes[curr_layer];
		params_host[6] = params.iter;
		params_host[7] = range[curr_layer];

		//copy to device
		cudaMemcpy(params_device_AB, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		//set parameters
		params_host[0] = data_B_size[curr_layer].channel;//channels
		params_host[1] = data_B_size[curr_layer].height;
		params_host[2] = data_B_size[curr_layer].width;
		params_host[3] = data_A_size[curr_layer].height;
		params_host[4] = data_A_size[curr_layer].width;

		//copy to device
		cudaMemcpy(params_device_BA, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		////set device pa, device pb, device ann and device annd
		dim3 blocksPerGridAB(data_A_size[curr_layer].width / 20 + 1, data_A_size[curr_layer].height / 20 + 1, 1);
		dim3 threadsPerBlockAB(20, 20, 1);
		ann_size_AB = data_A_size[curr_layer].width* data_A_size[curr_layer].height;
		dim3 blocksPerGridBA(data_B_size[curr_layer].width / 20 + 1, data_B_size[curr_layer].height / 20 + 1, 1);
		dim3 threadsPerBlockBA(20, 20, 1);
		ann_size_BA = data_B_size[curr_layer].width* data_B_size[curr_layer].height;

		//initialize ann if needed
		if (curr_layer == 0)//initialize, rows and cols both less than 32, just use one block
		{

			initialAnn_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, params_device_AB);
			initialAnn_kernel << <blocksPerGridBA, threadsPerBlockBA >> >(ann_device_BA, params_device_BA);

		}
		else {//upsampling, notice this block's dimension is twice the ann at this point
			unsigned int * ann_tmp;

			cudaMalloc(&ann_tmp, ann_size_AB * sizeof(unsigned int));
			upSample_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, ann_tmp, params_device_AB,
				data_A_size[curr_layer - 1].width, data_A_size[curr_layer - 1].height);//get new ann_device
			cudaMemcpy(ann_device_AB, ann_tmp, ann_size_AB * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
			cudaFree(ann_tmp);

			cudaMalloc(&ann_tmp, ann_size_BA * sizeof(unsigned int));
			upSample_kernel << <blocksPerGridBA, threadsPerBlockBA >> >(ann_device_BA, ann_tmp, params_device_BA,
				data_B_size[curr_layer - 1].width, data_B_size[curr_layer - 1].height);//get new ann_device
			cudaMemcpy(ann_device_BA, ann_tmp, ann_size_BA * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
			cudaFree(ann_tmp);

		}

		//normarlize two data
		float *Ndata_A, *Ndata_AP, *Ndata_B, *Ndata_BP;
		float *response_A, *response_BP;

		cudaMalloc(&Ndata_A, data_A_size[curr_layer].channel*data_A_size[curr_layer].width*data_A_size[curr_layer].height*sizeof(float));
		cudaMalloc(&Ndata_AP, data_A_size[curr_layer].channel*data_A_size[curr_layer].width*data_A_size[curr_layer].height*sizeof(float));
		cudaMalloc(&response_A, data_A_size[curr_layer].width*data_A_size[curr_layer].height*sizeof(float));
		cudaMalloc(&Ndata_B, data_B_size[curr_layer].channel*data_B_size[curr_layer].width*data_B_size[curr_layer].height*sizeof(float));
		cudaMalloc(&Ndata_BP, data_B_size[curr_layer].channel*data_B_size[curr_layer].width*data_B_size[curr_layer].height*sizeof(float));
		cudaMalloc(&response_BP, data_B_size[curr_layer].width*data_B_size[curr_layer].height*sizeof(float));


		norm(Ndata_A, data_A[curr_layer], response_A, data_A_size[curr_layer]);
		norm(Ndata_BP, data_BP[curr_layer], response_BP, data_B_size[curr_layer]);


		Mat temp1, temp2;
		cv::resize(img_AL, temp1, cv::Size(data_A_size[curr_layer].width, data_A_size[curr_layer].height));
		cv::resize(img_BPL, temp2, cv::Size(data_B_size[curr_layer].width, data_B_size[curr_layer].height));

		Mat response1, response2;
		response1 = Mat(temp1.size(), CV_32FC1);
		response2 = Mat(temp2.size(), CV_32FC1);

		cudaMemcpy(response1.data, response_A, data_A_size[curr_layer].width*data_A_size[curr_layer].height*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(response2.data, response_BP, data_B_size[curr_layer].width*data_B_size[curr_layer].height*sizeof(float), cudaMemcpyDeviceToHost);

		Mat response_byte1, response_byte2;
		response1.convertTo(response_byte1, CV_8UC1, 255);
		response2.convertTo(response_byte2, CV_8UC1, 255);

		blend << <blocksPerGridAB, threadsPerBlockAB >> >(response_A, data_A[curr_layer], data_AP[curr_layer], weight[curr_layer], params_device_AB);
		blend << <blocksPerGridBA, threadsPerBlockBA >> >(response_BP, data_BP[curr_layer], data_B[curr_layer], weight[curr_layer], params_device_BA);

		norm(Ndata_AP, data_AP[curr_layer], NULL, data_A_size[curr_layer]);
		norm(Ndata_B, data_B[curr_layer], NULL, data_B_size[curr_layer]);

		//patchmatch
		cout << "Finding nearest neighbor field using PatchMatch Algorithm at layer:" << params.layers[curr_layer] << ".\n";
		patchmatch << <blocksPerGridAB, threadsPerBlockAB >> >(Ndata_AP, Ndata_BP, Ndata_A, Ndata_B, ann_device_AB, annd_device_AB, params_device_AB);
		patchmatch << <blocksPerGridBA, threadsPerBlockBA >> >(Ndata_B, Ndata_A, Ndata_BP, Ndata_AP, ann_device_BA, annd_device_BA, params_device_BA);

		cudaFree(Ndata_A);
		cudaFree(Ndata_AP);
		cudaFree(Ndata_B);
		cudaFree(Ndata_BP);
		cudaFree(response_A);
		cudaFree(response_BP);

		//deconv
		if (curr_layer < numlayer - 2)
		{
			int next_layer = curr_layer + 2;

			/***************upsample***********************/
			// for better deconvolution
			params_host[0] = data_A_size[next_layer].channel;//channels
			params_host[1] = data_A_size[next_layer].height;
			params_host[2] = data_A_size[next_layer].width;
			params_host[3] = data_B_size[next_layer].height;
			params_host[4] = data_B_size[next_layer].width;
			params_host[5] = sizes[next_layer];
			params_host[6] = params.iter;
			params_host[7] = range[next_layer];

			//copy to device
			cudaMemcpy(params_device_AB, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

			//set parameters
			params_host[0] = data_B_size[next_layer].channel;//channels
			params_host[1] = data_B_size[next_layer].height;
			params_host[2] = data_B_size[next_layer].width;
			params_host[3] = data_A_size[next_layer].height;
			params_host[4] = data_A_size[next_layer].width;

			//copy to device
			cudaMemcpy(params_device_BA, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

			////set device pa, device pb, device ann and device annd
			dim3 blocksPerGridAB(data_A_size[next_layer].width / 20 + 1, data_A_size[next_layer].height / 20 + 1, 1);
			dim3 threadsPerBlockAB(20, 20, 1);
			ann_size_AB = data_A_size[next_layer].width* data_A_size[next_layer].height;
			dim3 blocksPerGridSC(data_B_size[next_layer].width / 20 + 1, data_B_size[next_layer].height / 20 + 1, 1);
			dim3 threadsPerBlockBA(20, 20, 1);
			ann_size_BA = data_B_size[next_layer].width* data_B_size[next_layer].height;

			unsigned int * ann_tmp;
			cudaMalloc(&ann_tmp, ann_size_AB * sizeof(unsigned int));
			upSample_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, ann_tmp, params_device_AB,
				data_A_size[curr_layer].width, data_A_size[curr_layer].height);//get new ann_device
			avg_vote << <blocksPerGridAB, threadsPerBlockAB >> >(ann_tmp, data_BP[next_layer], data_AP[next_layer], params_device_AB);
			cudaFree(ann_tmp);

			cudaMalloc(&ann_tmp, ann_size_BA * sizeof(unsigned int));
			upSample_kernel << <blocksPerGridBA, threadsPerBlockBA >> >(ann_device_BA, ann_tmp, params_device_BA,
				data_B_size[curr_layer].width, data_B_size[curr_layer].height);//get new ann_devices
			avg_vote << <blocksPerGridBA, threadsPerBlockBA >> >(ann_tmp, data_A[next_layer], data_B[next_layer], params_device_BA);
			cudaFree(ann_tmp);
			/***********************************************/

			//set parameters
			params_host[0] = data_A_size[curr_layer].channel;//channels
			params_host[1] = data_A_size[curr_layer].height;
			params_host[2] = data_A_size[curr_layer].width;
			params_host[3] = data_B_size[curr_layer].height;
			params_host[4] = data_B_size[curr_layer].width;
			params_host[5] = sizes[curr_layer];
			params_host[6] = params.iter;
			params_host[7] = range[curr_layer];

			//copy to device
			cudaMemcpy(params_device_AB, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

			//set parameters
			params_host[0] = data_B_size[curr_layer].channel;//channels
			params_host[1] = data_B_size[curr_layer].height;
			params_host[2] = data_B_size[curr_layer].width;
			params_host[3] = data_A_size[curr_layer].height;
			params_host[4] = data_A_size[curr_layer].width;

			//copy to device
			cudaMemcpy(params_device_BA, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

			////set device pa, device pb, device ann and device annd
			blocksPerGridAB = dim3(data_A_size[curr_layer].width / 20 + 1, data_A_size[curr_layer].height / 20 + 1, 1);
			threadsPerBlockAB = dim3(20, 20, 1);
			ann_size_AB = data_A_size[curr_layer].width* data_A_size[curr_layer].height;
			blocksPerGridBA = dim3(data_B_size[curr_layer].width / 20 + 1, data_B_size[curr_layer].height / 20 + 1, 1);
			threadsPerBlockBA = dim3(20, 20, 1);
			ann_size_BA = data_B_size[curr_layer].width* data_B_size[curr_layer].height;

			int num1 = data_A_size[curr_layer].channel*data_A_size[curr_layer].width*data_A_size[curr_layer].height;
			int num2 = data_A_size[next_layer].channel*data_A_size[next_layer].width*data_A_size[next_layer].height;

			float *target;
			cudaMalloc(&target, num1 * sizeof(float));
			avg_vote << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, data_BP[curr_layer], target, params_device_AB);
			deconv(&classifier_A, params.layers[curr_layer], target, data_A_size[curr_layer], params.layers[next_layer], data_AP[next_layer], data_A_size[next_layer]);
			cudaFree(target);

			num1 = data_B_size[curr_layer].channel*data_B_size[curr_layer].width*data_B_size[curr_layer].height;
			num2 = data_B_size[next_layer].channel*data_B_size[next_layer].width*data_B_size[next_layer].height;
			cudaMalloc(&target, num1 * sizeof(float));
			avg_vote << <blocksPerGridBA, threadsPerBlockBA >> >(ann_device_BA, data_A[curr_layer], target, params_device_BA);
			deconv(&classifier_B, params.layers[curr_layer], target, data_B_size[curr_layer], params.layers[next_layer], data_B[next_layer], data_B_size[next_layer]);
			cudaFree(target);

		}


	}

	//upsample
	int curr_layer = numlayer - 1;
	{
		//set parameters
		params_host[0] = 3;//channels
		params_host[1] = img_AL.rows;
		params_host[2] = img_AL.cols;
		params_host[3] = img_BPL.rows;
		params_host[4] = img_BPL.cols;
		params_host[5] = sizes[curr_layer];
		params_host[6] = params.iter;
		params_host[7] = range[curr_layer];
		//copy to device
		cudaMemcpy(params_device_AB, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		//set parameters
		params_host[0] = 3;//channels
		params_host[1] = img_BPL.rows;
		params_host[2] = img_BPL.cols;
		params_host[3] = img_AL.rows;
		params_host[4] = img_AL.cols;
		//copy to device
		cudaMemcpy(params_device_BA, params_host, param_size * sizeof(int), cudaMemcpyHostToDevice);

		////set device pa, device pb, device ann and device annd
		dim3 blocksPerGridAB(img_AL.cols / 20 + 1, img_AL.rows / 20 + 1, 1);
		dim3 threadsPerBlockAB(20, 20, 1);
		ann_size_AB = img_AL.cols* img_AL.rows;
		dim3 blocksPerGridBA(img_BPL.cols / 20 + 1, img_BPL.rows / 20 + 1, 1);
		dim3 threadsPerBlockBA(20, 20, 1);
		ann_size_BA = img_BPL.rows* img_BPL.cols;


		//updample
		unsigned int * ann_tmp;
		cudaMalloc(&ann_tmp, ann_size_AB * sizeof(unsigned int));
		upSample_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, ann_tmp, params_device_AB,
			data_A_size[curr_layer - 1].width, data_A_size[curr_layer - 1].height);//get new ann_device
		cudaMemcpy(ann_device_AB, ann_tmp, ann_size_AB * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		cudaFree(ann_tmp);

		cudaMalloc(&ann_tmp, ann_size_BA * sizeof(unsigned int));
		upSample_kernel << <blocksPerGridBA, threadsPerBlockBA >> >(ann_device_BA, ann_tmp, params_device_BA,
			data_B_size[curr_layer - 1].width, data_B_size[curr_layer - 1].height);//get new ann_device
		cudaMemcpy(ann_device_BA, ann_tmp, ann_size_BA * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
		cudaFree(ann_tmp);

		cudaMemcpy(ann_host_AB, ann_device_AB, ann_size_AB * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		cudaMemcpy(ann_host_BA, ann_device_BA, ann_size_BA * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//free space in device, only need to free pa and pb which are created temporarily
		//image downBAale
		Mat flow, result_AB, result_BA, err, out, normal;

		flow = reconstruct_dflow(img_AL, img_BPL, ann_host_AB, sizes[curr_layer]);
		result_AB = reconstruct_avg(img_AL, img_BPL, ann_host_AB, sizes[curr_layer]);

		cv::resize(result_AB, out, Size(), (float)ori_A_cols / cur_A_cols, (float)ori_A_rows / cur_A_rows, INTER_CUBIC);
		sprintf(fname, "resultAB.png");
		imwrite(path_output + fname, out);

		flow = reconstruct_dflow(img_BPL, img_AL, ann_host_BA, sizes[curr_layer]);
		result_BA = reconstruct_avg(img_BPL, img_AL, ann_host_BA, sizes[curr_layer]);

		cv::resize(result_BA, out, Size(), (float)ori_BP_cols / cur_BP_cols, (float)ori_BP_rows / cur_BP_rows, INTER_CUBIC);
		sprintf(fname, "resultBA.png");
		imwrite(path_output + fname, out);

		if (photoTransfer)
		{
			cout << "Refining photo transfer." << endl;
			Mat filtered_AB, filtered_BA, filtered_A, filtered_B, refine_AB, refine_BA;
			Mat origin_A, origin_B, res_AB, res_BA;
			img_AL.convertTo(origin_A, CV_32FC3, 1/255.0);
			img_BPL.convertTo(origin_B, CV_32FC3, 1 / 255.0);
			result_AB.convertTo(res_AB, CV_32FC3, 1 / 255.0);
			result_BA.convertTo(res_BA, CV_32FC3, 1 / 255.0);

			WeightedLeastSquare(filtered_AB, origin_A, res_AB);
			WeightedLeastSquare(filtered_BA, origin_B, res_BA);
			WeightedLeastSquare(filtered_A, origin_A, origin_A);
			WeightedLeastSquare(filtered_B, origin_B, origin_B);

			refine_AB = origin_A + filtered_AB - filtered_A;
			refine_BA = origin_B + filtered_BA - filtered_B;

			sprintf(fname, "refineAB.png");
			refine_AB.convertTo(normal, CV_32FC3, 255.0);
			cv::resize(normal, out, Size(), (float)ori_A_cols / cur_A_cols, (float)ori_A_rows / cur_A_rows, INTER_CUBIC);
			imwrite(path_output + fname, out);

			sprintf(fname, "refineBA.png");
			refine_BA.convertTo(normal, CV_32FC3, 255.0);
			cv::resize(normal, out, Size(), (float)ori_BP_cols / cur_BP_cols, (float)ori_BP_rows / cur_BP_rows, INTER_CUBIC);
			imwrite(path_output + fname, out);

		}

	}

	cout << "Saving flow result." << "\n";

	//save ann
	{
		ofstream output1;
		char fname[256];
		sprintf(fname, "flowAB.txt");
		output1.open(path_output + fname);
		for (int y = 0; y < img_AL.rows; y++)
		for (int x = 0; x < img_AL.cols; x++)
		{
			unsigned int v = ann_host_AB[y*img_AL.cols + x];
			int xbest = INT_TO_X(v);
			int ybest = INT_TO_Y(v);
			output1 << xbest - x << " " << ybest - y << endl;
		}
		output1.close();

		ofstream output2;
		sprintf(fname, "flowBA.txt");
		output2.open(path_output + fname);
		for (int y = 0; y < img_BPL.rows; y++){
			for (int x = 0; x < img_BPL.cols; x++)
			{
				unsigned int v = ann_host_BA[y*img_BPL.cols + x];
				int xbest = INT_TO_X(v);
				int ybest = INT_TO_Y(v);
				output2 << xbest - x << " " << ybest - y << endl;
			}
		}
		output2.close();
	}

	cudaFree(params_device_AB);
	cudaFree(ann_device_AB);
	cudaFree(annd_device_AB);
	cudaFree(params_device_BA);
	cudaFree(ann_device_BA);
	cudaFree(annd_device_BA);

	free(ann_host_AB);
	free(annd_host_AB);
	free(ann_host_BA);
	free(annd_host_BA);
	free(params_host);

	for (int i = 0; i < numlayer; i++)
	{
		cudaFree(data_A[i]);
		cudaFree(data_BP[i]);
	}

	finish = clock();
	duration = (double)(finish - start) / CLOCKS_PER_SEC;
	cout << "Finished finding ann. Time : " << duration << endl;

	google::ShutdownGoogleLogging();
	classifier_A.DeleteNet();
	classifier_B.DeleteNet();
}
