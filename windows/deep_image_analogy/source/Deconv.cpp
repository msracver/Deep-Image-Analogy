#include "Deconv.h"



void my_cost_function::f_gradf(const float *d_x, float *d_f, float *d_gradf)
{
	cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);

	m_classifier->net_->ForwardFromTo(m_id2 + 1, m_id1);
	const float* src = m_classifier->net_->blob_by_name(m_layer1)->gpu_data();

	float* diff = m_classifier->net_->blob_by_name(m_layer1)->mutable_gpu_diff();
	caffe_gpu_sub(m_num1, src, m_dy, diff);


	float* diff2;
	cudaMalloc(&diff2, m_num1 * sizeof(float));
	caffe_gpu_mul(m_num1, diff, diff, diff2);

	float total;
	caffe_gpu_asum(m_num1, diff2, &total);


	m_classifier->net_->BackwardFromTo(m_id1, m_id2 + 1);

	cudaMemcpy(d_gradf, m_classifier->net_->blob_by_name(m_layer2)->gpu_diff(), m_num2*sizeof(float), cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_f, &total, sizeof(float), cudaMemcpyHostToDevice);


	cudaFree(diff2);

}

void string_replace(string&s1, const string&s2, const string&s3)
{
	string::size_type pos = 0;
	string::size_type a = s2.size();
	string::size_type b = s3.size();
	while ((pos = s1.find(s2, pos)) != string::npos)
	{
		s1.replace(pos, a, s3);
		pos += b;
	}
}

void deconv(Classifier* classifier, string layer1, float* d_y, Dim dim1, string layer2, float* d_x, Dim dim2)
{

	int num1 = dim1.channel*dim1.height*dim1.width;
	int num2 = dim2.channel*dim2.height*dim2.width;
	int id1;
	int id2;

	string m_layer1 = layer1;
	string m_layer2 = layer2;

	string_replace(layer1, "conv", "relu");
	string_replace(layer2, "conv", "relu");
	if (layer2 == string("data")) layer2 = string("input");
	vector<string> layer_names = classifier->net_->layer_names();
	for (int i = 0; i < layer_names.size(); i++)
	{
		if (layer_names[i] == layer1)
			id1 = i;
		if (layer_names[i] == layer2)
			id2 = i;
	}


	my_cost_function func(classifier, m_layer1, d_y, num1, m_layer2, num2, id1, id2);

	lbfgs solver(func, Caffe::cublas_handle());

	lbfgs::status s = solver.minimize(d_x);

	cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);

	classifier->net_->ForwardFromTo(id2 + 1, id1);

}
