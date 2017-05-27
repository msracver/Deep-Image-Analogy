#include "Classifier.h"



Classifier::Classifier(const string& model_file,

	const string& trained_file
	) {



	Caffe::set_mode(Caffe::GPU);

	/* Load the network. */

	net_ = new Net<float>(model_file, TEST);

	net_->CopyTrainedLayersFrom(trained_file);

	

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";

	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";



	Blob<float>* input_layer = net_->input_blobs()[0];

	num_channels_ = input_layer->channels();

	CHECK(num_channels_ == 3 || num_channels_ == 1)

		<< "Input layer should have 1 or 3 channels.";




	/* Load the binaryproto mean file. */
	mean_ = cv::Scalar(103.939, 116.779, 123.68);

}


/* Load the mean file in binaryproto format. */




void Classifier::Predict(const cv::Mat& img, std::vector<std::string>& layers, std::vector<float *>& data_s, std::vector<float *>& data_d, std::vector<Dim>& size) {

	input_geometry_.width = img.cols;
	input_geometry_.height = img.rows;

	Blob<float>* input_layer = net_->input_blobs()[0];

	input_layer->Reshape(1, num_channels_,

		input_geometry_.height, input_geometry_.width);

	/* Forward dimension change to all layers. */

	net_->Reshape();


	std::vector<cv::Mat> input_channels;

	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();


	for (int i = 0; i < layers.size(); i++)
	{
		const shared_ptr<Blob<float> >& output_layer = net_->blob_by_name(layers[i]);

		int num = output_layer->num();
		int channel = output_layer->channels();
		int height = output_layer->height();
		int width = output_layer->width();
		size[i] = Dim(channel, height, width);

	
		cudaMalloc(&data_d[i], channel*height*width*sizeof(float));
		cudaMemcpy(data_d[i], output_layer->gpu_data(), channel*height*width*sizeof(float), cudaMemcpyDeviceToDevice);
		
		data_s[i] = output_layer->mutable_gpu_data();
		
	}
	

}

/* Wrap the input layer of the network in separate cv::Mat objects

* (one per channel). This way we save one memcpy operation and we

* don't need to rely on cudaMemcpy2D. The last preprocessing

* operation will write the separate channels directly to the input

* layer. */

void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {

	Blob<float>* input_layer = net_->input_blobs()[0];



	int width = input_layer->width();

	int height = input_layer->height();

	float* input_data = input_layer->mutable_cpu_data();

	for (int i = 0; i < input_layer->channels(); ++i) {

		cv::Mat channel(height, width, CV_32FC1, input_data);

		input_channels->push_back(channel);

		input_data += width * height;

	}

}



void Classifier::Preprocess(const cv::Mat& img,

	std::vector<cv::Mat>* input_channels) {

	/* Convert the input image to the input image format of the network. */

	cv::Mat sample;

	if (img.channels() == 3 && num_channels_ == 1)

		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);

	else if (img.channels() == 4 && num_channels_ == 1)

		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);

	else if (img.channels() == 4 && num_channels_ == 3)

		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);

	else if (img.channels() == 1 && num_channels_ == 3)

		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);

	else

		sample = img;

	

	cv::Mat sample_float;

	if (num_channels_ == 3)

		sample.convertTo(sample_float, CV_32FC3);

	else

		sample.convertTo(sample_float, CV_32FC1);



	cv::Mat sample_normalized;

	cv::subtract(sample_float, mean_, sample_normalized);



	/* This operation will write the separate BGR planes directly to the

	* input layer of the network because it is wrapped by the cv::Mat

	* objects in input_channels. */

	cv::split(sample_normalized, *input_channels);



	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)

		== net_->input_blobs()[0]->cpu_data())

		<< "Input channels are not wrapping the input layer of the network.";

}




/* visualize the squre with image */
void Classifier::Square_draw(std::vector<float>& vector, cv::Mat& img)
{
	float min = *(std::min_element(vector.begin(), vector.end()));
	float max = *(std::max_element(vector.begin(), vector.end()));

	int  height = img.rows;
	int  width = img.cols;


	for (int y = 0; y < height; y++)
	for (int x = 0; x < width; x++)
		img.at<uchar>(y, x) = (vector[y*width + x] - min) / (max - min) * 255;



}


void Classifier::Factorization(int channel, int &cols, int &rows)
{
	int ref = sqrt(channel);

	for (cols = ref; cols < channel; cols++)
	{
		if (channel%cols == 0)
		{
			rows = channel / cols;
			break;
		}
	}

}
