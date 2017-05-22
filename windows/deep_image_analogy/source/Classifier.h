#pragma once

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>

#include <opencv2/highgui/highgui.hpp>

#include <opencv2/imgproc/imgproc.hpp>


#include <algorithm>

#include <iosfwd>

#include <memory>

#include <string>

#include <utility>

#include <vector>


using namespace caffe;  // NOLINT(build/namespaces)

using std::string;


class Dim{
public:
	int channel;
	int height;
	int width;
	Dim(){};
	Dim(int c, int h, int w)
	{
		channel = c;
		height = h;
		width = w;
	};
	~Dim(){};
};



class Classifier {

public:

	Classifier(const string& model_file,

		const string& trained_file
	
		);


	void Predict(const cv::Mat& img, std::vector<std::string>& layers, std::vector<float *>& data_s, std::vector<float *>& data_d, std::vector<Dim>& size);

	void Draw(float* data, Dim dim, string filename);

	void DeleteNet() { delete net_; }

private:

	void SetMean(const string& mean_file);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);


	void Preprocess(const cv::Mat& img,

		std::vector<cv::Mat>* input_channels);

	void Factorization(int channel, int &cols, int &rows);
	void Square_draw(std::vector<float>& vector, cv::Mat& img);


public:
	Net<float>* net_;

	cv::Size input_geometry_;

	int num_channels_;

	cv::Scalar mean_;

	std::vector<string> labels_;

};