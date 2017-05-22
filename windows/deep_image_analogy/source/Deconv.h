#ifndef DECONV_H
#define DECONV_H

#include "lbfgs.h"
#include "Classifier.h"

class LBFGS_API my_cost_function : public cost_function
{
public:
	my_cost_function(Classifier* classifier, string layer1, float* d_y, size_t num1, string layer2, size_t num2, int id1, int id2) : cost_function(num2){
		m_classifier = classifier;
		m_dy = d_y;
		m_num1 = num1;
		m_num2 = num2;
		m_id1 = id1;
		m_id2 = id2;

		m_layer1 = layer1;
		m_layer2 = layer2;

	}
	

	virtual ~my_cost_function(){};

	void f_gradf(const float *d_x, float *d_f, float *d_gradf);

private:
	Classifier* m_classifier;
	float *m_dy;
	int m_num1;
	int m_num2;
	int m_id1;
	int m_id2;
	string m_layer1;
	string m_layer2;

};

void string_replace(string&s1, const string&s2, const string&s3);

void deconv(Classifier* classifier, string layer1, float* d_y, Dim dim1, string layer2, float* d_x, Dim dim2);

#endif