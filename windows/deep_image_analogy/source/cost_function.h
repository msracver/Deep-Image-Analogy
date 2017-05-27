/**
 *   ___ _   _ ___   _     _       ___ ___ ___ ___
 *  / __| | | |   \ /_\   | |  ___| _ ) __/ __/ __|
 * | (__| |_| | |) / _ \  | |_|___| _ \ _| (_ \__ \
 *  \___|\___/|___/_/ \_\ |____|  |___/_| \___|___/
 *                                               2012
 *     by Jens Wetzl           (jens.wetzl@fau.de)
 *    and Oliver Taubmann (oliver.taubmann@fau.de)
 *
 * This work is licensed under a Creative Commons
 * Attribution 3.0 Unported License. (CC-BY)
 * http://creativecommons.org/licenses/by/3.0/
 *
 * File cost_function.h: Cost function base classes.
 *                       Derive your own from one of them
 *                       and feed it to the minimizer.
 *
 **/

#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H

#include <iostream>
#include <cstdlib>

#ifdef WIN32
	#ifdef LBFGS_BUILD_DLL
		#define LBFGS_API __declspec(dllexport)
	#else
		#define LBFGS_API __declspec(dllimport)
	#endif
#else
	#define LBFGS_API
#endif

#ifdef LBFGS_CPU_float_PRECISION
	// 'real' is taken...
	typedef float floatfloat;
#else
	typedef float  floatfloat;
#endif

// GPU based cost functions can directly inherit from this class
// and implement f_gradf(), the passed pointers are supposed
// to reside in device memory.
class LBFGS_API cost_function
{
public:
	cost_function(size_t numDimensions)
		: m_numDimensions(numDimensions)
		{}

	virtual ~cost_function() {}
	
	// Implement this method computing both function value d_f
	// and gradient d_gradf of your cost function at point d_x.
	virtual void f_gradf(const float *d_x, float *d_f, float *d_gradf) = 0;

	size_t getNumberOfUnknowns() const
	{
		return m_numDimensions;
	}

protected:
	size_t m_numDimensions;
};


// CPU based cost functions can inherit from this class
// and implement cpu_f_gradf(), the results will be 
// written back to GPU memory automatically.
class LBFGS_API cpu_cost_function: public cost_function
{
public:
	cpu_cost_function(size_t numDimensions) : cost_function(numDimensions)
	{
		m_h_x     = new float[numDimensions];
		m_h_gradf = new float[numDimensions];
	}

	virtual ~cpu_cost_function()
	{
		delete [] m_h_x;
		delete [] m_h_gradf;
	}

	virtual void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf) = 0;

	void f_gradf(const float *d_x, float *d_f, float *d_gradf);

private:
	float *m_h_x;
	float *m_h_gradf;
};


#endif /* end of include guard: COST_FUNCTION_H */
