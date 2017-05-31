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
* File lbfgs.h: Interface of the minimizer.
*               This is the core class of the library.
*
**/

#ifndef LBFGS_H
#define LBFGS_H

#include "cost_function.h"
#include "error_checking.h"
#include "timer.h"

#ifdef WIN32
#ifdef LBFGS_BUILD_DLL
#define LBFGS_API __declspec(dllexport)
#else
#define LBFGS_API __declspec(dllimport)
#endif
#else
#define LBFGS_API
#endif

// Defines how many solution and gradient vector
// updates are kept to estimate the inverse Hessian
// during optimization.
#define HISTORY_SIZE 4

class LBFGS_API lbfgs
{
public:
	lbfgs(cost_function& cf, cublasHandle_t cublasHandle);
	~lbfgs();

	enum status {
		LBFGS_BELOW_GRADIENT_EPS,
		LBFGS_REACHED_MAX_ITER,
		LBFGS_REACHED_MAX_EVALS,
		LBFGS_LINE_SEARCH_FAILED
	};

	// Returns a string describing the status
	// indicated by the value of stat.
	static std::string statusToString(status stat);

	// Runs minimization of the cost function cf
	// using the L-BFGS method implemented in CUDA.
	//
	// d_x is the device memory location containing
	// the initial guess as cf.getNumberOfUnknowns()
	// consecutive floats. On output, d_x will
	// contain the solution of argmin_x(cf.f(x)) if
	// minimization succeeded, or the last solution
	// found when minimization was aborted.
	//
	// Returns a status code indicating why minimization
	// has stopped, see also lbfgs::status and
	// lbfgs::statusToString.
	status minimize(float *d_x);

	// Same as lbfgs::minimize, but the argument
	// is provided in host memory.
	status minimize_with_host_x(float *h_x);

	// Same as lbfgs::minimize.
	status gpu_lbfgs(float *d_x);

	// Runs minimization of the CPU cost function cf
	// using the CPU implementation of L-BFGS
	// (if enabled during compilation.
	status cpu_lbfgs(float *h_x);

	// The maximum number of iterations to be performed.
	//
	// Default value: 10000
	//
	size_t getMaxIterations() const         { return m_maxIter; }
	void   setMaxIterations(size_t maxIter) { m_maxIter = maxIter; }

	// The maximum number of function and gradient evaluations
	// (combined) to be performed.
	//
	// Default value: (unbounded)
	//
	size_t getMaxEvaluations() const          { return m_maxEvals; }
	void   setMaxEvaluations(size_t maxEvals) { m_maxEvals = maxEvals; }

	// The gradient epsilon represents a threshold for determining
	// if the current solution is already good enough.
	//
	// More precisely, the iteration is stopped as soon as
	// ||gradf(x)||^2 < gradientEpsilon * max(1, ||x||^2)
	//
	// Default value: 10^-4
	//
	float getGradientEpsilon() const            { return m_gradientEps; }
	void  setGradientEpsilon(float gradientEps) { m_gradientEps = gradientEps; }


private:
	cost_function& m_costFunction;

	size_t m_maxIter;
	size_t m_maxEvals;
	float  m_gradientEps;

	cublasHandle_t m_cublasHandle;

	// axpy  computes  dst = a * x + y
	// scale computes  dst = a * x
	// dot   computes  dst = x^T y
	//
	// x, y, dest (for axpy / scale) are n-vectors,
	// a,    dest (for dot)          are scalars.
	//
	// aDevicePointer / dstDevicePointer indicate whether
	// dst and a point to memory on the device or host.
	// All other pointers (marked with a d_) must point to device memory.

	void dispatch_axpy(const size_t n, float *d_dst, const float *d_y, const float *d_x, const float *a, bool aDevicePointer = true) const;
	void dispatch_scale(const size_t n, float *d_dst, const float *d_x, const float *a, bool aDevicePointer = true) const;
	void dispatch_dot(const size_t n, float *dst, const float *d_x, const float *d_y, bool dstDevicePointer = true) const;

	bool gpu_linesearch(float *d_x, float *d_z,
		float *d_fk, float *d_gk, size_t &evals, const float *d_gkm1,
		float *d_fkm1, lbfgs::status &stat, float *step, size_t maxEvals,
		timer *timer_evals, timer *timer_linesearch, float *d_tmp, int *d_status);

	static inline size_t index(size_t indexIn) { return indexIn % HISTORY_SIZE; }
};

#endif /* end of include guard: LBFGS_H */
