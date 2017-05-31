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
 * File lbfgs.cpp: Implementation of class lbfgs (cpu_lbfgs only).
 *
 **/

#include "lbfgs.h"

#include <algorithm>
#include <fstream>
#include <vector>
using namespace std;

#ifndef LBFGS_BUILD_CPU_IMPLEMENTATION

lbfgs::status lbfgs::cpu_lbfgs(float *h_x)
{
	cerr << "For using cpu_lbfgs, you must enable building the CPU version in CMake." << endl;
	exit(-1);
}

#else

#include "linesearch_cpu.h"

lbfgs::status lbfgs::cpu_lbfgs(float *h_x)
{
	const size_t NX = m_costFunction.getNumberOfUnknowns();

	floatdouble *d_x = new floatdouble[NX];

	for (size_t idx = 0; idx < NX; ++idx)
		d_x[idx] = h_x[idx];

	VectorX xk = VectorX::Map(d_x, NX); // x_k,     current solution
	VectorX gk(NX);                     // g_k,     gradient at x_k
	VectorX gkm1(NX);                   // g_{k-1}, gradient at x_{k-1}
	VectorX z(NX);                      // z,       search direction
	floatdouble fk;                     // f_k,     value at x_k
	floatdouble fkm1;                   // f_{k-1}, value at x_{k-1}
	floatdouble H0 = 1.0f;              // H_0,     initial inverse Hessian (diagonal, same value for all elements)

	// treat arrays as ring buffers!
	VectorX s[HISTORY_SIZE];            // s,       history of solution updates
	VectorX y[HISTORY_SIZE];            // y,       history of gradient updates
	floatdouble alpha[HISTORY_SIZE];    // alpha,   history of alphas (needed for z updates)
	floatdouble rho  [HISTORY_SIZE];    // rho,     history of rhos   (needed for z updates)

	for (size_t i = 0; i < HISTORY_SIZE; ++i)
	{
		s[i] = VectorX(NX);
		y[i] = VectorX(NX);
	}

	cpu_cost_function *cpucf = (cpu_cost_function*)&m_costFunction;

	cpucf->cpu_f_gradf(xk.data(), &fk, gk.data());

	size_t evals = 1;

	status stat = LBFGS_REACHED_MAX_ITER;

#ifdef LBFGS_VERBOSE
	std::cout << "lbfgs::cpu_lbfgs()" << std::endl;
#endif
	size_t it;
	for (it = 0; it < m_maxIter; ++it)
	{

#ifdef LBFGS_VERBOSE
		printf("f(x) = % 12e, ||grad||_2 = % 12e\n", fk, gk.norm());
#endif

		// Check for convergence
		// ---------------------

		floatdouble xSquaredNorm = std::max<floatdouble>(1.0f, xk.squaredNorm());

		if (gk.squaredNorm() < (m_gradientEps * m_gradientEps) * xSquaredNorm)
		{
			stat = LBFGS_BELOW_GRADIENT_EPS;
			break;
		}

		// Find search direction
		// ---------------------

		z = -gk;

		const size_t MAX_IDX = std::min<size_t>(it, HISTORY_SIZE);

		for (size_t i = 1; i <= MAX_IDX; ++i)
		{
			const size_t idx = index(it - i);

			alpha[idx] = s[idx].dot(z) * rho[idx];
			z -= alpha[idx] * y[idx];
		}

		z *= H0;

		for (size_t i = MAX_IDX; i > 0; --i)
		{
			const size_t idx = index(it - i);

			const floatdouble beta = rho[idx] * y[idx].dot(z);
			z += s[idx] * (alpha[idx] - beta);
		}

		// Perform backtracking line search
		// --------------------------------

		gkm1 = gk;
		fkm1 = fk;

		floatdouble step;

		if (!cpu_linesearch(xk, z, cpucf, fk, gk, evals, gkm1, fkm1, stat, step, m_maxEvals))
		{
			break;
		}

		// Update s, y, rho and H_0
		// ------------------------

		s[index(it)] = z * step;  // = x_k - x_{k-1}
		y[index(it)] = gk - gkm1;

		floatdouble yDotS = y[index(it)].dot(s[index(it)]);

		rho[index(it)] = 1.0f / yDotS;

		floatdouble yNorm2 = y[index(it)].squaredNorm();
		if (yNorm2 > 1e-5f)
			H0 = yDotS / yNorm2;
	}

	for (size_t i = 0; i < NX; ++i)
		h_x[i] = float(xk[i]);

#ifdef LBFGS_VERBOSE
	std::cout << "Number of iterations: " << it << std::endl;
	std::cout << "Number of function/gradient evaluations: " << evals << std::endl;
	std::cout << "Reason for termination: " << statusToString(stat) << std::endl;
#endif

	return stat;
}

#endif
