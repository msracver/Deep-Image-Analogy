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
 * File cost_function.cu: Implementation of cost function classes.
 *
 **/

#include "cost_function.h"
#include "error_checking.h"

void cpu_cost_function::f_gradf(const float *d_x, float *d_f, float *d_gradf)
{
	// Copy device x to host memory
	CudaSafeCall( cudaMemcpy(m_h_x, d_x, m_numDimensions * sizeof(float), cudaMemcpyDeviceToHost) );
	float h_f;

#ifdef LBFGS_CPU_DOUBLE_PRECISION
	std::cerr << "Don't try to use the GPU minimizer when LBFGS_CPU_DOUBLE_PRECISION is enabled." << std::endl;
	exit(EXIT_FAILURE);
#else
	cpu_f_gradf(m_h_x, &h_f, m_h_gradf);
#endif

	// Copy host f and gradf to device memory
	CudaSafeCall( cudaMemcpy(d_f,         &h_f,                   sizeof(float), cudaMemcpyHostToDevice) );
	CudaSafeCall( cudaMemcpy(d_gradf, m_h_gradf, m_numDimensions * sizeof(float), cudaMemcpyHostToDevice) );
}
