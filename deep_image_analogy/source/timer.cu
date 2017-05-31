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
 * File timer.cu: Implementation of class timer.
 *
 **/

#include "timer.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

timer::timer(const std::string measurementName)
	: m_measurementName(measurementName)
	, m_timerRunning(false)
	, m_accumulatedTime(0.0f)
{
	CudaSafeCall( cudaEventCreate(&m_start) );
	CudaSafeCall( cudaEventCreate(&m_stop)  );
}

timer::~timer()
{
	if (m_timerRunning)
	{
		stop();
		saveMeasurement();
	}
	
	CudaSafeCall( cudaEventDestroy(m_start) );
	CudaSafeCall( cudaEventDestroy(m_stop)  );
}

void timer::start()
{
	m_timerRunning = true;
	CudaSafeCall( cudaEventRecord(m_start, 0) );
}
	
float timer::stop()
{
	CudaSafeCall( cudaEventRecord(m_stop, 0) );
	CudaSafeCall( cudaEventSynchronize(m_stop) );
	m_timerRunning = false;
		
	float elapsedTime;
	CudaSafeCall( cudaEventElapsedTime(&elapsedTime, m_start, m_stop) );

	m_accumulatedTime += elapsedTime;

	return elapsed();
}
	
void timer::saveMeasurement() const
{
	std::string filename(timerPrefix);
	filename.append(m_measurementName);
	filename.append(".txt");

	std::ofstream stream;
	stream.open(filename.c_str(), std::ios_base::app);
	stream << elapsed() << std::endl;
	stream.close();
}

float timer::elapsed() const
{
	return m_accumulatedTime;
}

std::string timer::timerPrefix = "";

