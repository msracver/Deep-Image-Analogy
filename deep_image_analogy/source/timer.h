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
 * File timer.h: Timing functionality for CUDA.
 *
 **/
#pragma once

#ifndef TIMER_H
#define TIMER_H

#include "error_checking.h"

#include <cstring>
#include <fstream>

#ifdef WIN32
	#ifdef LBFGS_BUILD_DLL
		#define LBFGS_API __declspec(dllexport)
	#else
		#define LBFGS_API __declspec(dllimport)
	#endif
#else
	#define LBFGS_API
#endif

class LBFGS_API timer
{
public:
	timer(const std::string measurementName);
	~timer();

	void start();
	float stop();
	float elapsed() const;
	void saveMeasurement() const;

	static std::string timerPrefix;

private:
	std::string m_measurementName;

	cudaEvent_t m_start;
	cudaEvent_t m_stop;

	bool m_timerRunning;

	float m_accumulatedTime;
};

#endif /* end of include guard: TIMER_H */
