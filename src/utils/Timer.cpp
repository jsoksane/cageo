/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * This software contains source code provided by NVIDIA Corporation.
 *
 *
 * \file Timer.cpp
 */
#include "Timer.h"
#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

Timer::Timer()
{
	totalTime = 0.0f;
	totalMeasurements = 0;
	timerRunning = false;
}

Timer::~Timer()
{
	if(timerRunning) {
		endTimer();
	}
}

void Timer::startTimer(std::string timingTarget) {
	if(timerRunning) {
		printf("Timer is already running!\n");
		return;
	}
	else {
		timerRunning = true;
	}

	startTime.tv_sec  = endTime.tv_sec  = 0.0;
	startTime.tv_usec = endTime.tv_usec = 0.0;

	target = timingTarget;

	gettimeofday(&startTime, NULL);
}

void Timer::endTimer() {
	if(!timerRunning) {
		printf("Timer has already stopped!\n");
		return;
	}

	cudaDeviceSynchronize();

	gettimeofday(&endTime, NULL);
	timersub(&endTime, &startTime, &elapsedTime);

	float timeMillis = 1000.0*elapsedTime.tv_sec + elapsedTime.tv_usec/1000.0;

	lastTime = timeMillis;
	totalTime += timeMillis;
	totalMeasurements++;

	timerRunning = false;
}

std::string Timer::getTimingTarget() {
	return target;
}

float Timer::getTotalTime()
{
	return totalTime;
}

float Timer::getAverageTime() {
	return totalMeasurements > 0 ? totalTime/totalMeasurements : 0.0f;
}

float Timer::getLastTime()
{
	return lastTime;
}
