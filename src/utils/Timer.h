/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file Timer.h
 */
#ifndef TIMER_H_
#define TIMER_H_

#include <string>
#include <map>
#include <utility>
#include <vector>
#include <sys/time.h>

/**
 * \brief A simple class to measure the running times of arbitrary parts of
 * the program.
 */
class Timer {
private:
    /**
     * \brief The time when the timer was started last time.
     */
	timeval startTime;
    /**
     * \brief The time when the timer was stopped last time.
     */
    timeval endTime;
    /**
     * \brief The timedelta between the endTime and startTime.
     */
	timeval elapsedTime;
    /**
     * \brief The duration of the previous measurement.
     */
	float lastTime;

    /**
     * \brief A string telling what the timer is measuring.
     */
	std::string target;
    /**
     * \brief The cumulative measurement time.
     */
	float totalTime;
    /**
     * \brief The number of separate measurements.
     */
	int totalMeasurements;

    /**
     * \brief The state of the timer.
     */
	bool timerRunning;

public:
    /**
     * \brief Construct a timer without measuring target string.
     */
	Timer();
    /**
     * \brief The destructor.
     */
	virtual ~Timer();

    /**
     * \brief Start the timer.
     */
	void startTimer(std::string timingTarget);
    /**
     * \brief Stop the current measurement.
     *
     * If the timer is already running, do nothing.
     */
	void endTimer();
    /**
     * \brief Return the measuring target string.
     *
     * If the timer is not running, do nothing.
     *
     * This function calls cudaDeviceSynchronize() before measuring the finish
     * time.
     */
	std::string getTimingTarget();
    /**
     * \brief Return the cumulative sum of measurement times.
     */
	float getTotalTime();
    /**
     * \brief Return the average measurement duration.
     */
	float getAverageTime();
    /**
     * \brief Return the duration of the latest measurement.
     */
	float getLastTime();
};

#endif /* TIMER_H_ */
