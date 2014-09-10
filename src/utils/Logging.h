/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file Logging.h
 */

#ifndef LOGGING_H_
#define LOGGING_H_

#include <time.h>
#include <string>

#include "Timer.h"
#include "GlobalParameters.h"

std::string timestamp();

extern Timer deviceMemcpyTimer;
extern Timer slackTimer;
extern Timer ioReadTimer;

#define PRINT_TIMESTAMP timestamp() << " "

#define LOG_TRACE_ENABLE
//#define LOG_TRACE_ALGORITHM_ENABLE
#define TIMING_ENABLE
//#define TIMING_DEVICE_MEMCPY_ENABLE
//#define TIMING_SLACK_ENABLE
#define TIMING_IO_READ_ENABLE

#ifdef LOG_TRACE_ENABLE
#define LOG_TRACE(x) \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TRACE] " << x << " (" << __FUNCTION__  << ")" << std::endl; \
		}
#else
#define LOG_TRACE(x)
#endif

#ifdef LOG_TRACE_ALGORITHM_ENABLE
#define LOG_TRACE_ALGORITHM(x) \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"  [TRACE] " << x << " (" << __FUNCTION__  << ")" << std::endl; \
		}
#else
#define LOG_TRACE_ALGORITHM(x)
#endif

#ifdef TIMING_ENABLE
#define TIMING_START(x, y) \
		{ \
			x.startTimer(y); \
		}
#define TIMING_END(x) \
		{ \
			x.endTimer(); \
		}
#define TIMING_PRINT(x) \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TIMING] " << x.getTimingTarget() << ": " << x.getLastTime() << " ms" << std::endl; \
		}
#define TIMING_PRINT_TOTAL(x) \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TIMING] " << x.getTimingTarget() << ": " << x.getTotalTime() << " ms (total)" << std::endl; \
		}
#define TIMING_PRINT_AVERAGE(x) \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TIMING] " << x.getTimingTarget() << ": " << x.getAverageTime() << " ms (average)" << std::endl; \
		}
#define TIMING_END_PRINT(x) TIMING_END(x) TIMING_PRINT(x)
#else
#define TIMING_START(x, y)
#define TIMING_END(x)
#define TIMING_PRINT(x)
#define TIMING_PRINT_TOTAL(x)
#define TIMING_PRINT_AVERAGE(x)
#define TIMING_END_PRINT(x)
#endif

#ifdef TIMING_DEVICE_MEMCPY_ENABLE
#define TIMING_DEVICE_MEMCPY_START() \
		{ \
			deviceMemcpyTimer.startTimer("Device memcpy"); \
		}
#define TIMING_DEVICE_MEMCPY_STOP() \
		{ \
			deviceMemcpyTimer.endTimer(); \
		}
#define TIMING_DEVICE_MEMCPY_PRINT_TOTAL() \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TIMING] " << deviceMemcpyTimer.getTimingTarget() << ": " \
			<< deviceMemcpyTimer.getTotalTime() << " ms (total)" << std::endl; \
		}
#else
#define TIMING_DEVICE_MEMCPY_START()
#define TIMING_DEVICE_MEMCPY_STOP()
#define TIMING_DEVICE_MEMCPY_PRINT_TOTAL()
#endif

#ifdef TIMING_SLACK_ENABLE
#define TIMING_SLACK_START() \
		{ \
			slackTimer.startTimer("Slack time"); \
		}
#define TIMING_SLACK_STOP() \
		{ \
			slackTimer.endTimer(); \
		}
#define TIMING_SLACK_PRINT_TOTAL() \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TIMING] " << slackTimer.getTimingTarget() << ": " \
			<< slackTimer.getTotalTime() << " ms (total)" << std::endl; \
		}
#else
#define TIMING_SLACK_START()
#define TIMING_SLACK_STOP()
#define TIMING_SLACK_PRINT_TOTAL()
#endif

#ifdef TIMING_IO_READ_ENABLE
#define TIMING_IO_READ_START() \
		{ \
			ioReadTimer.startTimer("IO read time"); \
		}
#define TIMING_IO_READ_STOP() \
		{ \
			ioReadTimer.endTimer(); \
		}
#define TIMING_IO_READ_PRINT_TOTAL() \
		{ \
			std::cerr << PRINT_TIMESTAMP << \
			"[TIMING] " << ioReadTimer.getTimingTarget() << ": " \
			<< ioReadTimer.getTotalTime() << " ms (total)" << std::endl; \
		}
#else
#define TIMING_SLACK_START()
#define TIMING_SLACK_STOP()
#define TIMING_SLACK_PRINT_TOTAL()
#endif

#endif /* LOGGING_H_ */
