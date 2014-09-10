/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file Logging.cpp
 */

#include "Logging.h"

Timer deviceMemcpyTimer;
Timer slackTimer;
Timer ioReadTimer;

std::string timestamp() {
	time_t rawtime;
	struct tm * timeinfo;

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	char timechar[32];
	strftime(timechar, 32, "[%T]", timeinfo);

	return std::string(timechar);
}
