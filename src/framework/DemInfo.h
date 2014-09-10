/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file DemInfo.h
 */

#ifndef DEM_H_
#define DEM_H_

#include "DemData.h"

enum ByteOrder {
	MSBFIRST, LSBFIRST, VMS_FFLOAT
};

/**
 * \brief Information about the DEM data
 */
struct DemInfo {
	int fileCols;
	int fileRows;
	double xllCorner;
	double yllCorner;
	float cellSize;
	DemData noDataValue;
	ByteOrder byteOrder;
};

#endif /* DEM_H_ */
