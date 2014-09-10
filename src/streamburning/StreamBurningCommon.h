/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file StreamBurningCommon.h
 */
#ifndef STREAMBURNINGCOMMON_H_
#define STREAMBURNINGCOMMON_H_

#include "cuda_runtime.h"

#define MAX_BEVEL_RADIUS 5
#define MAX_CIRCULAR_ARRAY_SIZE ((2*MAX_BEVEL_RADIUS + 1)*(2*MAX_BEVEL_RADIUS + 1))

typedef short2 CoordOffset;

/**
 * \brief A simple 2D vector that contains the x- and y-components and the
 * length.
 */
struct CoordDist {
    /**
     * \brief The length squared of the vector.
     */
	int d;
    /**
     * \brief The components.
     */
	CoordOffset p;
};

/**
 * \brief Return true, if the first vector is shorter than the second and
 * false, if the second is shorter.
 *
 * If the two vectors have same length, first y-coordinates and then
 * x-coordinates are compared separately.
 */
bool coordDistCompare(
		CoordDist& first,
		CoordDist& second);

#endif /* STREAMBURNINGCOMMON_H_ */
