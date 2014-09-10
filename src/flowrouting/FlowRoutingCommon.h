/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowRoutingCommon.h
 */
#ifndef FLOWROUTINGCOMMON_H_
#define FLOWROUTINGCOMMON_H_

#include <vector>
#include "cuda_runtime.h"

#define MATH_SQRTTWO 1.41421356237309505f

#define FD_HAS_FLOW_DIR -1
#define FD_STREAM       -2
#define FD_NO_DATA      -3

typedef uint2 Coord;

#endif /* FLOWROUTINGCOMMON_H_ */
