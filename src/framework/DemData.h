/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file DemData.h
 */
#ifndef DEMDATA_H_
#define DEMDATA_H_

#define STREAM 0

// REMEMBER to clean project when changing this
#define DEM_FLOAT

#ifdef DEM_FLOAT
typedef float DemData;
#else
typedef int DemData;
#endif

#endif /* DEMDATA_H_ */
