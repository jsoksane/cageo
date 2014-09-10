/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file Utils.h
 */
#ifndef UTILS_H_
#define UTILS_H_

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <fftw3.h>
#include <errno.h>

#include "CellGrid.h"
#include <omp.h>

#include <cuda_runtime.h>
#include "DemInfo.h"
#include "Timer.h"

using namespace std;

//Align a to nearest higher multiple of b
int alignUp(int a, int b);

int isPowerOfTwo(unsigned int x);

cudaError_t cudaMemcpyTimed(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

namespace Utils {

inline
int pow2roundup (int x) {
    if (x < 0)
        return 0;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

}

template<typename T, typename U>
void performAddSurfacesCPU(
		CellGrid<T>& surfaceA,
		CellGrid<U>& surfaceB,
		T noDataValue)
{
	surfaceA.toHost();
	surfaceB.toHost();

	T* surfaceAData = surfaceA.getData();
	U* surfaceBData = surfaceB.getData();

#pragma omp parallel for
	for(int i = 0; i < surfaceA.size; ++i) {
		T data = surfaceAData[i];
		if(data != noDataValue) {
			surfaceAData[i] = data + (T) surfaceBData[i];
		}
	}
}

template<typename T>
void performRemoveNegativeElevationsCPU(
		CellGrid<T>& surface,
		T noDataValue)
{
	surface.toHost();
	T* surfaceData = surface.getData();

#pragma omp parallel for
	for(int i = 0; i < surface.size; i++) {
		T data = surfaceData[i];
		if(data <= 0.0 && data != noDataValue) {
			surfaceData[i] = 1.0;
		}
	}
}

void loadHeaderData(string fileName, DemInfo* demInfo);

#endif /* UTILS_H_ */
