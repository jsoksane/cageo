/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file StreamBurningAlgorithmCPUOpenMP.h
 */

#ifndef STREAMBURNINGALGORITHMCPUOPENMP_H_
#define STREAMBURNINGALGORITHMCPUOPENMP_H_

#include <math.h>

#include "StreamBurningAlgorithm.h"

template<typename T, typename U>
class StreamBurningAlgorithm_CPU_OpenMP : public StreamBurningAlgorithm<T, U> {
public:
	StreamBurningAlgorithm_CPU_OpenMP() {};
	virtual ~StreamBurningAlgorithm_CPU_OpenMP() {};

	void performStreamBurning(
			CellGrid<T>* output,
			CellGrid<T>* dem,
			CellGrid<U>* streams);
};

template<typename T, typename U>
void StreamBurningAlgorithm_CPU_OpenMP<T, U>::performStreamBurning(
		CellGrid<T>* output,
		CellGrid<T>* dem,
		CellGrid<U>* streams)
{
	output->toHost();
	dem->toHost();
	streams->toHost();

	T* outputData  = output->getData();
	T* demData     = dem->getData();
	U* streamsData = streams->getData();

#pragma omp parallel for
	for(int sy = 0; sy < dem->height; sy++) {
		for(int sx = 0; sx < dem->width; sx++) {
			int index = sy*streams->width + sx;

			if(streamsData[index] != this->noDataValueStream) {
				demData[index] = 0.0f;
			}

			CoordOffset* circularSearchArrayData = this->circularSearchArray->getData();

			for(int i = 0; i < MAX_CIRCULAR_ARRAY_SIZE; i++) {
				CoordOffset p = circularSearchArrayData[i];

				if(p.x == 0 && p.y == 0) break;

				int nsx = sx + p.x;
				int nsy = sy + p.y;

				if(nsx < 0 || nsy < 0 || nsx >= dem->width || nsy >= dem->height)
					continue;

				int nIndex = nsy*streams->width + nsx;

				if(streamsData[nIndex] != this->noDataValueStream &&
						  demData[nIndex] != this->noDataValueDEM)
				{
					float xDiff = (float) (sx - nsx);
					float yDiff = (float) (sy - nsy);
					float dist = sqrt(xDiff*xDiff + yDiff*yDiff);
					outputData[index] = demData[index] * (dist/this->bevelRadius);
					break;
				}
			}
		}
	}
}

#endif /* STREAMBURNINGALGORITHMCPUOPENMP_H_ */
