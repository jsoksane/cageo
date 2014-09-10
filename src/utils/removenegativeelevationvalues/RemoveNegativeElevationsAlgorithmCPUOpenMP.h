/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RemoveNegativeElevationsAlgorithmCPUOpenMP.h
 */

#ifndef REMOVENEGATIVEELEVATIONSALGORITHMCPUOPENMP_H_
#define REMOVENEGATIVEELEVATIONSALGORITHMCPUOPENMP_H_

#include "RemoveNegativeElevationsAlgorithm.h"

/**
 * \brief An OpenMP implementation of the RemoveNegativeElevationsAlgorithm.
 */
template<typename T>
class RemoveNegativeElevationsAlgorithm_CPU_OpenMP : public RemoveNegativeElevationsAlgorithm<T> {
public:
	RemoveNegativeElevationsAlgorithm_CPU_OpenMP() : RemoveNegativeElevationsAlgorithm<T>() {};
	virtual ~RemoveNegativeElevationsAlgorithm_CPU_OpenMP() {};

    /**
     * \brief Replace the negative values with 1.0 using OpenMP-parallelized
     * algorithm.
     */
	void performRemoveNegativeElevationValues(CellGrid<T>* input);
};

template<typename T>
void RemoveNegativeElevationsAlgorithm_CPU_OpenMP<T>::performRemoveNegativeElevationValues(CellGrid<T>* input)
{
	input->toHost();

	T* surfaceData = input->getData();

#pragma omp parallel for
	for(int i = 0; i < input->size; i++) {
		T data = surfaceData[i];
		if(data <= 0.0 && data != this->noDataValue) {
			surfaceData[i] = 1.0;
		}
	}
}

#endif /* REMOVENEGATIVEELEVATIONSALGORITHMCPUOPENMP_H_ */
