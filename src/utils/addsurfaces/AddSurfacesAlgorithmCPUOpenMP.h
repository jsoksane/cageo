/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file AddSurfacesAlgorithmCPUOpenMP.h
 */

#ifndef ADDSURFACESALGORITHMCPUOPENMP_H_
#define ADDSURFACESALGORITHMCPUOPENMP_H_

#include "AddSurfacesAlgorithm.h"

/**
 * \brief An OpenMP implementation of the AddSurfacesAlgorithm.
 */
template<typename T, typename U>
class AddSurfacesAlgorithm_CPU_OpenMP : public AddSurfacesAlgorithm<T, U> {
public:
	AddSurfacesAlgorithm_CPU_OpenMP() {};
	virtual ~AddSurfacesAlgorithm_CPU_OpenMP() {};

    /**
     * \brief The OpenMP-parallellized algorithm to add to arrays.
     */
	void performAddSurfaces(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);
};

template<typename T, typename U>
void AddSurfacesAlgorithm_CPU_OpenMP<T, U>::performAddSurfaces(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	inputA->toHost();
	inputB->toHost();

	T* outputData = output->getData();
	T* inputAData = inputA->getData();
	U* inputBData = inputB->getData();

#pragma omp parallel for
	for(int i = 0; i < inputA->size; ++i) {
		T data = inputAData[i];
		if(data != this->noDataValue) {
			outputData[i] = data + (T) inputBData[i];
		}
	}
}

#endif /* ADDSURFACESALGORITHMCPUCUDA_H_ */
