/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file MultiplySurfacesAlgorithmCPUOpenMP.h
 */

#ifndef MULTIPLYSURFACESALGORITHMCPUOPENMP_H_
#define MULTIPLYSURFACESALGORITHMCPUOPENMP_H_

#include "MultiplySurfacesAlgorithm.h"

template<typename T, typename U>
class MultiplySurfacesAlgorithm_CPU_OpenMP : public MultiplySurfacesAlgorithm<T, U> {
public:
	MultiplySurfacesAlgorithm_CPU_OpenMP();
	virtual ~MultiplySurfacesAlgorithm_CPU_OpenMP() {};

	void performMultiplySurfaces(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);
};

template<typename T, typename U>
void MultiplySurfacesAlgorithm_CPU_OpenMP<T, U>::performMultiplySurfaces(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	inputA->toHost();
	inputB->toHost();
	output->toHost();

#pragma omp parallel for
	for(int i = 0; i < inputA->size; i++) {
		T valA = inputA[i];
		U valB = inputB[i];

		if(valA != this->noDataValue)
			output[i] = (T) (valA*valB);
	}
}


#endif /* MULTIPLYSURFACESALGORITHMCPUOPENMP_H_ */
