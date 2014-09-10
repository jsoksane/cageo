/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file InitializeArrayAlgorithmCPUOpenMP.h
 */

#ifndef INITIALIZEARRAYALGORITHMCPUOPENMP_H_
#define INITIALIZEARRAYALGORITHMCPUOPENMP_H_

#include "InitializeArrayAlgorithm.h"

template<typename T>
class InitializeArrayAlgorithm_CPU_OpenMP : public InitializeArrayAlgorithm<T> {
public:
	InitializeArrayAlgorithm_CPU_OpenMP();
	virtual ~InitializeArrayAlgorithm_CPU_OpenMP() {};

	void performInitializeArray(CellGrid<T>* input);
};

template<typename T>
void InitializeArrayAlgorithm_CPU_OpenMP<T>::performInitializeArray(CellGrid<T>* input)
{
	T* inputData = input->getData();

#pragma omp parallel for
	for(int i = 0; i < input->size; i++) {
		inputData[i] = this->initializeValue;
	}
}


#endif /* INITIALIZEARRAYALGORITHMCPUOPENMP_H_ */
