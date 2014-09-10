/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ConvertToByteArrayCPUOpenMP.h
 */

#ifndef TOBYTEARRAYALGORITHMCPUOPENMP_H_
#define TOBYTEARRAYALGORITHMCPUOPENMP_H_

#include "thrust/pair.h"
#include "thrust/extrema.h"
#include <math.h>
#include <omp.h>

template<typename T>
class ToByteArrayAlgorithm_CPU_OpenMP: public ToByteArrayAlgorithm<T> {
public:
    ToByteArrayAlgorithm_CPU_OpenMP() {};
    virtual ~ToByteArrayAlgorithm_CPU_OpenMP() {};

    void convertToByteArray(CellGrid<Byte_t>* output,
                            CellGrid<T>*      input);
};

template<typename T>
void ToByteArrayAlgorithm_CPU_OpenMP<T>::convertToByteArray(
		CellGrid<Byte_t>* output,
		CellGrid<T>*      input)
{
	input->toHost();
	output->toHost();

    T* data_in = input->getData();
    Byte_t* data_out = output->getData();

	thrust::pair<T*, T*> result;

	result = thrust::minmax_element(data_in, data_in + input->size);
	float min = *result.first;
	float max = *result.second;

    #pragma omp parallel for
    for (int i = 0; i < input->size; i++) {
        data_out[i] = (Byte_t) rintf(255.0f * ((data_in[i] - min) / max));
    }
}

template void ToByteArrayAlgorithm_CPU_OpenMP<float>::convertToByteArray(CellGrid<Byte_t>*, CellGrid<float>*);

#endif
