/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RandomFieldAlgorithmCPUOpenMP.h
 */

#ifndef RANDOMFIELDALGORITHMCPUOPENMP_H_
#define RANDOMFIELDALGORITHMCPUOPENMP_H_

#include <thrust/random.h>
#include <thrust/random/normal_distribution.h>
#include <limits>
#include <omp.h>

#include "RandomFieldAlgorithm.h"
#include "RandomFieldClass.h"

/**
 * \brief OpenMP implementation of the RandomFieldAlgorithm.
 */
template<typename T>
class RandomFieldAlgorithm_CPU_OpenMP : public RandomFieldAlgorithm<T> {
public:
	RandomFieldAlgorithm_CPU_OpenMP();
	virtual ~RandomFieldAlgorithm_CPU_OpenMP() {};

    /**
     * \brief Transfer the randomField to HOST and fill it with random
     * numbers.
     */
	void generateRandomField(
				RandomFieldClass<T>* randomField,
				int seed);
};

template<typename T>
RandomFieldAlgorithm_CPU_OpenMP<T>::RandomFieldAlgorithm_CPU_OpenMP()
{
}

template<typename T>
void RandomFieldAlgorithm_CPU_OpenMP<T>::generateRandomField(
		RandomFieldClass<T>* randomField,
		int seed)
{
	randomField->toHost();

	T* randomFieldData = randomField->getData();

#pragma omp parallel
{
	int seedSpec = seed*omp_get_num_threads() + omp_get_thread_num();
	thrust::random::default_random_engine randomGenerator(seedSpec);
	thrust::random::normal_distribution<T> dist(0.0, 1.0);

#pragma omp for
	for(int i = 0; i < randomField->size; i++) {
		T r = dist(randomGenerator);

		// There is a bug which can sometimes produce a '-inf' value
		// If this happens, just keep generating new numbers until
		// you get a non-inf number
		while(r == -std::numeric_limits<T>::infinity()) {
			r = dist(randomGenerator);
		}
		randomFieldData[i] = this->mean + r * this->standardDeviation;
	}
}

}

#endif /* RANDOMFIELDALGORITHMCPUOPENMP_H_ */
