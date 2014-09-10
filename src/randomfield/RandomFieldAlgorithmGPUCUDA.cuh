/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * This software contains source code provided by NVIDIA Corporation.
 *
 *
 * \file RandomFieldAlgorithmGPUCUDA.h
 */

#ifndef RANDOMFIELDALGORITHMGPUCUDA_H_
#define RANDOMFIELDALGORITHMGPUCUDA_H_

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "RandomFieldAlgorithm.h"

/**
 * \brief A CUDA implementation of the RandomFieldAlgorithm.
 */
template<typename T>
class RandomFieldAlgorithm_GPU_CUDA : public RandomFieldAlgorithm<T> {
public:
	RandomFieldAlgorithm_GPU_CUDA();
	virtual ~RandomFieldAlgorithm_GPU_CUDA() {};

    /**
     * \brief Transfer the randomField to the device memory, and execute the
     * generateRandomField_CUDA kernel.
     */
	void generateRandomField(
				RandomFieldClass<T>* randomField,
				int seed);

protected:
    /**
     * \brief Fill the randomField with random numbers.
     *
     * Each thread needs initialized curandState object in order to produce
     * random numbers. It is more efficient to calculate many random numbers
     * per thread because then we don't have to initialize curandState objects
     * for every cell.
     *
     * The needed amount of curandState objects are initialized with
     * setupCurandStatesKernel() and the random numbers are calculated with
     * generateRandomFieldKernel().
     */
	void generateRandomField_CUDA(
			RandomFieldClass<T>* randomField,
			int seed);
};

template<typename T>
RandomFieldAlgorithm_GPU_CUDA<T>::RandomFieldAlgorithm_GPU_CUDA()
{
}

/**
 * \brief The CUDA kernel that initializes the CUDA random number state
 * variables.
 *
 * \param randState The curandState object to be initialized.
 * \param seed The seed used to initialize \a randState.
 */
__global__
void setupCurandStatesKernel(
		curandState* randState,
		int seed);

/**
 * \brief The CUDA kernel that computes the random numbers and stores them in
 * the given array.
 *
 * Random numbers \p r are generated using curand_normal() function, which
 * produces random numbers between 0 and 1. They are scaled using a formula
 *
 *     R = mean + standardDeviation * r
 *
 * to get a random number distribution {R} with desired properties.
 *
 * \param randomSurface The array to be filled with random numbers.
 * \param randSize The length of the \a randomSurface array.
 * \param randsPerBlock The number of random numbers calculated per thread
 *        block.
 * \param randsPerThread The number of random numbers calculater per thread.
 * \param randState The initialized curandState object.
 * \param mean The mean value for the produced random numbers.
 * \param standardDeviation The standard deviation for the produced random
 *        numbers.
 */
template<typename T>
__global__
void generateRandomFieldKernel(
		T* randomSurface,
		int randSize,
		int randsPerBlock,
		int randsPerThread,
		curandState* randState,
		float mean,
		float standardDeviation);

#endif /* RANDOMFIELDALGORITHMGPUCUDA_H_ */
