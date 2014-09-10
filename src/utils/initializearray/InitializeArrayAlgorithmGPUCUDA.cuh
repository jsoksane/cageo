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
 * \file InitializeArrayAlgorithmGPUCUDA.h
 */

#ifndef INITIALIZEARRAYALGORITHMGPUCUDA_H_
#define INITIALIZEARRAYALGORITHMGPUCUDA_H_

#include "InitializeArrayAlgorithm.h"

/**
 * \brief A CUDA implementation of the InitializeArrayAlgorithm.
 */
template<typename T>
class InitializeArrayAlgorithm_GPU_CUDA : public InitializeArrayAlgorithm<T> {
public:
    /**
     * \brief The constructor.
     */
	InitializeArrayAlgorithm_GPU_CUDA() {};
    /**
     * \brief The destructor.
     */
	virtual ~InitializeArrayAlgorithm_GPU_CUDA() {};

    /**
     * \brief Transfer the input to the DEVICE and execute the
     * initializeArray_CUDA().
     */
	void performInitializeArray(CellGrid<T>* input);

private:
    /**
     * \brief Calculate the CUDA parameters and launch the appropriate number
     * of initArrayKernel() kernels.
     */
	void initializeArray_CUDA(CellGrid<T>* input);
};

/**
 * \brief A CUDA kernel to fill the given array with values.
 *
 * \param array An array of type T.
 * \param value A value of type T used to fill the array.
 * \param size The length of the array, i.e. the number of values in the
 *        array.
 */
template<typename T>
__global__
void initArrayKernel(
		T* array,
		const T value,
		const int size);

#endif /* INITIALIZEARRAYALGORITHMGPUCUDA_H_ */
