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
 * \file RemoveNegativeElevationsAlgorithmGPUCUDA.h
 */

#ifndef REMOVENEGATIVEELEVATIONSALGORITHMGPUCUDA_H_
#define REMOVENEGATIVEELEVATIONSALGORITHMGPUCUDA_H_

#include "RemoveNegativeElevationsAlgorithm.h"

/**
 * \brief An NVidia CUDA implementation of the
 * RemoveNegativeElevationsAlgorithm.
 */
template<typename T>
class RemoveNegativeElevationsAlgorithm_GPU_CUDA : public RemoveNegativeElevationsAlgorithm<T> {
public:
	RemoveNegativeElevationsAlgorithm_GPU_CUDA() : RemoveNegativeElevationsAlgorithm<T>() {};
	virtual ~RemoveNegativeElevationsAlgorithm_GPU_CUDA() {};

    /**
     * \brief Transfer the data to the DEVICE and execute the
     * removeNegativeElevations_CUDA() function.
     */
	void performRemoveNegativeElevationValues(CellGrid<T>* input);

private:
    /**
     * \brief Calculate the CUDA parameters and launch the appropriate number of
     * removeNegativeElevationsKernel() CUDA kernels.
     */
	void removeNegativeElevations_CUDA(CellGrid<T>* input);
};

/**
 * \brief A CUDA kernel to replace negative values in the array with 1.0.
 */
template<typename T>
__global__
void removeNegativeElevationsKernel(
		T* surface,
		T noDataValue,
		int dataWidth,
		int dataHeight);

#endif /* REMOVENEGATIVEELEVATIONSALGORITHMGPUCUDA_H_ */
