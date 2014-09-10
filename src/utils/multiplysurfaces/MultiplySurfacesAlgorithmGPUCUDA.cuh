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
 * \file MultiplySurfacesAlgorithmGPUCUDA.h
 */

#ifndef MULTIPLYSURFACESALGORITHMGPUCUDA_H_
#define MULTIPLYSURFACESALGORITHMGPUCUDA_H_

#include "MultiplySurfacesAlgorithm.h"
#include "CellGrid.h"

/**
 * \brief A CUDA implementation of the MultiplySurfacesAlgorithm.
 */
template<typename T, typename U>
class MultiplySurfacesAlgorithm_GPU_CUDA : public MultiplySurfacesAlgorithm<T, U> {
public:
    /**
     * \brief The constructor.
     */
	MultiplySurfacesAlgorithm_GPU_CUDA() {};
    /**
     * \brief The destructor.
     */
	virtual ~MultiplySurfacesAlgorithm_GPU_CUDA() {};

    /**
     * \brief Transfer the data to the DEVICE and execute the
     * multiplySurfaces_CUDA().
     */
	void performMultiplySurfaces(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);

private:
    /**
     * \brief Calculate the CUDA parameters and launch the approriate number
     * of multiplySurfacesKernel() kernels.
     */
	void multiplySurfaces_CUDA(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);
};

/**
 * \brief The CUDA kernel to multiply the values of two array element-wise.
 */
template<typename T, typename U>
__global__
void multiplySurfacesKernel(
		T* surfaceA,
		U* surfaceB,
		T* result,
		T noDataValue,
		int dataWidth, int dataHeight);

#endif /* MULTIPLYSURFACESALGORITHMGPUCUDA_H_ */
