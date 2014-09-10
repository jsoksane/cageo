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
 * \file AddSurfacesAlgorithmGPUCUDA.h
 */

#ifndef ADDSURFACESALGORITHMGPUCUDA_H_
#define ADDSURFACESALGORITHMGPUCUDA_H_

#include "AddSurfacesAlgorithm.h"

/**
 * \brief An NVidia CUDA implementation of the AddSurfacesAlgorithm.
 */
template<typename T, typename U>
class AddSurfacesAlgorithm_GPU_CUDA : public AddSurfacesAlgorithm<T, U> {
public:
	AddSurfacesAlgorithm_GPU_CUDA() {};
	virtual ~AddSurfacesAlgorithm_GPU_CUDA() {};

    /**
     * \brief Transfer the arrays to the DEVICE and execute addSurfaces_CUDA()
     * function.
     */
	void performAddSurfaces(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);

private:
    /**
     * \brief Add the two array using addSurfacesKernel() CUDA kernel.
     */
	void addSurfaces_CUDA(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);
};

/**
 * \brief The CUDA kernel to add two array.
 */
template<typename T, typename U>
__global__
void addSurfacesKernel(
		T* output,
		T* inputA,
		U* inputB,
		T noDataValue,
		int dataWidth,
		int dataHeight);

#endif /* ADDSURFACESALGORITHMGPUCUDA_H_ */
