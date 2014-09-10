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
 * \file RemoveNegativeElevationsAlgorithmGPUCUDA.cu
 */

#include "RemoveNegativeElevationsAlgorithmGPUCUDA.cuh"
#include "DemData.h"

template<typename T>
void RemoveNegativeElevationsAlgorithm_GPU_CUDA<T>::performRemoveNegativeElevationValues(CellGrid<T>* input)
{
	input->toDevice();
	removeNegativeElevations_CUDA(input);
}

template void RemoveNegativeElevationsAlgorithm_GPU_CUDA<DemData>::performRemoveNegativeElevationValues(CellGrid<DemData>*);

template<typename T>
void RemoveNegativeElevationsAlgorithm_GPU_CUDA<T>::removeNegativeElevations_CUDA(CellGrid<T>* input)
{
	dim3 block(16, 16, 1);
	dim3 dims(input->width, input->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	removeNegativeElevationsKernel<<<grid, block>>> (
			input->getData(),
			this->noDataValue,
			input->width,
			input->height);
}

/**
 * If the value of the cell for the thread is not \a noDataValue and is zero
 * or less, replace it with a value 1.0.
 *
 * \param surface The data array.
 * \param noDataValue The value which means that the cell does not have an
 *        elevation data.
 * \param dataWidth The width of the data.
 * \param dataHeight The height of the data.
 */
template<typename T>
__global__
void removeNegativeElevationsKernel(
		T* surface,
		T noDataValue,
		int dataWidth,
		int dataHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= dataWidth || y >= dataHeight) {
		return;
	}

	int index = y*dataWidth + x;
	T elevation = surface[index];

	if(elevation <= 0 && elevation != noDataValue) {
		surface[index] = 1.0;
	}
}
