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
 * \file MultiplySurfacesAlgorithmGPUCUDA.cu
 */

#include "MultiplySurfacesAlgorithmGPUCUDA.cuh"
#include "DemData.h"

template<typename T, typename U>
void MultiplySurfacesAlgorithm_GPU_CUDA<T, U>::performMultiplySurfaces(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	inputA->toDevice();
	inputB->toDevice();
	output->toDevice();

	multiplySurfaces_CUDA(inputA, inputB, output);
}

template void MultiplySurfacesAlgorithm_GPU_CUDA<DemData, DemData>::performMultiplySurfaces(CellGrid<DemData>*, CellGrid<DemData>*, CellGrid<DemData>*);

template<typename T, typename U>
void MultiplySurfacesAlgorithm_GPU_CUDA<T, U>::multiplySurfaces_CUDA(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	dim3 block(16, 16, 1);
	dim3 dims(inputA->width, inputA->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	multiplySurfacesKernel<<<grid, block>>> (
			inputA->getData(),
			inputB->getData(),
			output->getData(),
			this->noDataValue,
			inputA->width,
			inputA->height);
}

template<typename T, typename U>
__global__
void multiplySurfacesKernel(
		T* surfaceA,
		U* surfaceB,
		T* result,
		T noDataValue,
		int dataWidth, int dataHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= dataWidth || y >= dataHeight)
		return;

	int index = y*dataWidth + x;

	T valA = surfaceA[index];
	U valB = surfaceB[index];

	if(valA == noDataValue)
		return;

	result[index] = (T) (valA*valB);
}
