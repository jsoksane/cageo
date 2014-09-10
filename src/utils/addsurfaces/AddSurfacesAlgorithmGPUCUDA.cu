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
 * \file AddSurfacesAlgorithmGPUCUDA.cu
 */

#include "AddSurfacesAlgorithmGPUCUDA.cuh"

template<typename T, typename U>
void AddSurfacesAlgorithm_GPU_CUDA<T, U>::performAddSurfaces(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	inputA->toDevice();
	inputB->toDevice();
	output->toDevice();

	addSurfaces_CUDA(inputA, inputB, output);
}

template void AddSurfacesAlgorithm_GPU_CUDA<float, float>::performAddSurfaces(CellGrid<float>*, CellGrid<float>*, CellGrid<float>*);

template<typename T, typename U>
void AddSurfacesAlgorithm_GPU_CUDA<T, U>::addSurfaces_CUDA(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	dim3 block(16, 16, 1);
	dim3 dims(inputA->width, inputA->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	addSurfacesKernel<<<grid, block>>> (
			output->getData(),
			inputA->getData(),
			inputB->getData(),
			this->noDataValue,
			inputA->width,
			inputA->height);
	cudaGetLastError();
}

template<typename T, typename U>
__global__
void addSurfacesKernel(
		T* output,
		T* inputA,
		U* inputB,
		T noDataValue,
		int dataWidth,
		int dataHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= dataWidth || y >= dataHeight)
		return;

	int index = y*dataWidth + x;

	T data = inputA[index];
	if(data != noDataValue) {
		output[index] = data + (T) inputB[index];
	}
}
