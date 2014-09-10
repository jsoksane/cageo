/*
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
 * \file InitializeArrayAlgorithmGPUCUDA.cu
 */

#include "InitializeArrayAlgorithmGPUCUDA.cuh"

template<typename T>
void InitializeArrayAlgorithm_GPU_CUDA<T>::performInitializeArray(CellGrid<T>* input)
{
    input->toDevice();

	initializeArray_CUDA(input);
}

template void InitializeArrayAlgorithm_GPU_CUDA<float>::performInitializeArray(CellGrid<float>*);
template void InitializeArrayAlgorithm_GPU_CUDA<int>::performInitializeArray(CellGrid<int>*);

template<typename T>
void InitializeArrayAlgorithm_GPU_CUDA<T>::initializeArray_CUDA(CellGrid<T>* input)
{
	dim3 block(512, 1, 1);
	dim3 dims(input->size, 1, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	initArrayKernel<<<grid, block>>> (
			input->getData(),
			this->initializeValue,
			input->size);
}

template<typename T>
__global__
void initArrayKernel(
		T* array,
		const T value,
		const int size)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;

	if(x >= size) {
		return;
	}

	array[x] = value;
}
