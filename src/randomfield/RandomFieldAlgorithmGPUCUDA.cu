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
 * \file RandomFieldAlgorithmGPUCUDA.cu
 */

#include "RandomFieldAlgorithmGPUCUDA.cuh"

template<typename T>
void RandomFieldAlgorithm_GPU_CUDA<T>::generateRandomField(
		RandomFieldClass<T>* randomField,
		int seed)
{
	randomField->toDevice();

	generateRandomField_CUDA(randomField, seed);
}

template void RandomFieldAlgorithm_GPU_CUDA<RandomFieldType>::generateRandomField(RandomFieldClass<RandomFieldType>*, int);

template<typename T>
void RandomFieldAlgorithm_GPU_CUDA<T>::generateRandomField_CUDA(
		RandomFieldClass<T>* randomField,
		int seed)
{
	const dim3 blockSize      = dim3(64, 1, 1);
	const int  randsPerThread = 1024;
	const int  randsPerBlock  = blockSize.x*randsPerThread;
	const int  maxGridSize    = CudaUtils::deviceProp.maxGridSize[0];

	dim3 gridSize  = dim3(min(maxGridSize, (randomField->size + (randsPerBlock-1))/randsPerBlock), 1, 1);

	curandState* d_randStates;
	CUDA( cudaMalloc((void**)&d_randStates, gridSize.x*blockSize.x*sizeof(curandState)) );

	setupCurandStatesKernel<<<gridSize, blockSize>>>(
			d_randStates,
			seed);

	generateRandomFieldKernel<<<gridSize, blockSize>>>(
			randomField->getData(),
			randomField->size,
			randsPerBlock,
			randsPerThread,
			d_randStates,
			this->mean,
			this->standardDeviation);

	CUDA( cudaFree(d_randStates) );
}

__global__
void setupCurandStatesKernel(
		curandState* randState,
		int seed)
{
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	curand_init(seed, idx, 0, &randState[idx]);
}

template<typename T>
__global__
void generateRandomFieldKernel(
		T* randomSurface,
		int randSize,
		int randsPerBlock,
		int randsPerThread,
		curandState* randState,
		float mean,
		float standardDeviation)
{
	int id  = blockIdx.x*blockDim.x    + threadIdx.x;
	int idx = blockIdx.x*randsPerBlock + threadIdx.x;
	curandState state = randState[id];

	for(int i = 0; i < randsPerThread && idx < randSize; i++, idx += blockDim.x) {
		randomSurface[idx] = (T) (mean + standardDeviation*curand_normal(&state));
	}
}
