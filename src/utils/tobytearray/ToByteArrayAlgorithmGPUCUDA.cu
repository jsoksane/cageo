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
 * \file ConvertToByteArrayGPUCUDA.cu
 */

#include "ToByteArrayAlgorithmGPUCUDA.cuh"

#include <thrust/device_ptr.h>
#include <thrust/pair.h>
#include <thrust/extrema.h>

template<typename T>
void ToByteArrayAlgorithm_GPU_CUDA<T>::convertToByteArray(
		CellGrid<Byte_t>* output,
		CellGrid<T>*      input)
{
	input->toDevice();
	output->toDevice();

	using namespace thrust;
	device_ptr<T> d_ptr(input->getData());
	pair<device_ptr<T>, device_ptr<T> > result;

	result = minmax_element(d_ptr, d_ptr + input->size);
	float min = *result.first;
	float max = *result.second;

	convertToByteArray_CUDA(output, input, min, max);
}

template void ToByteArrayAlgorithm_GPU_CUDA<float>::convertToByteArray(CellGrid<Byte_t>*, CellGrid<float>*);

template<typename T>
void ToByteArrayAlgorithm_GPU_CUDA<T>::convertToByteArray_CUDA(
		CellGrid<Byte_t>* output,
		CellGrid<T>*      input,
		T min,
		T max)
{
	dim3 block(32, 32, 1);
	dim3 dims(input->width, input->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	convertToByteArrayKernel<<<grid, block>>>(
			output->getData(),
			input->getData(),
			min, max,
			input->width,
			input->height);
}

template<typename T, typename U>
__global__
void convertToByteArrayKernel(
		T* output,
		U* input,
		U min,
		U max,
		int dataWidth,
		int dataHeight)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
	int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= dataWidth || y >= dataHeight) {
		return;
	}

	int index = y*dataWidth + x;

	output[index] = (Byte_t) rintf(255.0f*((input[index] - min) / max));
}
