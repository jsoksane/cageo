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
 * \file ConvolutionAlgorithmGPUCUDA.cu
 */

#include "ConvolutionAlgorithmGPUCUDA.cuh"

template<typename T>
void ConvolutionAlgorithm_GPU_CUDA<T>::performConvolution(RandomFieldClass<T>* sourceData)
{
	sourceData->toDevice();

	FilterModel filterModel = this->filterModel;
	int filterLength        = this->getFilterLength();
    int filterSize          = (filterModel == GAUSSIAN) ? filterLength : filterLength*filterLength;

    T* d_buffer;
    T* d_filter;
    CUDA( cudaMalloc((void **) &d_buffer, sourceData->size*sizeof(T)) );
    CUDA( cudaMemcpy(d_buffer, sourceData->getData(), sourceData->size*sizeof(T), cudaMemcpyDeviceToDevice) );

	CUDA( cudaMalloc((void **) &d_filter, filterSize*sizeof(T)) );
	CUDA( cudaMemcpy(d_filter, this->filter->getData(), filterSize*sizeof(T), cudaMemcpyHostToDevice) );

	if(filterModel == GAUSSIAN) {
		performConvolutionSeparable_CUDA(
				sourceData,
				d_buffer,
				d_filter,
				this->getFilterRadius());
		cudaGetLastError();

		CUDA( cudaFree(d_buffer) );
	}
	else if(filterModel == EXPONENTIAL) {
		performConvolution_CUDA(
				d_buffer,
				sourceData,
				d_filter,
				this->getFilterRadius());
		cudaGetLastError();

		sourceData->freeData();
		sourceData->setData(d_buffer);
	}

	CUDA( cudaFree(d_filter) );
}

template void ConvolutionAlgorithm_GPU_CUDA<RandomFieldType>::performConvolution(RandomFieldClass<RandomFieldType>*);

template<typename T>
void ConvolutionAlgorithm_GPU_CUDA<T>::performConvolution_CUDA(
		T* output,
		CellGrid<T>* input,
		T* kernel,
		int filterRadius)
{
	dim3 block(32, 32, 1);
	dim3 dims(input->width, input->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	convolutionKernel<<<grid, block>>>(
			output, input->getData(),
			kernel, filterRadius,
			input->width,    input->height);
	CUDA( cudaGetLastError() );
}

template<typename T>
void ConvolutionAlgorithm_GPU_CUDA<T>::performConvolutionSeparable_CUDA(
		CellGrid<T>* input,
		T* buffer,
		T* kernel,
		int filterRadius)
{
	dim3 block(32, 32, 1);
	dim3 dims(input->width, input->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	convolutionColsKernel<<<grid, block>>>(
			buffer, input->getData(),
			kernel, filterRadius,
			input->width, input->height);

	convolutionRowsKernel<<<grid, block>>>(
			input->getData(), buffer,
			kernel, filterRadius,
			input->width, input->height);
}

template<typename T>
__global__
void convolutionKernel(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= dataWidth || y >= dataHeight) {
    	return;
    }

	int filterLength = filterRadius*2 + 1;

	T sum = 0.0f;

    /* Check are we too close to the boundary. */
	if (x < filterRadius || x > dataWidth - filterRadius - 1 ||
		y < filterRadius || y > dataHeight - filterRadius - 1) {
        /* Yes, we may need to modify the indices. */
		for (int filterY = 0; filterY < filterLength; filterY++) {
			int sy = y - filterRadius + filterY;
			if (sy < 0)
				sy += dataHeight;
			else if (sy >= dataHeight)
				sy -= dataHeight;
			for (int filterX = 0; filterX < filterLength; filterX++) {
				int sx = x - filterRadius + filterX;
				if (sx < 0)
					sx += dataWidth;
				else if (sx >= dataWidth)
					sx -= dataWidth;

				sum += inputData[sy * dataWidth + sx] * 
                       filter[filterY * filterLength + filterX];
			}
		}
		outputData[y * dataWidth + x] = sum;
	} else {
        /* No, filter fits inside the data */
		for (int filterY = 0; filterY < filterLength; filterY++) {
			int sy = y - filterRadius + filterY;
			for (int filterX = 0; filterX < filterLength; filterX++) {
				int sx = x - filterRadius + filterX;
				sum += inputData[sy * dataWidth + sx] *
                       filter[filterY * filterLength + filterX];
			}
		}
		outputData[y * dataWidth + x] = sum;
	}
}

template<typename T>
__global__
void convolutionRowsKernel(
			T* outputData,
			T* inputData,
			T* filter,
			int filterRadius,
			int dataWidth, int dataHeight)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= dataWidth || y >= dataHeight) {
    	return;
    }

	int filterLength = filterRadius*2 + 1;

	if (x < filterRadius || x > dataWidth - filterRadius - 1)
	{
		T sum = 0.0f;
		int sx = x - filterRadius;
		for (int filterX = 0; filterX < filterLength; filterX++, sx++)
		{
			if(sx < 0)
				sx += dataWidth;
			else if(sx >= dataWidth)
				sx -= dataWidth;

			sum += inputData[y*dataWidth + sx]*filter[filterX];
		}
		outputData[y*dataWidth + x] = sum;
	}
	else {
		T sum = 0.0f;
		int sx = x - filterRadius;
		for (int filterX = 0; filterX < filterLength; filterX++, sx++)
		{
			sum += inputData[y*dataWidth + sx]*filter[filterX];
		}
		outputData[y*dataWidth + x] = sum;
	}
}

template<typename T>
__global__
void convolutionColsKernel(
			T* outputData,
			T* inputData,
			T* filter,
			int filterRadius,
			int dataWidth, int dataHeight)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= dataWidth || y >= dataHeight) {
    	return;
    }

	const int filterLength = filterRadius*2 + 1;

	if (y < filterRadius || y > dataHeight - filterRadius-1)
	{
		T sum = 0.0f;
		int sy = y - filterRadius;
		for(int filterY = 0; filterY < filterLength; filterY++, sy++)
		{
			if(sy < 0)
				sy += dataHeight;
			else if(sy >= dataHeight)
				sy -= dataHeight;

			sum += inputData[sy*dataWidth + x] * filter[filterY];
		}
		outputData[y*dataWidth + x] = sum;
	}
	else {
		T sum = 0.0f;
		int sy = (y - filterRadius);
		for(int filterY = 0; filterY < filterLength; filterY++, sy++)
		{
			sum += inputData[sy*dataWidth + x] * filter[filterY];
		}
		outputData[y*dataWidth + x] = sum;
	}
}
