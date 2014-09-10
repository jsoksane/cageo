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
 * \file StreamBurningAlgorithmGPUCUDA.cu
 */

#include "StreamBurningAlgorithmGPUCUDA.cuh"
#include "StreamBurningCommon.h"
#include "KernelUtilsFactory.h"
#include "DemData.h"
#include "CellGrid.h"

/**
 * \brief A list of the translation vectors shorter than \a bevelRadius in the
 * DEVICE memory.
 */
__constant__ CoordOffset circularSearchArray[MAX_CIRCULAR_ARRAY_SIZE];

template<typename T, typename U>
void StreamBurningAlgorithm_GPU_CUDA<T, U>::cudaMemcpyCircularSearchArrayToSymbol(CoordOffset* h_circularSeachArray)
{
	CUDA( cudaMemcpyToSymbol(circularSearchArray, h_circularSeachArray, MAX_CIRCULAR_ARRAY_SIZE*sizeof(CoordOffset)) );
}

/**
 * Creates an empty data object used as a mask, initialize with ones using
 * InitializeArrayAlgorithm_GPU_CUDA, burn the stream data on it using
 * createStreamBurnTemplateKernel() CUDA kernel, and multiply the \a dem
 * with the mask.
 */
template<typename T, typename U>
void StreamBurningAlgorithm_GPU_CUDA<T, U>::performStreamBurning(
		CellGrid<T>* output,
		CellGrid<T>* dem,
		CellGrid<U>* streams)
{
	dem->toDevice();
	streams->toDevice();
	output->toDevice();

	CellGrid<float> streamBurnTemplate(*streams, AS_TEMPLATE);

	InitializeArrayAlgorithm_GPU_CUDA<float> initializeArrayAlgorithm;
	MultiplySurfacesAlgorithm_GPU_CUDA<T, U> multiplySurfacesAlgorithm;

	initializeArrayAlgorithm.setInitializeValue(1.0f);
	initializeArrayAlgorithm.execute(&streamBurnTemplate);

	cudaMemcpyCircularSearchArrayToSymbol(this->circularSearchArray->getData());
	createStreamBurnTemplate_CUDA(&streamBurnTemplate, streams);


	multiplySurfacesAlgorithm.setNoDataValue(this->noDataValueDEM);
	multiplySurfacesAlgorithm.execute(dem, &streamBurnTemplate, output);
}

template void StreamBurningAlgorithm_GPU_CUDA<DemData, DemData>::performStreamBurning(CellGrid<DemData>*, CellGrid<DemData>*, CellGrid<DemData>*);

template<typename T, typename U>
void StreamBurningAlgorithm_GPU_CUDA<T, U>::createStreamBurnTemplate_CUDA(
		CellGrid<T>* streamBurnTemplate,
		CellGrid<U>* streams)
{
	dim3 block(16, 16, 1);
	dim3 dims(streams->width, streams->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	createStreamBurnTemplateKernel<<<grid, block>>>(
			streamBurnTemplate->getData(),
			streams->getData(),
			this->noDataValueStream,
			this->bevelRadius,
			streams->width, streams->height);
	cudaGetLastError();
}

/**
 * Each thread operates on one cell. If the cell is located on a stream, set
 * \a streamBurnTemplate at the same location to zero. If the distance \a dist
 * of the cell to the closest stream is smaller than the \a bevelRadius, set
 * the value to the \a dist / \a bevelRadius.
 */
template<typename T, typename U>
__global__
void createStreamBurnTemplateKernel(
		T* streamBurnTemplate,
		U* streamData,
		U noDataValue,
		int bevelRadius,
		int width, int height)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
    	return;

	int index = y*width + x;

	if(streamData[index] != noDataValue) {
		streamBurnTemplate[index] = STREAM;
		return;
	}

	int dx, dy;

	for(int i = 0; i < MAX_CIRCULAR_ARRAY_SIZE; i++) {
		short2 p = circularSearchArray[i];

		if(p.x == 0 && p.y == 0)
			break;

		dx = x + p.x;
		if(dx < 0 || dx >= width)
			continue;

		dy = y + p.y;
		if(dy < 0 || dy >= height)
			continue;

		if(streamData[dy*width + dx] != noDataValue) {
			float dist = sqrtf(p.x*p.x + p.y*p.y);
			streamBurnTemplate[index] = (T) (dist/(float)bevelRadius);
			break;
		}
	}
}
