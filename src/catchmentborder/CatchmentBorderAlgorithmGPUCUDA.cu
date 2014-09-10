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
 * \file CatchmentBorderAlgorithmGPUCUDA.cu
 */

#include "CatchmentBorderAlgorithmGPUCUDA.cuh"
#include "FlowTracingCommon.h"

template<typename T, typename U>
void CatchmentBorderAlgorithm_GPU_CUDA<T, U>::performExtractDrainageBorder(
		CatchmentBorderClass<T>* catchmentBorder,
		FlowTraceDataClass<U>* flowTrace)
{
	catchmentBorder->toDevice();
	flowTrace->toDevice();

	extractDrainageBorder_CUDA(catchmentBorder, flowTrace);
}

template void CatchmentBorderAlgorithm_GPU_CUDA<CatchmentBorderType, FlowTraceDataType>::performExtractDrainageBorder(CatchmentBorderClass<CatchmentBorderType>*, FlowTraceDataClass<FlowTraceDataType>*);

template<typename T, typename U>
void CatchmentBorderAlgorithm_GPU_CUDA<T, U>::extractDrainageBorder_CUDA(
		CatchmentBorderClass<T>* catchmentBorder,
		FlowTraceDataClass<U>* flowTrace)
{
	dim3 block(32, 32, 1);
	dim3 dims(catchmentBorder->width, catchmentBorder->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	extractDrainageBorderKernel<<<grid, block>>>(
			catchmentBorder->getData(),
			flowTrace->getData(),
			catchmentBorder->width,    catchmentBorder->height);
}

/**
 * The \a flowTraceData is divided into separate areas with unique id numbers.
 * This kernel finds the borders of those areas and marks them into the \a
 * catchmentBorderData.
 *
 * The cell is defined to be located on the border, if
 * - it is not located on the lake and
 * - it has a neighbour who has larger flowTraceData value and it is not on
 *   the lake.
 *
 *
 *       flowTraceData        catchmentBorderData
 *
 *     1 1 4 4 4 4 3 3 3       . # . . . . # . .
 *     1 1 1 4 4 4 3 3 3       . # # . . . # . .
 *     1 1 1 4 4 3 3 3 2       . . # . . # # . #
 *     1 1 1 4 3 3 3 2 2   =>  . . # . # # . # #
 *     1 1 1 1 3 3 3 2 2       . . # # # . . # .
 *     1 1 1 3 3 3 2 2 2       . . # . . # # # .
 *     1 1 3 3 3 3 2 2 2       . # . . . . # . .
 *
 * In the \a catchmentBorderData the cells on the border have value 1, other
 * cells are 0.
 *
 * \param catchmentBorderData The array where to store the extracted borders.
 * \param flowTraceData The array of id numbers of the streams to where the
 *        flow paths of the cells have been traced.
 * \param width The width of the data.
 * \param height The height of the data.
 */
template<typename T, typename U>
__global__
void extractDrainageBorderKernel(
		T* catchmentBorderData,
		U* flowTraceData,
		int width,    int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 0 || x >= width || y < 0 || y >= height) {
    	return;
    }

	int index = y * width + x;
	U centerValue = flowTraceData[index];

	if (centerValue != FLOWTRACE_LAKE) {
		for(int n = 0; n < 8; n++) {
			int nx, ny, dx, dy;

			switch(n) {
				case 0: dx =  0; dy = -1; break;
				case 1: dx =  1; dy = -1; break;
				case 2: dx =  1; dy =  0; break;
				case 3: dx =  1; dy =  1; break;
				case 4: dx =  0; dy =  1; break;
				case 5: dx = -1; dy =  1; break;
				case 6: dx = -1; dy =  0; break;
				case 7: dx = -1; dy = -1; break;
			}

			nx = x + dx;
			ny = y + dy;

			if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
				continue;
			}

			int nIndex = ny * width + nx;
			if (centerValue < flowTraceData[nIndex] && flowTraceData[nIndex] != FLOWTRACE_LAKE) {
				catchmentBorderData[index] = 1.0f;
				return;
			}
		}
	}
}
