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
 * \file FlowRoutingAlgorithmGPUCUDAOriginal.cu
 */

#include <cuda_runtime.h>
#include <limits.h>

#include "FlowRoutingAlgorithmGPUCUDAOriginal.cuh"
#include "FlowRoutingCommon.h"
#include "DemData.h"
#include "Utils.h"

#include "Reduce.cuh"
#include "ReduceUtils.h"

#define PROCESS_NEXT_SIZE 4096
#define BLOCK_SIZE_X      16
#define BLOCK_SIZE_Y      16
#define BLOCK_DIM         (BLOCK_SIZE_X*BLOCK_SIZE_Y)
#define BLOCK_DIM_REDUCE  BLOCK_DIM

/**
 * \brief Transfer the data of the given data objects to the DEVICE and start
 * determining the flow directions for the cells. If there are flat cells
 * found, they are processed later by performFlowRoutingOnFlatSurfaces()
 * function.
 */
template<typename T, typename U>
void FlowRoutingAlgorithm_GPU_CUDA_Original<T, U>::performFlowRouting(
		FlowDirClass<T>* flowDir,
		FlatDist_t*      flatDist,
		CellGrid<U>*     dem)
{
	flowDir->toDevice();
	flatDist->toDevice();
	dem->toDevice();

	flowDir->clearData();
	flatDist->clearData();

	flowRouting_CUDA(flowDir, flatDist, dem);
}

template void FlowRoutingAlgorithm_GPU_CUDA_Original<FlowDirDataType, DemData>::performFlowRouting(FlowDirClass<FlowDirDataType>*, FlatDist_t*, CellGrid<DemData>*);

/**
 * \brief Transfer the data of the given objects to the DEVICE and start
 * determining the flow directions for the flat cells.
 */
template<typename T, typename U>
void FlowRoutingAlgorithm_GPU_CUDA_Original<T, U>::performFlowRoutingOnFlatSurfaces(
		FlowDirClass<T>* flowDir,
		FlatDist_t*      flatDist,
		CellGrid<U>*     dem)
{
	flowDir->toDevice();
	flatDist->toDevice();
	dem->toDevice();

	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 dims(flowDir->width, flowDir->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	size_t blockDoneSize = grid.x * grid.y;
    size_t blockDoneReducedSize = calculate_reduced_size(blockDoneSize, BLOCK_DIM_REDUCE);

    // An array to keep track of which blocks have finished assigning the flow
    // directions, so that the kernels launched in subsequent iterations can
    // return immediately.
    bool* d_blockDone        = NULL;
	bool* d_blockDoneReduced = NULL;

	CUDA( cudaMalloc((void**) &d_blockDone,        blockDoneSize       *sizeof(bool)) );
	CUDA( cudaMalloc((void**) &d_blockDoneReduced, blockDoneReducedSize*sizeof(bool)) );

	CUDA( cudaMemset(d_blockDone,        0,        blockDoneSize*sizeof(bool)) );
	CUDA( cudaMemset(d_blockDoneReduced, 0, blockDoneReducedSize*sizeof(bool)) );

	bool* h_blockDoneReduced = (bool*) calloc(blockDoneReducedSize, sizeof(bool));

    /**
     * Execute the handleFlatSurfaces_CUDA() until all the flat cells have
     * been assigned a flow direction.
     */
	for(int iter = 1;; iter++) {
		handleFlatSurfaces_CUDA(flowDir, flatDist, dem, d_blockDone, iter);

        // Collect the status of the thread blocks to the reduced array
		reduceAnd<BLOCK_DIM_REDUCE><<<blockDoneReducedSize, BLOCK_DIM_REDUCE, BLOCK_DIM_REDUCE*sizeof(bool)>>>
					   (d_blockDone, d_blockDoneReduced, blockDoneSize);

        // Do the final reduction on the host
		CUDA( cudaMemcpy(h_blockDoneReduced, d_blockDoneReduced, blockDoneReducedSize*sizeof(bool), cudaMemcpyDeviceToHost) );

		bool allBlocksDone = true;
		for(int i = 0; i < blockDoneReducedSize; i++) {
			if(!h_blockDoneReduced[i]){
				allBlocksDone = false;
				break;
			}
		}
		if(allBlocksDone) {
			break;
		}
	}

	CUDA( cudaFree(d_blockDone) );
	CUDA( cudaFree(d_blockDoneReduced) );

	free(h_blockDoneReduced);
}

template void FlowRoutingAlgorithm_GPU_CUDA_Original<FlowDirDataType, DemData>::performFlowRoutingOnFlatSurfaces(FlowDirClass<FlowDirDataType>*, FlatDist_t*, CellGrid<DemData>*);

/**
 * \brief Determine the CUDA thread block grid size and execute the
 * flowRoutingKernel() CUDA kernel.
 */
template<typename T, typename U>
void FlowRoutingAlgorithm_GPU_CUDA_Original<T, U>::flowRouting_CUDA(
		FlowDirClass<T>* flowDir,
		FlatDist_t*      flatDist,
		CellGrid<U>*     dem)
{
	dim3 block(16, 16, 1);
	dim3 dims(dem->width, dem->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	flowRoutingKernel<<<grid, block>>>(
			flowDir->getData(), flatDist->getData(), dem->getData(),
			this->noDataValueDEM,
			dem->width,   dem->height);
	cudaGetLastError();
}

/**
 * \brief Determine the CUDA thread block grid size and execute the
 * handleFlatSurfacesKernel() CUDA kernel.
 */
template<typename T, typename U>
void FlowRoutingAlgorithm_GPU_CUDA_Original<T, U>::handleFlatSurfaces_CUDA(
		FlowDirClass<T>* flowDir,
		FlatDist_t*      flatDist,
		CellGrid<U>*     dem,
		bool*            d_blockDone,
		FlatDistType     flatDistCount)
{
	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 dims(dem->width, dem->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	handleFlatSurfacesKernel<<<grid, block>>>(
			flowDir->getData(),
			flatDist->getData(),
			dem->getData(),
			this->noDataValueDEM,
			flatDistCount,
			d_blockDone,
			dem->width,   dem->height);
	cudaGetLastError();
}

/**
 * This kernel calculates the derivative of the elevation towards each of the
 * neighbours, and the flow direction is set to the direction of the steepest
 * slope.
 *
 * If the cell is flat, i.e. none of the neighbours have lower
 * elevation, the cell is marked as flat in \a flatDistData, and handled later
 * by different algorithm.
 *
 * If the cell is on the stream or does not have elevation data, it is also
 * marked into the \a flatDistData.
 *
 * \param flowDirData The data array for the flow directions.
 * \param flatDistData The array indicating the type of the cells.
 * \param demData The pit-filled elevation data.
 * \param noDataValue The elevation value in the \a demData array which
 *        indicates that the cell does not have an elevation value.
 * \param demWidth The width of the DEM data.
 * \param demHeight The height of the DEM data.
 */
template<typename T, typename U, typename V>
__global__
void flowRoutingKernel(
		T* flowDirData,
		U* flatDistData,
		V* demData,
		V noDataValue,
		int demWidth, int demHeight)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

	if(x >= demWidth || y >= demHeight) {
		return;
	}

	int index = y * demWidth + x;

	int dx = 0;
	int dy = 0;

    // At the DEM border the flow direction is set to outsize the DEM.
	if     (x == 0)             dx = -1;
	else if(x == demWidth - 1)  dx =  1;
	if     (y == 0)             dy = -1;
	else if(y == demHeight - 1) dy =  1;

	if(dx != 0 || dy != 0) {
		flowDirData[index] = (T) {dx, dy};
		flatDistData[index] = FD_HAS_FLOW_DIR;
		return;
	}
	if(demData[index] == STREAM) {
		flatDistData[index] = FD_STREAM;
		return;
	}
	if(demData[index] == noDataValue) {
		flatDistData[index] = FD_NO_DATA;
		return;
	}

	V steepest = 0.0f;
	int steepestDx = 0;
	int steepestDy = 0;

    // Find out to which neighbour the derivative of the elevation is
    // steepest, and set the flow direction to that neighbour.
	for(int n = 0; n < 8; n++) {
		int nx, ny;

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

		if(nx < 0 || nx >= demWidth || ny < 0 || ny >= demHeight) {
			continue;
		}

		int nIndex = ny*demWidth + nx;
		if(demData[nIndex] == noDataValue) {
			continue;
		}

		V steepness = demData[index] - demData[nIndex];

		if(steepness > 0.0f) {
			if(n%2 == 1) {
				steepness = steepness/MATH_SQRTTWO;
			}
			if(steepness > steepest) {
				steepest = steepness;
				steepestDx = dx;
				steepestDy = dy;
			}
		}
	}

	if(steepest > 0.0f) {
		flowDirData[index]  = (T) {steepestDx, steepestDy};
		flatDistData[index] = FD_HAS_FLOW_DIR;
	}
}

/**
 * \brief FlowRoutingAlgorithm-specific functions
 */
namespace FlowRoutingOriginal {

    /**
     * \brief A CUDA implementation of a parallel reduction AND algorithm. If
     * all of the values in the \a data array are true, after the execution the
     * first element is true. If there is even one false in the array, the
     * first element is false.
     */
	__device__
	void sharedReduceAnd(
			const int &threadId,
			bool* data)
	{
		for(int s = (blockDim.x*blockDim.y)/2; s > 0; s >>= 1) {
			if(threadId < s) {
				data[threadId] &= data[threadId + s];
			}
			__syncthreads();
		}
	}
}

/**
 * This kernel only processes cells that do not have flow direction
 * determined. If the cell has neighbour whose
 * - elevation is not higher than the cell's own elevation and
 * - flow direction has been determined before, i.e. not during the same
 *   iteration in the algorithm,
 *
 * the flow direction of this cell is set towards the neighbour.
 * Nearest-neighbours are check first, then the second-nearest. The \a
 * flatDistCount is saved in the \a flatDistData to indicate the iteration the
 * cell was given a flow direction.
 *
 * \param flowDirData The flow direction array.
 * \param flatDistData An array containing information about the cells.
 * \param demData The pit-filled elevation data.
 * \param noDataValue The elevation value in the \a demData array which
 *                    indicates that the cell does not have an elevation
 *                    value.
 * \param flatDistCount A value set for the cell's flatDistData when the flow
 *                      direction is determined (usually the current iteration
 *                      number).
 * \param demWidth The width of the DEM data.
 * \param blockDoneArray The array used to keep track which blocks have
 *                       finished their work.
 * \param demHeight The height of the DEM data.
 */
template<typename T, typename U, typename V>
__global__
void handleFlatSurfacesKernel(
		T* flowDirData,
		U* flatDistData,
		V* demData,
		V noDataValue,
		U flatDistCount,
		bool* blockDoneArray,
		int demWidth, int demHeight)
{
	__shared__ bool threadDone[BLOCK_DIM];

    // The index of the thread in the block
	int tid = threadIdx.y*blockDim.x + threadIdx.x;

    // The index of the block in the thread block grid
	int bid = blockIdx.y*gridDim.x  + blockIdx.x;

    // If the threads in this block have already determined the flow
    // directions to all the cells in the previous iterations, nothing needs
    // to be done.
	if(blockDoneArray[bid]) {
		return;
	}

    // Assume the thread has already determined flow direction for the cell
    // and did not do anything in this iteration.
	threadDone[tid] = true;
    
    // The coordinates of the cell in the data grid.
	int x = blockDim.x*blockIdx.x + threadIdx.x;
	int y = blockDim.y*blockIdx.y + threadIdx.y;

	bool outside = (x >= demWidth || y >= demHeight);

	int index = y*demWidth + x;

	__syncthreads();

	// flatDistData == 0 means that the cell is flat and has not been
    // assigned a flow direction.
	if (!outside && flatDistData[index] == 0) {
		V    elev          = demData[index];
		bool flowDirsGiven = false;

		// find possible non-flat neighbour
		for (int i = 0; i < 8; i++) {
			int dx, dy;
			int nx, ny;

			switch(i) {
				case 0: dx =  0; dy = -1; break;
				case 1: dx = -1; dy =  0; break;
				case 2: dx =  1; dy =  0; break;
				case 3: dx =  0; dy =  1; break;
				case 4: dx = -1; dy = -1; break;
				case 5: dx =  1; dy = -1; break;
				case 6: dx = -1; dy =  1; break;
				case 7: dx =  1; dy =  1; break;
			}

			nx = x + dx;
			ny = y + dy;

			if(nx < 0 || nx >= demWidth || ny < 0 || ny >= demHeight) {
				continue;
			}

			int nIndex = ny*demWidth + nx;
            V nElev = demData[nIndex];
			U nFlatDist = flatDistData[nIndex];

            /*
             * The first condition is met when the neighbour cell is at the
             * edge of a flat region.
             *
             * The second condition is met when the neighbour cell is flat and
             * has been assigned flow direction on (some) previous iteration.
             * Ignoring the values from this iteration renders this algorithm
             * unaffected to the execution order of the thread blocks.
             *
             * When the suitable neighbour cell is found, set the flow
             * direction towards that neighbour and set this cell's
             * flatDistData to \a flatDistCount.
             */
			if ( (nFlatDist < 0 && (nElev != noDataValue && elev >= nElev)) ||
                 (nFlatDist >= 1 && nFlatDist + 1 <= flatDistCount) ) {
				flowDirData[index].x = dx;
				flowDirData[index].y = dy;
				flatDistData[index] = flatDistCount;
				flowDirsGiven = true;
				break;
			}
		}
		if (!flowDirsGiven) {
            // The cell that belongs to this thread still has no flow
            // direction set.
			threadDone[tid] = false;
		}

	}

	__syncthreads();

    // If even one thread has not assigned a flow direction, the thread block
    // is not done.
    FlowRoutingOriginal::sharedReduceAnd(tid, threadDone);

	if(tid == 0) {
		if(threadDone[0]) {
			blockDoneArray[bid] = true;
		}
	}
}
