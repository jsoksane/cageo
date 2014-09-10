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
 * \file PitFillingAlgorithmGPUCUDA.cu
 */

#include "PitFillingAlgorithmGPUCUDA.cuh"
#include "PitFillingCommon.h"
#include "Utils.h"

#include "Reduce.cuh"
#include "ReduceUtils.h"

#include <iostream>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define BLOCK_DIM        (BLOCK_SIZE_X*BLOCK_SIZE_Y)
#define BLOCK_DIM_REDUCE BLOCK_DIM/2

/**
 * Copy all the needed data to the device, and perform the filling.
 */
template<typename T>
void PitFillingAlgorithm_GPU_CUDA<T>::performPitFilling(
		RasterDEM<T>* elevatedDem,
		RasterDEM<T>* dem)
{
	elevatedDem->toDevice();
	dem->toDevice();

    ProcLater_t* procLater = new ProcLater_t(*dem, AS_TEMPLATE);

	if(this->iter == 0) {
		initializePitFillingData_CUDA(elevatedDem, procLater, dem);
	}

	pitFilling_CUDA(elevatedDem, procLater, dem);

    delete procLater;
}

template void
PitFillingAlgorithm_GPU_CUDA<DemData>::performPitFilling(RasterDEM<DemData>*, RasterDEM<DemData>*);

template<typename T>
void PitFillingAlgorithm_GPU_CUDA<T>::initializePitFillingData_CUDA(
		RasterDEM<T>* elevatedDem,
		ProcLater_t* procLater,
		RasterDEM<T>* dem)
{
	dim3 block(32, 32, 1);
	dim3 dims(elevatedDem->width, elevatedDem->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	initializePitFillingDataKernel<<<grid, block>>>(
			elevatedDem->getData(),
			procLater->getData(),
			dem->getData(),
			this->noDataValueDEM,
			dem->width, dem->height);
	cudaGetLastError();
}

/**
 * The data is divided into BLOCK_SIZE_X * BLOCK_SIZE_Y sub-areas titled
 * 1, 2, 3 and 4 according to the description below
 *
 *     +---+---+---+---+--
 *     | 1 | 2 | 1 | 2 |
 *     +---+---+---+---+--
 *     | 3 | 4 | 3 | 4 |
 *     +---+---+---+---+--
 *     | 1 | 2 | 1 | 2 |
 *     +---+---+---+---+--
 *     | 3 | 4 | 3 | 4 |
 *     +---+---+---+---+--
 *     |   |   |   |   |
 *
 * Only the sub-areas with same title can be processed in parallel (due to
 * possible race conditions at the edges of the sub-areas).
 *
 * The sub-areas are processed in turns, and after each iteration it is
 * checked if there were any modifications to the cells. When all the
 * sub-areas have been idle in the same iteration, the algorithm is finished.
 */
template<typename T>
void PitFillingAlgorithm_GPU_CUDA<T>::pitFilling_CUDA(
		RasterDEM<T>* elevatedDem,
		ProcLater_t* procLater,
		RasterDEM<T>* dem) {

	dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
	dim3 dims(elevatedDem->width, elevatedDem->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	int blockIdleSize        = grid.x * grid.y;
	int blockIdleSizeReduced = calculate_reduced_size(blockIdleSize,
                                                      BLOCK_DIM_REDUCE);

    // If we launch blockIdleSize many kernels, 3/4 would do nothing, so let's
    // calculate a more appropriate amount.
    dim3 gridSim;
    dim3 dimsSim(elevatedDem->width / 2, elevatedDem->height / 2, 1);
    CudaUtils::calcGridSize(block, dimsSim, &gridSim);
    assert (gridSim.x * 2 * block.x >= elevatedDem->width);
    assert (gridSim.y * 2 * block.y >= elevatedDem->height);

    // One boolean for each thread block. Used to keep track which blocks have
    // been idling during the iterations.
	bool* d_blockIdle;
    // Smaller boolean array, used as the output of CUDA-parallelized
    // reduceAnd algorithm.
	bool* d_blockIdleReduced;

	CUDA( cudaMalloc((void**)&d_blockIdle,        blockIdleSize       *sizeof(bool)) );
	CUDA( cudaMalloc((void**)&d_blockIdleReduced, blockIdleSizeReduced*sizeof(bool)) );

	CUDA( cudaMemset(d_blockIdle,        0, blockIdleSize       *sizeof(bool)) );
	CUDA( cudaMemset(d_blockIdleReduced, 0, blockIdleSizeReduced*sizeof(bool)) );

	bool* h_blockIdleReduced = new bool[blockIdleSizeReduced];

	for (int iter = 0;; iter++) {
        // Assume each block is doing nothing during this iteration.
		CUDA( cudaMemset(d_blockIdle, 1, blockIdleSize*sizeof(bool)) );

        // In turns process all of the sub-areas
        for (int offsetY = 0; offsetY < 2; offsetY++) {
            for (int offsetX = 0; offsetX < 2; offsetX++) {
			pitfillingKernel<<<gridSim, block>>>(
					elevatedDem->getData(),
					procLater->getData(),
					dem->getData(),
					d_blockIdle,
					this->noDataValueDEM,
					offsetX, offsetY,
					dem->width, dem->height);
            }
		}

        // Reduce the values from d_blockIdle to the smaller d_blockIdleReduced.
		reduceAnd<BLOCK_DIM_REDUCE><<<blockIdleSizeReduced, BLOCK_DIM_REDUCE, BLOCK_DIM_REDUCE*sizeof(bool)>>>
					   (d_blockIdle, d_blockIdleReduced, blockIdleSize);

        // Do the final reduction on the host
		CUDA( cudaMemcpy(h_blockIdleReduced, d_blockIdleReduced, blockIdleSizeReduced*sizeof(bool), cudaMemcpyDeviceToHost) );

		bool allBlocksIdle = true;
		for (int i = 0; i < blockIdleSizeReduced; i++) {
			if (!h_blockIdleReduced[i]) {
				allBlocksIdle = false;
				break;
			}
		}
		if (allBlocksIdle) {
			break;
		}
	}

	CUDA( cudaFree(d_blockIdle) );
	CUDA( cudaFree(d_blockIdleReduced) );

	delete[] h_blockIdleReduced;
}

/**
 * Set the \a filledDem to a high value (effectively to infinity) in all the
 * cells except the ones that
 * - are located on the streams or on the border of the global DEM data or
 * - do not have elevation data.
 * In those cases the value from the original DEM used.
 *
 * Set processLater value to true for all the cell at the border of the DEM
 * and on the streams.
 *
 * \param filledDem The array to hold the DEM data where the pits have been
 *                  filled.
 * \param processLaterData The array of boolean values.
 * \param demData The array that hold the original DEM values.
 * \param noDataValue The value used to mark cells that do not have meaningful
 *                    elevation data (e.g. cells on streams or at the border
 *                    of DEM).
 * \param demWidth The width of the global DEM data.
 * \param demHeight The height of the global DEM data.
 */
template<typename T>
__global__
void initializePitFillingDataKernel(
		T*    filledDem,
		bool* processLaterData,
		T*    demData,
		T     noDataValue,
		int demWidth,  int demHeight) {
    int gx = blockIdx.x*blockDim.x + threadIdx.x;
    int gy = blockIdx.y*blockDim.y + threadIdx.y;

    if (gx >= demWidth || gy >= demHeight) {
    	return;
    }

	int index = gy * demWidth + gx;

	bool processLater  = false;
	T    newElevation         = 99990000.0f;
	T    origElevation = demData[index];

	if (origElevation == noDataValue) {
		newElevation = origElevation;
	} else if (origElevation == STREAM ||
		       gx == 0 || gx == demWidth - 1 || gy == 0 || gy == demHeight - 1) {
		newElevation = origElevation;
		processLater = true;
	}

	filledDem[index] = newElevation;
	processLaterData[index] = processLater;
}

/**
 * \brief Functions specific to the PitFillingAlgorithm.
 */
namespace PitFilling {
    /**
     * GPU CUDA parallel reduce algorithm. If there is even one true value in
     * the given \a data array, after the reduction the first item in the array
     * is true.
     */
	__device__
	void sharedReduceOr(
			const int &threadId,
			bool* data)
	{
		for(int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
			if(threadId < s) {
				data[threadId] |= data[threadId + s];
			}
			__syncthreads();
		}
	}
}

/**
 * Fill the pits in the sub-area defined by the dimension and the index of the
 * thread block and \a offsetX and \a offsetY parameters.
 *
 * \param filledDem An array of DemData values representing the DEM data with
 *                  filled pits.
 * \param procLaterData An array of boolean values to keep track of which cells
 *                      should be processed in the next iteration.
 * \param demData The original DEM data with burned stream information.
 * \param blockIdleData The array used to keep track which blocks are idle
 *                      during the execution.
 * \param noDataValue The value used to mark cells that do not have meaningful
 *                    elevation data.
 * \param offsetX A parameter used to locate the data for this thread block.
 * \param offsetY A parameter used to locate the data for this thread block.
 * \param width The width of the data.
 * \param height The height of the data.
 */
template<typename T>
__global__
void pitfillingKernel(
		T*    filledDem,
		bool* procLaterData,
		T*    demData,
		bool* blockIdleData,
		T   noDataValue,
		int offsetX, int offsetY,
		int width, int height)
{
	__shared__ bool blockNeedsProcessing[BLOCK_DIM];

    // The thread index in the block
	int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int blockX = 2 * blockIdx.x + offsetX;
    int blockY = 2 * blockIdx.y + offsetY;
    // The index of the block in the thread block grid
	int bid = blockY * gridDim.x + blockX;

	blockNeedsProcessing[tid] = false;

	int x = blockX * blockDim.x + threadIdx.x;
	int y = blockY * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}

	int index = y * width + x;

    // First check if any thread in this block has work to do
	blockNeedsProcessing[tid] = procLaterData[index];

	__syncthreads();

	PitFilling::sharedReduceOr(tid, blockNeedsProcessing);

	if (blockNeedsProcessing[0]) {
		if (tid == 0) {
			blockIdleData[bid] = false;
		}

		while(blockNeedsProcessing[0]) {
			bool blockIdle = true;

			T cElevated = filledDem[index]; // Save locally to prevent other threads from modifying it

			bool skip = true;
			if (procLaterData[index]) {
				procLaterData[index] = false;
				skip = false;
			}

			for (int i = 0; i < 8; i++) {
				int dx, dy;
				int ndx, ndy;

				__syncthreads();

				if(skip) {
					continue;
				}

				switch(i) {
					case 0: dx = -1; dy = -1; break;
					case 1: dx =  0; dy = -1; break;
					case 2: dx =  1; dy = -1; break;
					case 3: dx = -1; dy =  0; break;
					case 4: dx =  1; dy =  0; break;
					case 5: dx = -1; dy =  1; break;
					case 6: dx =  0; dy =  1; break;
					case 7: dx =  1; dy =  1; break;
				}

				ndx = x + dx;
				ndy = y + dy;

				if (ndx <= 0 || ndx >= width - 1 || ndy <= 0 || ndy >= height - 1) {
					continue;
				}

				int nIndex = ndy * width + ndx;
				T   nElevated = filledDem[nIndex];

                // If neighbour has newElevation value and it is higher than
                // this cell's newElevation value it is not the original
                // elevation value, modify the neighbour's newElevation value
                // and set it's prosessLater value to true.
				if (nElevated != noDataValue && cElevated < nElevated &&
                    nElevated != demData[nIndex]) {
					filledDem[nIndex]     = max(cElevated, demData[nIndex]);
					procLaterData[nIndex] = true;
					blockIdle             = false;
				}
			}

            // If the thread modified any neighbour, there may be more work to
            // do
			blockNeedsProcessing[tid] = !blockIdle;

			__syncthreads();

			PitFilling::sharedReduceOr(tid, blockNeedsProcessing);
		}
	}
}

