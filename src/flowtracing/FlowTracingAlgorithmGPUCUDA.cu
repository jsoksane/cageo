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
 * \file FlowTracingAlgorithmGPUCUDA.cu
 */

#include "FlowTracingAlgorithmGPUCUDA.cuh"
#include "KernelUtilsFactory.h"
#include "FlowTracingCommon.h"
#include "FlowTraceDataClass.h"
#include "FlowDirClass.h"
#include "DemData.h"

template<typename T, typename U, typename V>
void FlowTracingAlgorithm_GPU_CUDA<T, U, V>::performFlowTracing(
			FlowTraceDataClass<T>* flowTrace,
			FlowDirClass<U>*       flowDir,
			StreamData<V>*         streams)
{
	flowTrace->toDevice();
	flowDir->toDevice();
	streams->toDevice();

	InitializeArrayAlgorithm_GPU_CUDA<T> initializeArrayAlgorithm;

	initializeArrayAlgorithm.setInitializeValue(FLOWTRACE_DEFAULT);
	initializeArrayAlgorithm.execute(flowTrace);

	flowTrace_CUDA(flowTrace, flowDir, streams);
}

template void FlowTracingAlgorithm_GPU_CUDA<FlowTraceDataType, FlowDirDataType, DemData>::
	performFlowTracing(FlowTraceDataClass<FlowTraceDataType>*, FlowDirClass<FlowDirDataType>*, StreamData<DemData>*);

template<typename T, typename U, typename V>
void FlowTracingAlgorithm_GPU_CUDA<T, U, V>::performConnectFlowPath(
			FlowTraceDataClass<T>* flowTrace)
{
	flowTrace->toDevice();

	connectFlowPath_CUDA(flowTrace);
}

template void FlowTracingAlgorithm_GPU_CUDA<FlowTraceDataType, FlowDirDataType, DemData>::performConnectFlowPath(FlowTraceDataClass<FlowTraceDataType>*);

template<typename T, typename U, typename V>
void FlowTracingAlgorithm_GPU_CUDA<T, U, V>::flowTrace_CUDA(
		FlowTraceDataClass<T>* flowTrace,
		FlowDirClass<U>*       flowDir,
		StreamData<V>*         streams)
{
	dim3 block(16, 16, 1);
	dim3 dims(flowTrace->width, flowTrace->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	flowTraceKernel<<<grid, block>>>(
			flowTrace->getData(),
			flowDir->getData(),
			streams->getData(),
			flowTrace->width,   flowTrace->height);
}

template<typename T, typename U, typename V>
void FlowTracingAlgorithm_GPU_CUDA<T, U, V>::connectFlowPath_CUDA(
		FlowTraceDataClass<T>* flowTrace)
{
	dim3 block(16, 16, 1);
	dim3 dims(flowTrace->width, flowTrace->height, 1);
	dim3 grid;

	CudaUtils::calcGridSize(block, dims, &grid);

	connectFlowPathKernel<<<grid, block>>>(
			flowTrace->getData(),
			flowTrace->width,   flowTrace->height);
}

/**
 * The \a flowDirData is logically divided into areas defined by the thread
 * block size, and each cell belongs to a separate thread. The flow path of
 * each cell is then followed, until
 * - the path leads to a stream, in which case the negative of the streams id
 *   value is stored in the \a flowTraceData.
 * - the path leads outside of the area belonging to this thread block, in
 *   which case
 *   - if the cell is stream, again store the negative of the stream id into
 *     the  \a flowTraceData,
 *   - otherwise store the array index of the cell into the \a flowTraceData,
 *     and it will be handled later by the connectFlowPath algorithm.
 */
template<typename T, typename U, typename V>
__global__
void flowTraceKernel(
		U* flowTraceData,
		T* flowDirData,
		V* streamData,
		int demWidth, int demHeight)
{
    // The x-index of the cell in the top left corner of the block.
	int blockX = blockIdx.x*blockDim.x;
    // The y-index of the cell in the top left corner of the block.
	int blockY = blockIdx.y*blockDim.y;

    // The x-index of the cell this thread operates on
    int x = blockX + threadIdx.x;
    // The y-index of the cell this thread operates on
    int y = blockY + threadIdx.y;

    if(x >= demWidth || y >= demHeight) {
    	return;
    }

	const int LEFT_BORDER = blockX - 1;
	const int TOP_BORDER  = blockY - 1;
	const int RIGHT_BORDER  = min(blockX + (int) blockDim.x, demWidth);
	const int BOTTOM_BORDER = min(blockY + (int) blockDim.y, demHeight);

    // The array index of the cell this thread operates on
	int index = y*demWidth + x;
	int curIndex = index;

	// Are we at a stream/lake?
	if (streamData[index] >= 0.0) {
		flowTraceData[index] = FLOWTRACE_LAKE;
		return;
	}

	if (flowDirData[index].x == 0 && flowDirData[index].y == 0) {
		flowTraceData[index] = FLOWTRACE_ERROR;
		printf("Error: found cell without flow directions when tracing flow. Exiting...\n");
		return;
	}

	int iter = 0;
    /*
     * Start to follow the path defined by the flow direction of the cells
     * until we hit stream, DEM border or thread block border.
     */
	while (true) {
		int dx = (int) flowDirData[curIndex].x;
		int dy = (int) flowDirData[curIndex].y;

		x += dx;
		y += dy;
		curIndex = y * demWidth + x;

		if (iter > 1000) {
			printf("Infinite loop when tracing flow directions!\n");
			flowTraceData[curIndex] = FLOWTRACE_ERROR;
			return;
		}

		if (dx == 0 && dy == 0) {
			printf("Error: found cell without flow directions when tracing flow. Exiting...\n");
			return;
		}

		/*
         * Check if we have wondered off from the area that belongs to
         * this thread block.
         */
        bool onTheBorder = (x == LEFT_BORDER || x == RIGHT_BORDER ||
                            y == TOP_BORDER || y == BOTTOM_BORDER);

		if (onTheBorder) {
			if (x < 0 || x >= demWidth || y < 0 || y >= demHeight) {
                // We are outside of the DEM
				flowTraceData[index] = FLOWTRACE_OUTSIDE;
			} else {
                /*
                 * We are inside of the DEM but in wrong block.
                 *
                 * If the cell in the adjacent block is a stream or a lake,
                 * set it's id number (multiplied by -1) to the original cell
                 * in the flowTraceData.
                 *
                 * Otherwise, save that cell's array index to the original
                 * cell in the flowTraceData, and it will be handled by the
                 * connectFlowPathKernel() later.
                 */
				if(streamData[curIndex] >= 0.0f) {
					flowTraceData[index] = -streamData[curIndex];
				} else {
					flowTraceData[index] = curIndex;
				}
			}
			break;
		} else {
            /*
             * All good, we are still inside the area that belongs to our
             * block. If we found a stream/lake, save the negative of that
             * stream's id to the original cell in the flowTraceData.
             */
            if(streamData[curIndex] >= 0.0) {
			    flowTraceData[index] = -streamData[curIndex];
			    break;
            }
		}
		iter++;
	}
}

/**
 * This kernel only operates on cells that are 0 or larger. Positive number
 * means that the cell's flow path was previously followed to lead to a cell
 * whose array index this number is.
 *
 * This algorithm starts from a cell i with value p >= 0 and follows the path
 * by substituting
 *
 *     p = flowTraceData[p]
 *
 * until p < 0, which means the path finally ended into a stream or to the
 * edge of the elevation data.
 */
template<typename T>
__global__
void connectFlowPathKernel(
		T* flowTraceData,
		int width, int height)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if(x >= width || y >= height) {
        return;
    }

        int origIndex = y*width + x;
        int newIndex = origIndex;

        // Only concentrate on cells that have positive flowTraceData
        if(flowTraceData[origIndex] < 0) {
                return;
        }

        int iter = 0;
        while (true) {
                /*
                 * Positive value means that it is "pointing" (it is an array
                 * index) to another cell in the flowTraceData array.
                 */
                newIndex = (int) flowTraceData[newIndex];

                if (iter > 100) {
                        printf("Infinite loop?\n");
                        return;
                }

                if (newIndex >= 0) {
                    // Found yet another pointer
                    continue;
                } else if (newIndex == FLOWTRACE_DEFAULT) {
                    break;
                } else if (newIndex < 0) {
                    // Finally found a cell that contains an id value of some
                    // stream.
                    flowTraceData[origIndex] = (T) newIndex;
                    break;
                }

                iter++;
        }

}
