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
 * \file FlowTracingAlgorithmGPUCUDA.h
 */

#ifndef FLOWTRACINGALGORITHMGPUCUDA_H_
#define FLOWTRACINGALGORITHMGPUCUDA_H_

#include "FlowTracingAlgorithm.h"

/**
 * \brief A CUDA implementation of the FlowTracingAlgorithm.
 */
template<typename T, typename U, typename V>
class FlowTracingAlgorithm_GPU_CUDA : public FlowTracingAlgorithm<T, U, V> {
public:
	FlowTracingAlgorithm_GPU_CUDA() {};
	virtual ~FlowTracingAlgorithm_GPU_CUDA() {};

    /**
     * \brief Trace the flow paths inside the thread blocks.
     */
	void performFlowTracing(
			FlowTraceDataClass<T>* flowTrace,
			FlowDirClass<U>*       flowDir,
			StreamData<V>*         streams);

    /**
     * \brief Connect the flow paths determined previously by independent
     * thread blocks.
     */
	void performConnectFlowPath(
			FlowTraceDataClass<T>* flowTrace);

private:
    /**
     * \brief Calculate the CUDA parameters and launch the appropriate number
     * of flowTraceKernel() CUDA kernels.
     */
	void flowTrace_CUDA(
			FlowTraceDataClass<T>* flowTrace,
			FlowDirClass<U>*       flowDir,
			StreamData<V>*         streams);

    /**
     * \brief Calculate the CUDA parameters and launch the appropriate number
     * of connectFlowPathKernel() CUDA kernels.
     */
	void connectFlowPath_CUDA(
			FlowTraceDataClass<T>* flowTrace);
};

/**
 * \brief A CUDA kernel to trace the flow paths of the cells inside the thread
 * block.
 */
template<typename T, typename U, typename V>
__global__
void flowTraceKernel(
		U* flowTraceData,
		T* flowDirData,
		V* streamData,
		int demWidth, int demHeight);

/**
 * \brief A CUDA kernel to connect the traced flow paths determined previously
 * inside thread blocks.
 */
template<typename T>
__global__
void connectFlowPathKernel(
		T* flowTraceData,
		int width, int height);


#endif /* FLOWTRACINGALGORITHMGPUCUDA_H_ */
