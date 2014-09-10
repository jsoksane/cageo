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
 * \file CatchmentBorderAlgorithmGPUCUDA.h
 */

#ifndef CATCHMENTBORDERALGORITHMGPUCUDA_H_
#define CATCHMENTBORDERALGORITHMGPUCUDA_H_

#include "CatchmentBorderAlgorithm.h"

/**
 * \brief The CUDA implementation of the CatchmentBorderAlgorithm.
 */
template<typename T, typename U>
class CatchmentBorderAlgorithm_GPU_CUDA : public CatchmentBorderAlgorithm<T, U> {
public:
    /**
     * \brief The constructor.
     */
	CatchmentBorderAlgorithm_GPU_CUDA() {};
    /**
     * \brief The destructor.
     */
	virtual ~CatchmentBorderAlgorithm_GPU_CUDA() {};

    /**
     * \brief Transfer the data to DEVICE and execute the
     * extractDrainageBorder_CUDA().
     */
	void performExtractDrainageBorder(
				CatchmentBorderClass<T>* catchmentBorder,
				FlowTraceDataClass<U>* flowTrace);

protected:
    /**
     * \brief Calculate the CUDA kernel parameters and launch appropriate
     * number of extractDrainageBorderKernel() CUDA kernels.
     */
	void extractDrainageBorder_CUDA(
				CatchmentBorderClass<T>* catchmentBorder,
				FlowTraceDataClass<U>* flowTrace);
};

/**
 * \brief The CUDA kernel to determine the drainage borders from the given
 * flowTraceData.
 */
template<typename T, typename U>
__global__
void extractDrainageBorderKernel(
		T* catchmentBorderData,
		U* flowTraceData,
		int width, int height);

#endif /* CATCHMENTBORDERALGORITHMGPUCUDA_H_ */
