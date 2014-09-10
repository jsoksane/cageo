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
 * \file PitFillingAlgorithmGPUCUDA.h
 */

#ifndef PITFILLINGALGORITHMGPUCUDA_H_
#define PITFILLINGALGORITHMGPUCUDA_H_

#include "PitFillingAlgorithm.h"

/**
 * \brief CUDA implementation of the PitFillingAlgorithm.
 */
template<typename T>
class PitFillingAlgorithm_GPU_CUDA : public PitFillingAlgorithm<T> {
public:
	/**
	 * \brief Construct the algorithm object.
	 */
	PitFillingAlgorithm_GPU_CUDA() {};
	virtual ~PitFillingAlgorithm_GPU_CUDA() {};

	/**
	 * \brief Fill the spill model.
	 */
	void performPitFilling(
				RasterDEM<T>* spill,
				RasterDEM<T>* dem);

private:
    /**
     * \brief Calculate the CUDA parameters and launch the appropriate number
     * of initializePitFillingDataKernel() CUDA kernels.
     */
	void initializePitFillingData_CUDA(
			RasterDEM<T>* spill,
			ProcLater_t* procLater,
			RasterDEM<T>* dem);

    /**
     * \brief Calculate the CUDA parameters, allocate necessary memory and
     * start the iterative process of filling the pits using
     * pitfillingKernel() CUDA kernels.
     */
	void pitFilling_CUDA(
			RasterDEM<T>* filledDem,
			ProcLater_t* procLater,
			RasterDEM<T>* dem);
};

/**
 * \brief A CUDA kernel to initialize the arrays needed in the pit filling
 * algorithm.
 */
template<typename T>
__global__
void initializePitFillingDataKernel(
		T*    filledDem,
		bool* processLaterData,
		T*    demData,
		T     noDataValue,
		int demWidth,  int demHeight);

/**
 * \brief A CUDA kernel to perform one iteration in the pit filling algorithm.
 */
template<typename T>
__global__
void pitfillingKernel(
		T*    filledDem,
		bool* processLaterData,
		T*    demData,
		bool* needsMoreProcessing,
		T   noDataValue,
		int offsetX, int offsetY,
		int demWidth, int demHeight);

#endif /* PITFILLINGALGORITHMGPUCUDA_H_ */
