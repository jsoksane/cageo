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
 * \file FlowRoutingAlgorithmGPUCUDAOriginal.cuh
 */

#ifndef FLOWROUTINGALGORITHMGPUCUDAORIGINAL_H_
#define FLOWROUTINGALGORITHMGPUCUDAORIGINAL_H_

#include "FlowRoutingAlgorithm.h"
#include "FlowDirClass.h"
#include "FlatDistClass.h"
#include "CellGrid.h"

/**
 * \brief A CUDA implementation of the FlowRoutingAlgorithm.
 */
template<typename T, typename U>
class FlowRoutingAlgorithm_GPU_CUDA_Original : public FlowRoutingAlgorithm<T, U> {
public:
	FlowRoutingAlgorithm_GPU_CUDA_Original() {};
	virtual ~FlowRoutingAlgorithm_GPU_CUDA_Original() {};

	void performFlowRouting(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem);

	void performFlowRoutingOnFlatSurfaces(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem);

private:
	void flowRouting_CUDA(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem);

	void handleFlatSurfaces_CUDA(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem,
			bool*            d_blockDone,
			FlatDistType     flatDistCount);

};

/**
 * \brief The CUDA kernel to determine the flow directions of the non-flat
 * cells.
 */
template<typename T, typename U, typename V>
__global__
void flowRoutingKernel(
		T* flowDirData,
		U* flatDistData,
		V* demData,
		V noDataValue,
		int demWidth, int demHeight);

/**
 * \brief The CUDA kernel to perform on iteration in algorithm to determine
 * the flow directions of the flat cells.
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
		int demWidth, int demHeight);

#endif /* FLOWROUTINGALGORITHMGPUCUDAORIGINAL_H_ */
