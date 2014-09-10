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
 * \file ConvertToByteArrayGPUCUDA.h
 */

#ifndef TOBYTEARRAYALGORITHMGPUCUDA_H_
#define TOBYTEARRAYALGORITHMGPUCUDA_H_

#include "ToByteArrayAlgorithm.h"

template<typename T>
class ToByteArrayAlgorithm_GPU_CUDA : public ToByteArrayAlgorithm<T> {
public:
	ToByteArrayAlgorithm_GPU_CUDA() {};
	virtual ~ToByteArrayAlgorithm_GPU_CUDA() {};

	void convertToByteArray(
			CellGrid<Byte_t>* output,
			CellGrid<T>*      input);

private:
	void convertToByteArray_CUDA(
			CellGrid<Byte_t>* output,
			CellGrid<T>*      input,
			T min,
			T max);
};

template<typename T, typename U>
__global__
void convertToByteArrayKernel(
		T* output,
		U* input,
		U min,
		U max,
		int dataWidth,
		int dataHeight);

#endif /* TOBYTEARRAYALGORITHMGPUCUDA_H_ */
