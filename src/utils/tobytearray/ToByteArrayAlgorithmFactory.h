/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ToByteArrayAlgorithmFactory.h
 */

#ifndef TOBYTEARRAYALGORITHMFACTORY_H_
#define TOBYTEARRAYALGORITHMFACTORY_H_

#include <assert.h>

#include "AbstractFactory.h"
#include "ToByteArrayAlgorithmGPUCUDA.cuh"
#include "ToByteArrayAlgorithmCPUOpenMP.h"

class ToByteArrayAlgorithmFactory : AbstractFactory {
public:
	ToByteArrayAlgorithmFactory() {};
	virtual ~ToByteArrayAlgorithmFactory() {};

	template<typename T>
	static ToByteArrayAlgorithm<T>* create();
};

template<typename T>
ToByteArrayAlgorithm<T>* ToByteArrayAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new ToByteArrayAlgorithm_CPU_OpenMP<T>();

	case GPU:
	case GPU_CUDA:   return new ToByteArrayAlgorithm_GPU_CUDA<T>();
	default:         assert(false); break;
	}
}

#endif /* TOBYTEARRAYALGORITHMFACTORY_H_ */
