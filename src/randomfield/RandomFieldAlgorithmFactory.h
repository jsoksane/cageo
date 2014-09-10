/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RandomFieldAlgorithmFactory.h
 */

#ifndef RANDOMFIELDALGORITHMFACTORY_H_
#define RANDOMFIELDALGORITHMFACTORY_H_

#include "AbstractFactory.h"
#include "RandomFieldAlgorithm.h"
#include "RandomFieldAlgorithmCPUOpenMP.h"
#include "RandomFieldAlgorithmGPUCUDA.cuh"

/**
 * \brief Create and return a new RandomFieldAlgorithm instance.
 */
class RandomFieldAlgorithmFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	RandomFieldAlgorithmFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~RandomFieldAlgorithmFactory() {};

    /**
     * \brief Create and return a new RandomFieldAlgorithm instance compatible
     * with the default implementation.
     */
	template<typename T>
	static RandomFieldAlgorithm<T>* create();
};

template<typename T>
RandomFieldAlgorithm<T>* RandomFieldAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new RandomFieldAlgorithm_CPU_OpenMP<T>();

	case GPU:
	case GPU_CUDA:   return new   RandomFieldAlgorithm_GPU_CUDA<T>();
	default:         assert(false); break;
	}
}

#endif /* RANDOMFIELDALGORITHMFACTORY_H_ */
