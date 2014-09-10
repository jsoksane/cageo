/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file StreamBurningAlgorithmFactory.h
 */

#ifndef STREAMBURNINGALGORITHMFACTORY_H_
#define STREAMBURNINGALGORITHMFACTORY_H_

#include "AbstractFactory.h"
#include "StreamBurningAlgorithmCPUOpenMP.h"
#include "StreamBurningAlgorithmGPUCUDA.cuh"

/**
 * \brief Create and return a new StreamBurningAlgorithm instance.
 */
class StreamBurningAlgorithmFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	StreamBurningAlgorithmFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~StreamBurningAlgorithmFactory();

    /**
     * \brief Create and return a new StreamBurningAlgorithm instance
     * compatible with the default implementation.
     */
	template<typename T, typename U>
	static StreamBurningAlgorithm<T, U>* create();
};

template<typename T, typename U>
StreamBurningAlgorithm<T, U>* StreamBurningAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new StreamBurningAlgorithm_CPU_OpenMP<T, U>();

	case GPU:
	case GPU_CUDA:   return new   StreamBurningAlgorithm_GPU_CUDA<T, U>();
	default:         assert(false); break;
	}
}

#endif /* STREAMBURNINGALGORITHMFACTORY_H_ */
