/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowTracingAlgorithmFactory.h
 */

#ifndef FLOWTRACINGALGORITHMFACTORY_H_
#define FLOWTRACINGALGORITHMFACTORY_H_

#include "AbstractFactory.h"
#include "FlowTracingAlgorithmCPUOpenMP.h"
#include "FlowTracingAlgorithmGPUCUDA.cuh"

/**
 * \brief Create and return a new FlowTracingAlgorithm instance.
 */
class FlowTracingAlgorithmFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	FlowTracingAlgorithmFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~FlowTracingAlgorithmFactory();

    /**
     * \brief Create and return a new FlowTracingAlgorithm instance compatible
     * with the default implementation.
     */
	template<typename T, typename U, typename V>
	static FlowTracingAlgorithm<T, U, V>* create();
};

template<typename T, typename U, typename V>
FlowTracingAlgorithm<T, U, V>* FlowTracingAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new FlowTracingAlgorithm_CPU_OpenMP<T, U, V>();

	case GPU:
	case GPU_CUDA:   return new   FlowTracingAlgorithm_GPU_CUDA<T, U, V>();
	default:         assert(false); break;
	}
}

#endif /* FLOWTRACINGALGORITHMFACTORY_H_ */
