/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowRoutingAlgorithmFactory.h
 */

#ifndef FLOWROUTINGALGORITHMFACTORY_H_
#define FLOWROUTINGALGORITHMFACTORY_H_

#include <assert.h>

#include "AbstractFactory.h"
#include "FlowRoutingAlgorithmCPUOpenMP.h"
#include "FlowRoutingAlgorithmGPUCUDAOriginal.cuh"

/**
 * \brief Create and return a new FlowRoutingAlgorithm instance.
 */
class FlowRoutingAlgorithmFactory : public AbstractFactory {
private:
    /**
     * \copydoc AbstractFactory::AbstractFactory()
     */
	FlowRoutingAlgorithmFactory();
    /**
     * \brief Create and return a new FlowRoutingAlgorithm instance compatible
     * with the implementation \a impl.
     */
	template<typename T, typename U>
	static FlowRoutingAlgorithm<T, U>* create(Type impl);
public:
    /**
     * \copydoc AbstractFactory::~AbstractFactory()
     */
	virtual ~FlowRoutingAlgorithmFactory();
    /**
     * \brief Create and return an instance of FlowRoutingAlgorithm compatible
     * with the default implementation.
     */
	template<typename T, typename U>
	static FlowRoutingAlgorithm<T, U>* create();

};

template<typename T, typename U>
FlowRoutingAlgorithm<T, U>* FlowRoutingAlgorithmFactory::create()
{
	return create<T, U>(defaultImpl);
}

template<typename T, typename U>
FlowRoutingAlgorithm<T, U>* FlowRoutingAlgorithmFactory::create(Type impl)
{
	switch(impl) {
	case CPU:
	case CPU_OpenMP: return new FlowRoutingAlgorithm_CPU_OpenMP<T, U>();

	case GPU:
	case GPU_CUDA: return new FlowRoutingAlgorithm_GPU_CUDA_Original<T, U>();
	default: assert(false); break;
	}
}

#endif /* FLOWROUTINGALGORITHMFACTORY_H_ */
