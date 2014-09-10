/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ConvolutionAlgorithmFactory.h
 */

#ifndef CONVOLUTIONALGORITHMFACTORY_H_
#define CONVOLUTIONALGORITHMFACTORY_H_

#include <assert.h>
#include "AbstractFactory.h"
#include "ConvolutionAlgorithmCPUOpenMP.h"
#include "ConvolutionAlgorithmGPUCUDA.cuh"

/**
 * \brief Create and return a new ConvolutionAlgorithm instance.
 */
class ConvolutionAlgorithmFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	ConvolutionAlgorithmFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~ConvolutionAlgorithmFactory();

    /**
     * \brief Create and return a new ConvolutionAlgorithm instance compatible
     * with the default implementation.
     */
	template<typename T>
	static ConvolutionAlgorithm<T>* create();
};

template<typename T>
ConvolutionAlgorithm<T>* ConvolutionAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new ConvolutionAlgorithm_CPU_OpenMP<T>();

	case GPU:
	case GPU_CUDA:   return new   ConvolutionAlgorithm_GPU_CUDA<T>();
	default:         assert(false); break;
	}
}

#endif /* CONVOLUTIONALGORITHMFACTORY_H_ */
