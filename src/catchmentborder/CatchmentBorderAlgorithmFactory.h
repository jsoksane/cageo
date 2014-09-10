/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file CatchmentBorderAlgorithmFactory.h
 */

#ifndef CATCHMENTBORDERALGORITHMFACTORY_H_
#define CATCHMENTBORDERALGORITHMFACTORY_H_

#include "AbstractFactory.h"
#include "CatchmentBorderAlgorithmCPUOpenMP.h"
#include "CatchmentBorderAlgorithmGPUCUDA.cuh"

/**
 * \brief Create and return a new CatchmentBorderAlgorithm instance.
 */
class CatchmentBorderAlgorithmFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	CatchmentBorderAlgorithmFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~CatchmentBorderAlgorithmFactory();

    /**
     * \brief Create and return a new CatchmentBorderAlgorithm instance
     * compatible with the default implementation.
     */
	template<typename T, typename U>
	static CatchmentBorderAlgorithm<T, U>* create();
};

template<typename T, typename U>
CatchmentBorderAlgorithm<T, U>* CatchmentBorderAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new CatchmentBorderAlgorithm_CPU_OpenMP<T, U>();

	case GPU:
	case GPU_CUDA:   return new   CatchmentBorderAlgorithm_GPU_CUDA<T, U>();
	default:         assert(false); break;
	}
}

#endif /* CATCHMENTBORDERALGORITHMFACTORY_H_ */
