/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file PitFillingAlgorithmFactory.h
 */

#ifndef PITFILLINGALGORITHMFACTORY_H_
#define PITFILLINGALGORITHMFACTORY_H_

#include "AbstractFactory.h"
#include "PitFillingAlgorithmCPUOpenMP.h"
#include "PitFillingAlgorithmGPUCUDA.cuh"

/**
 * \brief Create and return a new PitFillingAlgorithm instance.
 */
class PitFillingAlgorithmFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	PitFillingAlgorithmFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~PitFillingAlgorithmFactory();

    /**
     * \brief Create and return a new PitFillingAlgorithm instance compatible
     * with the default implementation.
     */
	template<typename T>
	static PitFillingAlgorithm<T>* create();
};

template<typename T>
PitFillingAlgorithm<T>* PitFillingAlgorithmFactory::create()
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new PitFillingAlgorithm_CPU_OpenMP<T>();

	case GPU:
	case GPU_CUDA:   return new   PitFillingAlgorithm_GPU_CUDA<T>();
	default:         assert(false); break;
	}
}

#endif /* PITFILLINGALGORITHMFACTORY_H_ */
