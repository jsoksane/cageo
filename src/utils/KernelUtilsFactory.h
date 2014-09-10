/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file KernelUtilsFactory.h
 */

#ifndef KERNELUTILSFACTORY_H_
#define KERNELUTILSFACTORY_H_

#include <assert.h>

#include "AbstractFactory.h"

#include "RemoveNegativeElevationsAlgorithmCPUOpenMP.h"
#include "RemoveNegativeElevationsAlgorithmGPUCUDA.cuh"

#include "AddSurfacesAlgorithmCPUOpenMP.h"
#include "AddSurfacesAlgorithmGPUCUDA.cuh"

#include "InitializeArrayAlgorithmCPUOpenMP.h"
#include "InitializeArrayAlgorithmGPUCUDA.cuh"

#include "MultiplySurfacesAlgorithmCPUOpenMP.h"
#include "MultiplySurfacesAlgorithmGPUCUDA.cuh"

/**
 * \brief A class that creates and returns utility algorithm instances for the
 * current framework.
 */
class KernelUtilsFactory : public AbstractFactory {
private:
    /** \copydoc AbstractFactory::AbstractFactory() */
	KernelUtilsFactory();
public:
    /** \copydoc AbstractFactory::~AbstractFactory() */
	virtual ~KernelUtilsFactory() {};
    /**
     * \brief Return a new RemoveNegativeElevationsAlgorithm for the default
     * framework.
     */
	template<typename T>
	static RemoveNegativeElevationsAlgorithm<T>* createRemoveNegativeElevationsAlgorithm();
    /**
     * \brief Return a new RemoveNegativeElevationsAlgorithm for the given
     * framework.
     */
	template<typename T>
	static RemoveNegativeElevationsAlgorithm<T>* createRemoveNegativeElevationsAlgorithm(Type impl);

    /**
     * \brief Return a new AddSurfacesAlgorithm for the default framework.
     */
	template<typename T, typename U>
	static AddSurfacesAlgorithm<T, U>*           createAddSurfacesAlgorithm();
    /**
     * \brief Return a new AddSurfacesAlgorithm for the given framework.
     */
	template<typename T, typename U>
	static AddSurfacesAlgorithm<T, U>*           createAddSurfacesAlgorithm(Type impl);

    /**
     * \brief Return a new InitializeArrayAlgorithm for the default framework.
     */
	template<typename T>
	static InitializeArrayAlgorithm<T>*          createInitializeArrayAlgorithm();
    /**
     * \brief Return a new InitializeArrayAlgorithm for the given framework.
     */
	template<typename T>
	static InitializeArrayAlgorithm<T>*          createInitializeArrayAlgorithm(Type impl);

    /**
     * \brief Return a new MultiplySurfacesAlgorithm for the default
     * framework.
     */
	template<typename T, typename U>
	static MultiplySurfacesAlgorithm<T, U>*      createMultiplySurfacesAlgorithm();
    /**
     * \brief Return a new MultiplySurfacesAlgorithm for the given framework.
     */
	template<typename T, typename U>
	static MultiplySurfacesAlgorithm<T, U>*      createMultiplySurfacesAlgorithm(Type impl);
};

template<typename T>
RemoveNegativeElevationsAlgorithm<T>* KernelUtilsFactory::createRemoveNegativeElevationsAlgorithm() {
	return createRemoveNegativeElevationsAlgorithm<T>(defaultImpl);
}

template<typename T>
InitializeArrayAlgorithm<T>* KernelUtilsFactory::createInitializeArrayAlgorithm() {
	return createInitializeArrayAlgorithm<T>(defaultImpl);
}

template<typename T, typename U>
AddSurfacesAlgorithm<T, U>* KernelUtilsFactory::createAddSurfacesAlgorithm() {
	return createAddSurfacesAlgorithm<T, U>(defaultImpl);
}

template<typename T, typename U>
MultiplySurfacesAlgorithm<T, U>* KernelUtilsFactory::createMultiplySurfacesAlgorithm() {
	return createMultiplySurfacesAlgorithm<T, U>(defaultImpl);
}

template<typename T>
RemoveNegativeElevationsAlgorithm<T>* KernelUtilsFactory::createRemoveNegativeElevationsAlgorithm(Type impl)
{
	switch(impl) {
	case CPU:
	case CPU_OpenMP: return new RemoveNegativeElevationsAlgorithm_CPU_OpenMP<T>();

	case GPU:
	case GPU_CUDA:   return new RemoveNegativeElevationsAlgorithm_GPU_CUDA<T>();
	default:         assert(false); return NULL;
	}
}

template<typename T, typename U>
AddSurfacesAlgorithm<T, U>* KernelUtilsFactory::createAddSurfacesAlgorithm(Type impl)
{
	switch(impl) {
	case CPU:
	case CPU_OpenMP: return new AddSurfacesAlgorithm_CPU_OpenMP<T, U>();

	case GPU:
	case GPU_CUDA:   return new AddSurfacesAlgorithm_GPU_CUDA<T, U>();
	default:         assert(false); return NULL;
	}
}

template<typename T>
InitializeArrayAlgorithm<T>* KernelUtilsFactory::createInitializeArrayAlgorithm(Type impl)
{
	switch(impl) {
	case CPU:
	case CPU_OpenMP: return new InitializeArrayAlgorithm_CPU_OpenMP<T>();

	case GPU:
	case GPU_CUDA:   return new InitializeArrayAlgorithm_GPU_CUDA<T>();
	default:         assert(false); return NULL;
	}
}

template<typename T, typename U>
MultiplySurfacesAlgorithm<T, U>* KernelUtilsFactory::createMultiplySurfacesAlgorithm(Type impl)
{
	switch(impl) {
	case CPU:
	case CPU_OpenMP: return new MultiplySurfacesAlgorithm_CPU_OpenMP<T, U>();

	case GPU:
	case GPU_CUDA:   return new MultiplySurfacesAlgorithm_GPU_CUDA<T, U>();
	default:         assert(false); return NULL;
	}
}

#endif /* KERNELUTILSFACTORY_H_ */
