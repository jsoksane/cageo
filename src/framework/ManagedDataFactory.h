/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ManagedDataFactory.h
 */

#ifndef MANAGEDDATAFACTORY_H_
#define MANAGEDDATAFACTORY_H_

#include "ManagedData.h"
#include "ManagedDataCUDA.h"
#include "GlobalParameters.h"

/**
 * \brief A factory that constructs ManagedData object based on the framework
 * used.
 */
class ManagedDataFactory {
public:
    /**
     * \brief Create an empty ManagedData object.
     */
	template<typename T>
	static ManagedData<T>* create();

    /**
     * \brief Create an empty ManagedData for \a dataSize values from
     * \a storageType.
     */
	template<typename T>
	static ManagedData<T>* create(
			size_t dataSize,
			StorageType storageType);

    /**
     * \brief Create a new ManagedData from \a storageTypeby copying.
     */
	template<typename T>
	static ManagedData<T>* create(
			ManagedData<T>* model,
			StorageType storageType,
			CopyType copyType);
};

template<typename T>
ManagedData<T>* ManagedDataFactory::create()
{
	switch(G::framework) {
	case G::CUDA: return new ManagedDataCUDA<T>();
	default:           return NULL;
	}
}

template<typename T>
ManagedData<T>* ManagedDataFactory::create(
		size_t dataSize,
		StorageType storageType)
{
	switch(G::framework) {
	case G::CUDA: return new ManagedDataCUDA<T>(dataSize, storageType);
	default:           return NULL;
	}
}

template<typename T>
ManagedData<T>* ManagedDataFactory::create(
		ManagedData<T>* model,
		StorageType storageType,
		CopyType copyType)
{
	switch(G::framework) {
	case G::CUDA: return new ManagedDataCUDA<T>(model, storageType, copyType);
	default:           return NULL;
	}
}

#endif /* MANAGEDDATAFACTORY_H_ */
