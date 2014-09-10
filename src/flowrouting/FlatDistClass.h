/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlatDistClass.h
 */

#ifndef FLATDISTCLASS_H_
#define FLATDISTCLASS_H_

#include "CellGrid.h"

/**
 * \brief A grid that contains the information whether the cells are flat.
 */
template<class T>
class FlatDistClass : public CellGrid<T> {
public:
	FlatDistClass();

    /**
     * \brief Construct a new FlatDistClass by copying.
     */
	template<class U>
	FlatDistClass(
			CellGrid<U>& model,
			CopyType copyType);

	template<class U>
	FlatDistClass(
			CellGrid<U>& model,
			StorageType storageType,
			CopyType copyType);

	FlatDistClass(
			int width,
			int height,
			StorageType ptrType);
	virtual ~FlatDistClass();
};

typedef int FlatDistType;
typedef FlatDistClass<FlatDistType> FlatDist_t;

/*
 * Definitions
 */

template<class T>
FlatDistClass<T>::FlatDistClass() : CellGrid<T>()
{
}

template<class T>
template<class U>
FlatDistClass<T>::FlatDistClass(
		CellGrid<U>& model,
		CopyType copyType)
: CellGrid<T>(model, copyType)
{
}

template<class T>
template<class U>
FlatDistClass<T>::FlatDistClass(
		CellGrid<U>& model,
		StorageType storageType,
		CopyType copyType)
: CellGrid<T>(model, storageType, copyType)
{
}

template<class T>
FlatDistClass<T>::FlatDistClass(
		int width,
		int height,
		StorageType ptrType) : CellGrid<T>(width, height, ptrType)
{
}

template<class T>
FlatDistClass<T>::~FlatDistClass()
{
}

#endif /* FLATDISTCLASS_H_ */
