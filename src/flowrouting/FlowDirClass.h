/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowDirClass.h
 */

#ifndef FLOWDIRCLASS_H_
#define FLOWDIRCLASS_H_

#include "CellGrid.h"

/**
 * \brief A grid that contains the flow directions of the cells.
 */
template<class T>
class FlowDirClass : public CellGrid<T> {
public:
	FlowDirClass();

	template<class U>
	FlowDirClass(
			CellGrid<U>& model,
			CopyType copyType);

	template<class U>
	FlowDirClass(
			CellGrid<U>& model,
			StorageType storageType,
			CopyType copyType);

	FlowDirClass(
			int width,
			int height,
			StorageType ptrType);
	virtual ~FlowDirClass();
};

typedef char2 FlowDirDataType;
typedef FlowDirClass<FlowDirDataType> FlowDir_t;

/*
 * Definitions
 */

template<class T>
FlowDirClass<T>::FlowDirClass() : CellGrid<T>()
{
}

template<class T>
template<class U>
FlowDirClass<T>::FlowDirClass(
		CellGrid<U>& model,
		CopyType copyType)
: CellGrid<T>(model, copyType)
{
}

template<class T>
template<class U>
FlowDirClass<T>::FlowDirClass(
		CellGrid<U>& model,
		StorageType storageType,
		CopyType copyType)
: CellGrid<T>(model, storageType, copyType)
{
}

template<class T>
FlowDirClass<T>::FlowDirClass(
		int width,
		int height,
		StorageType ptrType) : CellGrid<T>(width, height, ptrType)
{
}

template<class T>
FlowDirClass<T>::~FlowDirClass()
{
}

#endif /* FLOWDIRCLASS_H_ */
