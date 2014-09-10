/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowTraceDataClass.h
 */

#ifndef FLOWTRACEDATACLASS_H_
#define FLOWTRACEDATACLASS_H_

#include "CellGrid.h"

/**
 * \brief A grid that contains the id values of the streams that the flow from
 * the cells will lead to.
 */
template<class T>
class FlowTraceDataClass : public CellGrid<T> {
public:
	FlowTraceDataClass();

	template<typename U>
	FlowTraceDataClass(
			CellGrid<U>& model,
			CopyType copyType);

	template<typename U>
	FlowTraceDataClass(
			CellGrid<U>& model,
			StorageType storageType,
			CopyType copyType);

	FlowTraceDataClass(
			int width,
			int height,
			StorageType ptrType);
	virtual ~FlowTraceDataClass();
};

typedef int FlowTraceDataType;
typedef FlowTraceDataClass<FlowTraceDataType> FlowTraceData_t;

/*
 * Definitions
 */

template<class T>
FlowTraceDataClass<T>::FlowTraceDataClass() : CellGrid<T>()
{
}

template<class T>
template<typename U>
FlowTraceDataClass<T>::FlowTraceDataClass(
		CellGrid<U>& model,
		CopyType copyType)
: CellGrid<T>(model, copyType)
{
}

template<class T>
template<typename U>
FlowTraceDataClass<T>::FlowTraceDataClass(
		CellGrid<U>& model,
		StorageType storageType,
		CopyType copyType)
: CellGrid<T>(model, storageType, copyType)
{
}

template<class T>
FlowTraceDataClass<T>::FlowTraceDataClass(
		int width,
		int height,
		StorageType ptrType) : CellGrid<T>(width, height, ptrType)
{
}

template<class T>
FlowTraceDataClass<T>::~FlowTraceDataClass()
{
}

#endif /* FLOWTRACEDATACLASS_H_ */
