/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ProcLaterClass.h
 */

#ifndef PROCLATERCLASS_H_
#define PROCLATERCLASS_H_

#include "CellGrid.h"

/**
 * \brief A grid used to keep track whether the elevation of the cells during
 * the pit filling may still change.
 */
template<class T>
class ProcLaterClass : public CellGrid<T> {
public:
	ProcLaterClass();

	ProcLaterClass(
			int width,
			int height,
			StorageType ptrType);
	virtual ~ProcLaterClass();

	template<typename U>
	ProcLaterClass(
			CellGrid<U>& model,
			CopyType copyType) : CellGrid<T>(model, copyType) {}

};

typedef ProcLaterClass<bool> ProcLater_t;

/*
 * Definitions
 */

template<class T>
ProcLaterClass<T>::ProcLaterClass() : CellGrid<T>()
{
}

template<class T>
ProcLaterClass<T>::ProcLaterClass(
		int width,
		int height,
		StorageType ptrType) : CellGrid<T>(width, height, ptrType)
{
	this->clearData();
}

template<class T>
ProcLaterClass<T>::~ProcLaterClass()
{
}


#endif /* PROCLATERCLASS_H_ */
