/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file CatchmentBorderClass.h
 */

#ifndef CATCHMENTBORDERCLASS_H_
#define CATCHMENTBORDERCLASS_H_

#include "CellGrid.h"

/**
 * \brief A grid that contains the catchment borders.
 */
template<typename T>
class CatchmentBorderClass : public CellGrid<T> {
public:
	CatchmentBorderClass();

	template<typename U>
	CatchmentBorderClass(
			CellGrid<U>& model,
			StorageType storageType,
			CopyType copyType);
	template<typename U>

    /**
     * \brief Construct a new CatchmentBorderClass by copying.
     */
	CatchmentBorderClass(
			CellGrid<U>& model,
			CopyType copyType);

	virtual ~CatchmentBorderClass();
};

typedef float CatchmentBorderType;
typedef CatchmentBorderClass<CatchmentBorderType> CatchmentBorder_t;

/*
 * Definitions
 */

template<class T>
CatchmentBorderClass<T>::CatchmentBorderClass()
{
}

template<class T>
template<typename U>
CatchmentBorderClass<T>::CatchmentBorderClass(
		CellGrid<U>& model,
		StorageType storageType,
		CopyType copyType)
: CellGrid<T>(model, storageType, copyType)
{
	CellGrid<T>::clearData();
}

template<class T>
template<typename U>
CatchmentBorderClass<T>::CatchmentBorderClass(
		CellGrid<U>& model,
		CopyType copyType)
: CellGrid<T>(model, copyType)
{
	CellGrid<T>::clearData();
}

template<class T>
CatchmentBorderClass<T>::~CatchmentBorderClass()
{
}


#endif /* CATCHMENTBORDERCLASS_H_ */
