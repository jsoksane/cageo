/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file AddSurfacesAlgorithm.h
 */

#ifndef ADDSURFACESALGORITHM_H_
#define ADDSURFACESALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "CellGrid.h"

/**
 * \brief An algorithm to add two CellGrid object together, i.e. adding their
 * values element-wise.
 *
 * This is an abstract class that must be subclassed.
 */
template<typename T, typename U>
class AddSurfacesAlgorithm : public AbstractAlgorithm {
public:
    /**
     * \brief The cells with this value are not added.
     */
	T noDataValue;

public:
	AddSurfacesAlgorithm();
	virtual ~AddSurfacesAlgorithm() {};

    /**
     * \brief Add the values of \a inputA and \a inputB element-wise, and save the
     * result to \a output.
     */
	void execute(CellGrid<T>* inputA, CellGrid<U>* inputB, CellGrid<T>* output);
    /**
     * \brief The implementation-dependent function that must be implemented
     * when subclassing.
     */
	virtual void performAddSurfaces(CellGrid<T>* inputA, CellGrid<U>* inputB, CellGrid<T>* output) = 0;

    /**
     * \brief Set the value that is used as noDataValue.
     */
	void setNoDataValue(T noDataValue);
};

template<typename T, typename U>
AddSurfacesAlgorithm<T, U>::AddSurfacesAlgorithm()
{
	noDataValue = 0;
}

template<typename T, typename U>
void AddSurfacesAlgorithm<T, U>::execute(CellGrid<T>* inputA, CellGrid<U>* inputB, CellGrid<T>* output)
{
	performAddSurfaces(inputA, inputB, output);
}

template<typename T, typename U>
void AddSurfacesAlgorithm<T, U>::setNoDataValue(T noDataValue)
{
	this->noDataValue = noDataValue;
}

#endif /* ADDSURFACESALGORITHM_H_ */
