/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RemoveNegativeElevationsAlgorithm.h
 */

#ifndef REMOVENEGATIVEELEVATIONSALGORITHM_H_
#define REMOVENEGATIVEELEVATIONSALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "CellGrid.h"

/**
 * \brief An algorithm to remove negative values from the given CellGrid.
 *
 * This class must be subclassed, and the
 * performRemoveNegativeElevationValues() function must be implemented.
 */
template<typename T>
class RemoveNegativeElevationsAlgorithm : public AbstractAlgorithm {
public:
	T noDataValue;

public:
	RemoveNegativeElevationsAlgorithm();
	virtual ~RemoveNegativeElevationsAlgorithm() {};

    /**
     * \brief Set the value that is used as noDataValue in the input data.
     */
	void setNoDataValue(T noDataValue);

    /**
     * \brief Replace the negative values from the given CellGrid with 1.0.
     */
	void execute(CellGrid<T>* input);
	virtual void performRemoveNegativeElevationValues(CellGrid<T>* input) = 0;
};

template<typename T>
RemoveNegativeElevationsAlgorithm<T>::RemoveNegativeElevationsAlgorithm()
{
	this->noDataValue = 0;
}

template<typename T>
void RemoveNegativeElevationsAlgorithm<T>::setNoDataValue(T noDataValue)
{
	this->noDataValue = noDataValue;
}

template<typename T>
void RemoveNegativeElevationsAlgorithm<T>::execute(CellGrid<T>* input)
{
	performRemoveNegativeElevationValues(input);
}

#endif /* REMOVENEGATIVEELEVATIONSALGORITHM_H_ */
