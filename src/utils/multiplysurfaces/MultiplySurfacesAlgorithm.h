/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file MultiplySurfacesAlgorithm.h
 */

#ifndef MULTIPLYSURFACESALGORITHM_H_
#define MULTIPLYSURFACESALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "CellGrid.h"

/**
 * \brief An abstract algorithm class to multiply the values of two arrays
 * element-wise.
 */
template<typename T, typename U>
class MultiplySurfacesAlgorithm : public AbstractAlgorithm {
protected:
    /**
     * \brief The value used to inform that the grid point does not have a
     * proper value.
     */
	T noDataValue;

public:
    /**
     * \brief The constructor.
     */
	MultiplySurfacesAlgorithm();
    /**
     * \brief The destructor.
     */
	virtual ~MultiplySurfacesAlgorithm() {};

    /**
     * \brief Execute the algorithm.
     */
	void execute(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output);

    /**
     * \brief Multiply the two arrays element-wise.
     */
	virtual void performMultiplySurfaces(
			CellGrid<T>* inputA,
			CellGrid<U>* inputB,
			CellGrid<T>* output) = 0;

    /**
     * \brief Set the value for the \a noDataValue.
     */
	void setNoDataValue(T noDataValue) {
		this->noDataValue = noDataValue;
	}
};

template<typename T, typename U>
MultiplySurfacesAlgorithm<T, U>::MultiplySurfacesAlgorithm()
{
	noDataValue = 0;
}

template<typename T, typename U>
void MultiplySurfacesAlgorithm<T, U>::execute(
		CellGrid<T>* inputA,
		CellGrid<U>* inputB,
		CellGrid<T>* output)
{
	performMultiplySurfaces(inputA, inputB, output);
}



#endif /* MULTIPLYSURFACESALGORITHM_H_ */
