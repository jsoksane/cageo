/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file InitializeArrayAlgorithm.h
 */

#ifndef INITIALIZEARRAYALGORITHM_H_
#define INITIALIZEARRAYALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "CellGrid.h"

/**
 * \brief An abstract class representing an algorithm that fills the given
 * data object with given value.
 */
template<typename T>
class InitializeArrayAlgorithm : public AbstractAlgorithm {
protected:
    /**
     * \brief The value used to fill the data object.
     */
	T initializeValue;

public:
    /**
     * \brief The constructor. The \a initializeValue is set to zero.
     */
	InitializeArrayAlgorithm() : initializeValue(0) {};
    /**
     * \brief The destructor.
     */
	virtual ~InitializeArrayAlgorithm() {};

    /**
     * \brief Execute the algorithm.
     */
	void execute(CellGrid<T>* input);
    /**
     * \brief Fill the data object with algorithm implementation that is
     * compatible with the used framework.
     */
	virtual void performInitializeArray(CellGrid<T>* input) = 0;

    /**
     * \brief Set the value for the \a ìnitializeValue.
     */
	void setInitializeValue(T initializeValue);
};

template<typename T>
void InitializeArrayAlgorithm<T>::setInitializeValue(T initializeValue)
{
	this->initializeValue = initializeValue;
}

template<typename T>
void InitializeArrayAlgorithm<T>::execute(CellGrid<T>* input)
{
	performInitializeArray(input);
}

#endif /* INITIALIZEARRAYALGORITHM_H_ */
