/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RandomFieldClass.h
 */

#ifndef RANDOMFIELDCLASS_H_
#define RANDOMFIELDCLASS_H_

#include "CellGrid.h"

/**
 * \brief A grid that contains random numbers.
 *
 * A class that is used to store random numbers on grid created by the
 * RandomFieldAlgorithm class.
 */
template<typename T>
class RandomFieldClass : public CellGrid<T> {
public:
    /**
     * \brief Construct an empty RandomFieldClass instance.
     */
	RandomFieldClass() {};

    /**
     * \brief Construct a RandomFieldClass by copying from given CellGrid
     * instance.
     */
	template<typename U>
	RandomFieldClass(
			CellGrid<U>& model,
			CopyType copyType) : CellGrid<T>(model, copyType) {}

    /**
     * \brief Construct a RandomFieldClass of given width and height, and
     * allocate memory for the data from ptrType.
     */
	RandomFieldClass(
			int width,
			int height,
			StorageType ptrType) : CellGrid<T>(width, height, ptrType) {};

    /**
     * \brief Destructor.
     */
	virtual ~RandomFieldClass() {};
};

typedef float RandomFieldType;
typedef RandomFieldClass<RandomFieldType> RandomField_t;

/*
 * DEFINITIONS
 */

#endif /* RANDOMFIELDCLASS_H_ */
