/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ToByteArrayAlgorithm.h
 */

#ifndef TOBYTEARRAYALGORITHM_H_
#define TOBYTEARRAYALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "ManagedData.h"
#include "CellGrid.h"

#include "jpge.h"

typedef unsigned char Byte_t;

template<typename T>
class ToByteArrayAlgorithm : public AbstractAlgorithm {
public:
	ToByteArrayAlgorithm() {};
	virtual ~ToByteArrayAlgorithm() {};

	void execute(
			CellGrid<Byte_t>* output,
			CellGrid<T>*      input);

protected:
	virtual
	void convertToByteArray(
			CellGrid<Byte_t>* output,
			CellGrid<T>*      input) = 0;
};

template<typename T>
void ToByteArrayAlgorithm<T>::execute(
		CellGrid<Byte_t>* output,
		CellGrid<T>*      input)
{
    convertToByteArray(output, input);
}


#endif /* TOBYTEARRAYALGORITHM_H_ */
