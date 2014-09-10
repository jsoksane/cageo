/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file AbstractAlgorithm.h
 */

#ifndef ABSTRACTALGORITHM_H_
#define ABSTRACTALGORITHM_H_

#include "ManagedData.h"

/**
 * \brief An abstract class, from which the different algorithm classes should
 * be inherited.
 */
class AbstractAlgorithm {

public:
	AbstractAlgorithm() {};
	virtual ~AbstractAlgorithm() {};
};

#endif /* ABSTRACTALGORITHM_H_ */
