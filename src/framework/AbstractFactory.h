/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file AbstractFactory.h
 */

#ifndef ABSTRACTFACTORY_H_
#define ABSTRACTFACTORY_H_

/**
 * \brief A model factory class to produce instances of classes that are
 * compatible with the used implementation.
 *
 * If there are implementation-dependent classes, the compatible instances of
 * those classes are created using the factory for the class in question.
 *
 * The convention is that when subclassing the class, the function create()
 * should be implemented that returns the compatible instance.
 */
class AbstractFactory {
public:
    /**
     * \brief The currently available implementations.
     */
	enum Type {
		CPU, /*!< A general CPU-only implementation. Default to CPU_OpenMP. */
		CPU_OpenMP, /*!< A CPU implementation that uses OpenMP for parallelization. */

		GPU, /*!< A general GPU implementation. Defaults to GPU_CUDA. */
		GPU_CUDA, /*!< The default GPU implementation. */
	};
    /**
     * \brief The currently used implementation.
     */
	static Type defaultImpl;

public:
    /**
     * \brief The default constructor.
     */
	AbstractFactory();
    /**
     * \brief The destructor.
     */
	virtual ~AbstractFactory();
};

#endif /* ABSTRACTFACTORY_H_ */
