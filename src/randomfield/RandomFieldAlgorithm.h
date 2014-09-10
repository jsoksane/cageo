/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RandomFieldAlgorithm.h
 */

#ifndef RANDOMFIELDALGORITHM_H_
#define RANDOMFIELDALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "RandomFieldClass.h"
#include "Logging.h"

/**
 * \brief The algorithm that fills the given RandomFieldClass with random
 * numbers.
 *
 * This class cannot be used directly, but must be subclassed. When
 * subclassing, an implementation for the abstract function
 * \a generateRandomField must be provided.
 */
template<typename T>
class RandomFieldAlgorithm : public AbstractAlgorithm {
protected:
    /**
     * \brief The mean value of the random number distribution.
     */
	float mean;
    /**
     * \brief The standard deviation of the random number distribution.
     */
	float standardDeviation;

public:
    /**
     * \brief Construct the RandomFieldAlgorithm object.
     */
	RandomFieldAlgorithm();
    /**
     * \brief The destructor of the object.
     */
	virtual ~RandomFieldAlgorithm() {};

    /**
     * \brief Fill the given RandomFieldClass with random numbers.
     */
	void execute(RandomFieldClass<T>* randomField, int seed);
    /**
     * \brief Set the mean value for the random number distribution.
     */
	void setMean(float mean);
    /**
     * \brief Set the standard deviation for the random number distribution.
     */
	void setStandardDeviation(float standardDeviation);

protected:
    /**
     * \brief Fill the RandomFieldClass with random numbers that are generated
     * using the given seed.
     */
	virtual void generateRandomField(
			RandomFieldClass<T>* randomField,
			int seed) = 0;
};

typedef RandomFieldAlgorithm<RandomFieldType> RandomFieldAlgorithm_t;

template<typename T>
RandomFieldAlgorithm<T>::RandomFieldAlgorithm()
{
	this->mean              = 0.0f;
	this->standardDeviation = 1.0f;
}

template<typename T>
void RandomFieldAlgorithm<T>::execute(RandomFieldClass<T>* randomField,
                                      int seed)
{
	LOG_TRACE("RANDOM FIELD");

    LOG_TRACE_ALGORITHM("Processing");

    generateRandomField(randomField, seed);

    randomField->cacheDataIfNeeded("randomField");
}

template<typename T>
void RandomFieldAlgorithm<T>::setMean(float mean)
{
	this->mean = mean;
}

template<typename T>
void RandomFieldAlgorithm<T>::setStandardDeviation(float standardDeviation)
{
	this->standardDeviation = standardDeviation;
}

#endif /* RANDOMFIELDALGORITHM_H_ */
