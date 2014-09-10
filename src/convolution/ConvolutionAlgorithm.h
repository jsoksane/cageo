/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ConvolutionAlgorithm.h
 */

#ifndef CONVOLUTIONALGORITHM_H_
#define CONVOLUTIONALGORITHM_H_

#include <math.h>

#include "AbstractAlgorithm.h"
#include "ConvolutionCommon.h"
#include "ManagedDataFactory.h"
#include "RandomFieldClass.h"
#include "Logging.h"

typedef float filter_t;

/**
 * \brief The algorithm that performs the process convolution for the given
 * random field.
 *
 * This class cannot be used directly, but must be subclassed, and an
 * implementation for the abstract function performConvolution() must be
 * provided.
 */
template<typename T>
class ConvolutionAlgorithm : public AbstractAlgorithm {
protected:
    /**
     * \brief The process convolution filter array.
     */
	ManagedData<filter_t>* filter;

    /**
     * \brief The radius of the filter.
     */
	int filterRadius;
    /**
     * \brief The type of the filter.
     */
	FilterModel filterModel;

public:
	ConvolutionAlgorithm();
	virtual ~ConvolutionAlgorithm() {};

    /**
     * \brief Perform the convolution.
     */
	void execute(RandomFieldClass<T>* randomField);

    /**
     * \brief Calculate the filter matrix values for given \a practicalRange
     * and \a cellSize.
     */
	void createFilter(float practicalRange, float cellSize);
    /**
     * \brief Return the radius of the filter.
     */
	int  getFilterRadius();
    /**
     * \brief Return the length (or width) of the filter.
     */
	int  getFilterLength();
    /**
     * \brief Return the number of data points in the convolution filter.
     */
	int  getFilterSize();
    /**
     * \brief Set the type of filter used.
     */
    void setFilterModel(FilterModel filterModel);
    /**
     * \brief Set the radius of the filter.
     */
	void setFilterRadius(int filterRadius);

protected:
    /**
     * \brief Perform the convolution on the given data.
     *
     * This implementation-dependent function, and therefore must be
     * implemented when subclassing.
     */
	virtual void performConvolution(RandomFieldClass<T>* sourceData) = 0;

    /**
     * \brief Compute the correlogram array that is used to compute the filter
     * array.
     */
	void createCorrelogram(
			filter_t* correlogram,
			float practicalRange,
			float cellSize);
    /**
     * \brief Compute the filter array from the computed correlogram array.
     */
	void createFilterFromCorrelogram(filter_t* correlogram);
	void createKernelVector(
			filter_t* kernelVector,
			filter_t* inputKernel);
};
typedef ConvolutionAlgorithm<RandomFieldType> ConvolutionAlgorithm_t;

template<typename T>
ConvolutionAlgorithm<T>::ConvolutionAlgorithm()
{
	this->filter       = NULL;
	this->filterRadius = 5;
	this->filterModel  = GAUSSIAN;
}

/**
 * \brief Execute the performConvolution() function.
 */
template<typename T>
void ConvolutionAlgorithm<T>::execute(
		RandomFieldClass<T>* randomField)
{
	LOG_TRACE("CONVOLUTION");

	if(filter == NULL) {
		std::cout << "No filter created! Exiting..." << std::endl;
		exit(EXIT_FAILURE);
	}

    LOG_TRACE_ALGORITHM("Processing");

    performConvolution(randomField);

    LOG_TRACE_ALGORITHM("processing");
}

template<typename T>
void ConvolutionAlgorithm<T>::setFilterRadius(int filterRadius)
{
	this->filterRadius = filterRadius;
}

template<typename T>
int ConvolutionAlgorithm<T>::getFilterRadius()
{
	return filterRadius;
}

template<typename T>
int ConvolutionAlgorithm<T>::getFilterLength()
{
	return 2*getFilterRadius() + 1;
}

template<typename T>
int ConvolutionAlgorithm<T>::getFilterSize()
{
	return getFilterLength()*getFilterLength();
}

template<typename T>
void ConvolutionAlgorithm<T>::setFilterModel(FilterModel filterModel)
{
	this->filterModel = filterModel;
}

#endif /* CONVOLUTIONALGORITHM_H_ */
