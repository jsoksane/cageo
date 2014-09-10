/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * This software contains source code provided by NVIDIA Corporation.
 *
 *
 * \file ConvolutionAlgorithmGPUCUDA.h
 */

#ifndef CONVOLUTIONALGORITHMGPUCUDA_H_
#define CONVOLUTIONALGORITHMGPUCUDA_H_

#include "ConvolutionAlgorithm.h"

/**
 * \brief An NVidia CUDA implementation of the ConvolutionAlgorithm.
 */
template<typename T>
class ConvolutionAlgorithm_GPU_CUDA : public ConvolutionAlgorithm<T> {
public:
	ConvolutionAlgorithm_GPU_CUDA();

	virtual ~ConvolutionAlgorithm_GPU_CUDA() {};

protected:
    /**
     * \brief Perform the convolution with CUDA GPU algorithms.
     */
	void performConvolution(RandomFieldClass<T>* sourceData);

private:
    /**
     * \brief Perform the convolution with CUDA GPU algorithm for exponential
     * filter model.
     *
     * Calculate the CUDA parameters and launch appropriate number of
     * convolutionKernel() CUDA kernels.
     */
	void performConvolution_CUDA(
			T*           output,
			CellGrid<T>* input,
			T*           filter,
			int          filterRadius);

    /**
     * \brief Perform the convolution with CUDA GPU algorithm for Gaussian
     * filter model.
     *
     * Calculate the CUDA parameters and launch appropriate number of
     * convolutionColsKernel() and convolutionRowsKernel() CUDA kernels.
     */
	void performConvolutionSeparable_CUDA(
			CellGrid<T>* input,
			T*           buffer,
			T*           filter,
			int          filterRadius);
};

template<typename T>
ConvolutionAlgorithm_GPU_CUDA<T>::ConvolutionAlgorithm_GPU_CUDA()
{
};

/**
 * \brief Calculate the convolution of the given data with the given filter.
 *
 * The filter is positioned over each cell in the convolution process.
 * When a cell is near the edge of the data, the filter will partly miss
 * the data. In those cases we use the data points from the other side of
 * the data.
 *
 *              +-------------+
 *     filter   |       data  |
 *           +--+--+          |
 *           |  |  |          |
 *           |  |  |          |
 *           +--+--+          |
 *              |             |
 *              +-------------+
 *
 * \param outputData The array to save the convoluted field.
 * \param inputData The array to hold the data to be convoluted.
 * \param filter The array that holds the filter data.
 * \param filterRadius The radius of the \a filter. The size of the \a filter
 *        must be (2 * filterRadius + 1) * (2 * filterRadius + 1).
 * \param dataWidth The width of the \a inputData.
 * \param dataHeight The height of the \a inputData.
 */
template<typename T>
__global__
void convolutionKernel(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight);

/**
 * \brief Calculate the convolution of the \a inputData with \a filter.
 *
 * Identical to the convolutionKernel() except that the height of the \a
 * filter is 1.
 */
template<typename T>
__global__
void convolutionRowsKernel(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight);

/**
 * \brief Calculate the convolution of the \a inputData with \a filter.
 *
 * Identical to the convolutionKernel() except that the width of the \a filter
 * is 1.
 */
template<typename T>
__global__
void convolutionColsKernel(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight);

#endif /* CONVOLUTIONALGORITHMGPUCUDA_H_ */
