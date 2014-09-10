/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ConvolutionAlgorithmCPUOpenMP.h
 */

#ifndef CONVOLUTIONALGORITHMCPUOPENMP_H_
#define CONVOLUTIONALGORITHMCPUOPENMP_H_

#include "ConvolutionAlgorithm.h"

/**
 * \brief An OpenMP implementation of the ConvolutionAlgorithm.
 */
template<typename T>
class ConvolutionAlgorithm_CPU_OpenMP : public ConvolutionAlgorithm<T> {
public:
	ConvolutionAlgorithm_CPU_OpenMP();
	virtual ~ConvolutionAlgorithm_CPU_OpenMP() {};

protected:
    /**
     * \brief Perform the convolution with OpenMP-parallelized algorithm.
     */
	void performConvolution(RandomFieldClass<T>* sourceData);

private:
    /**
     * \brief Convolution algorithm for exponential filter model.
     */
	void performConvolution_OpenMP(
			T* outputData,
			T* inputData,
			T* filter,
			int filterRadius,
			int dataWidth, int dataHeight);

    /**
     * \brief Convolution algorithm for the Gaussian filter model.
     */
	void performConvolutionRows_OpenMP(
			T* outputData,
			T* inputData,
			T* filter,
			int filterRadius,
			int dataWidth, int dataHeight);

    /**
     * \brief Convolution algorithm for the Gaussian filter model.
     */
	void performConvolutionCols_OpenMP(
			T* outputData,
			T* inputData,
			T* filter,
			int filterRadius,
			int dataWidth, int dataHeight);
};

template<typename T>
ConvolutionAlgorithm_CPU_OpenMP<T>::ConvolutionAlgorithm_CPU_OpenMP()
{
}

template<typename T>
void ConvolutionAlgorithm_CPU_OpenMP<T>::performConvolution(RandomFieldClass<T>* sourceData)
{
	sourceData->toHost();

	FilterModel filterModel = this->filterModel;

    T* h_buffer = new T[sourceData->size];
    memcpy(h_buffer, sourceData->getData(), sourceData->size*sizeof(T));

	if(filterModel == GAUSSIAN) {
		performConvolutionCols_OpenMP(
				h_buffer,
				sourceData->getData(),
				this->filter->getData(), this->getFilterRadius(),
				sourceData->width, sourceData->height);
		performConvolutionRows_OpenMP(
				sourceData->getData(),
				h_buffer,
				this->filter->getData(), this->getFilterRadius(),
				sourceData->width, sourceData->height);

		delete[] h_buffer;
	}
	else if(filterModel == EXPONENTIAL) {
		performConvolution_OpenMP(
				h_buffer,
				sourceData->getData(),
				this->filter->getData(), this->getFilterRadius(),
				sourceData->width, sourceData->height);
		sourceData->freeData();
		sourceData->setData(h_buffer);
	}
}

template<typename T>
void ConvolutionAlgorithm_CPU_OpenMP<T>::performConvolution_OpenMP(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight)
{
	int filterLength = filterRadius*2 + 1;

#pragma omp parallel for
	for(int y = 0; y < dataHeight; y++) {
		for(int x = 0; x < dataWidth; x++)
		{
			T sum = 0.0f;

			if (x < filterRadius || x >  dataWidth - filterRadius-1 ||
				y < filterRadius || y > dataHeight - filterRadius-1)
			{
				for (int filterY = 0; filterY < filterLength; filterY++)
				{
					int sy = y - filterRadius + filterY;
					if(sy < 0)
						sy += dataHeight;
					else if(sy >= dataHeight)
						sy -= dataHeight;

					for (int filterX = 0; filterX < filterLength; filterX++)
					{
						int sx = x - filterRadius + filterX;
						if(sx < 0)
							sx += dataWidth;
						else if(sx >= dataWidth)
							sx -= dataWidth;

						sum += inputData[sy*dataWidth + sx]*filter[filterY*filterLength + filterX];
					}
				}
				outputData[y*dataWidth + x] = sum;
			}
			else {
				for (int filterY = 0; filterY < filterLength; filterY++)
				{
					int sy = y - filterRadius + filterY;

					for (int filterX = 0; filterX < filterLength; filterX++)
					{
						int sx = x - filterRadius + filterX;

						sum += inputData[sy*dataWidth + sx] * filter[filterY*filterLength + filterX];
					}
				}
				outputData[y*dataWidth + x] = sum;
			}
		}
	}

}

template<typename T>
void ConvolutionAlgorithm_CPU_OpenMP<T>::performConvolutionRows_OpenMP(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight)
{
	int filterLength = filterRadius*2 + 1;

#pragma omp parallel for
	for(int y = 0; y < dataHeight; y++) {
		for(int x = 0; x < dataWidth; x++)
		{
			if (x < filterRadius || x > dataWidth - filterRadius - 1)
			{
				T sum = 0.0f;
				for (int filterX = 0; filterX < filterLength; filterX++)
				{
					int sx = x - filterRadius + filterX;
					if(sx < 0)
						sx += dataWidth;
					else if(sx >= dataWidth)
						sx -= dataWidth;

					sum += inputData[y*dataWidth + sx]*filter[filterX];
				}
				outputData[y*dataWidth + x] = sum;
			}
			else {
				T sum = 0.0f;
				for (int filterX = 0; filterX < filterLength; filterX++)
				{
					int sx = x - filterRadius + filterX;

					sum += inputData[y*dataWidth + sx]*filter[filterX];
				}
				outputData[y*dataWidth + x] = sum;
			}
		}
	}
}

template<typename T>
void ConvolutionAlgorithm_CPU_OpenMP<T>::performConvolutionCols_OpenMP(
		T* outputData,
		T* inputData,
		T* filter,
		int filterRadius,
		int dataWidth, int dataHeight)
{
	int filterLength = filterRadius*2 + 1;

#pragma omp parallel for
	for(int y = 0; y < dataHeight; y++) {
		for(int x = 0; x < dataWidth; x++)
		{
			if (y < filterRadius || y > dataHeight - filterRadius-1)
			{
				T sum = 0.0f;
				for (int filterY = 0; filterY < filterLength; filterY++)
				{
					int sy = y - filterRadius + filterY;
					if(sy < 0)
						sy += dataHeight;
					else if(sy >= dataHeight)
						sy -= dataHeight;

					sum += inputData[sy*dataWidth + x] * filter[filterY];
				}
				outputData[y*dataWidth + x] = sum;
			}
			else {
				T sum = 0.0f;
				for (int filterY = 0; filterY < filterLength; filterY++)
				{
					int sy = y - filterRadius + filterY;

					sum += inputData[sy*dataWidth + x] * filter[filterY];
				}
				outputData[y*dataWidth + x] = sum;
			}
		}
	}
}

#endif /* CONVOLUTIONALGORITHMCPUOPENMP_H_ */
