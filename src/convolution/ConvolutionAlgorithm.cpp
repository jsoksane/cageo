/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ConvolutionAlgorithm.cpp
 */

#include <Eigen/SVD>
#include <fftw3.h>

#include "ConvolutionAlgorithm.h"

template<typename T>
void ConvolutionAlgorithm<T>::createFilter(
		float practicalRange,
		float cellSize)
{
	if(filter != NULL) {
		delete filter;
		filter = NULL;
	}

	filter = ManagedDataFactory::create<filter_t>(getFilterSize(), HOST);
	filter->allocateData();

	filter_t* correlogram = new filter_t[getFilterSize()];

	createCorrelogram(correlogram, practicalRange, cellSize);
	createFilterFromCorrelogram(correlogram);

	if(filterModel == GAUSSIAN) {
		filter_t* filterTemp = new filter_t[getFilterLength()];
		createKernelVector(filterTemp, filter->getData());
		filter->freeData();
		filter->setSize(getFilterLength());
		filter->setData(filterTemp);
	}

	delete[] correlogram;
}

template void ConvolutionAlgorithm<RandomFieldType>::createFilter(float practicalRange, float cellSize);

template<typename T>
void ConvolutionAlgorithm<T>::createCorrelogram(
		filter_t* correlogram,
		float practicalRange,
		float cellSize)
{
	switch(filterModel) {
	case EXPONENTIAL:
		{
			int i = 0;
			for(int y = -filterRadius; y <= filterRadius; y++) {
				for(int x = -filterRadius; x <= filterRadius; x++, i++) {
					correlogram[i] = (filter_t) exp(-sqrt(float(x*x + y*y))/((practicalRange/3.0f)/cellSize));
				}
			}
			break;
		}
	case GAUSSIAN:
		{
			int i = 0;
			for(int y = -filterRadius; y <= filterRadius; y++) {
				for(int x = -filterRadius; x <= filterRadius; x++, i++) {
					correlogram[i] = (filter_t) exp(-(x*x + y*y)/(pow((practicalRange/1.73205080757f)/cellSize, 2)));
				}
			}
			break;
		}
	default:
		{
			std::cout << "Unknown or unsupported autocorrelation model provided: " << filterModel << std::endl;
			break;
		}
	}
}

template<typename T>
void ConvolutionAlgorithm<T>::createFilterFromCorrelogram(filter_t* correlogram)
{
	int kernelSize   = getFilterSize();
	int kernelLength = getFilterLength();
	int fftSize      = kernelLength*(kernelLength/2 + 1);

	fftw_complex*  fft = (fftw_complex*) fftw_malloc(fftSize*sizeof(fftw_complex));
	double*       ifft = new double[kernelSize];
	double* corrDouble = new double[kernelSize];

	for(size_t i = 0; i < kernelSize; i++) {
		corrDouble[i] = (double) correlogram[i];
	}

	fftw_plan plan = fftw_plan_dft_r2c_2d(
						kernelLength, kernelLength,
						corrDouble, fft,
						FFTW_ESTIMATE);

	fftw_execute(plan);

	fftw_destroy_plan(plan);

	delete[] corrDouble;

	for(size_t i = 0; i < fftSize; i++) {
		double re = fft[i][0];
		double im = fft[i][1];
		fft[i][0] = std::sqrt(std::sqrt( re*re + im*im ));
		fft[i][1] = 0.0;
	}

	plan = fftw_plan_dft_c2r_2d(
						kernelLength, kernelLength,
						fft, ifft,
						FFTW_ESTIMATE);

	fftw_execute(plan);

	fftw_destroy_plan(plan);

	fftw_free(fft);

	for(size_t i = 0; i < kernelSize; i++) {
		ifft[i] /= (double) kernelSize;
	}

	const size_t iShift = kernelLength/2;
	const size_t jShift = kernelLength/2;

	filter_t* filterData = filter->getData();

	for(size_t j = 0; j < kernelLength; j++) {
		int newJ = (j+jShift)%kernelLength;

		for(size_t i = 0; i < kernelLength; i++) {
			int newI = (i+iShift)%kernelLength;

			filterData[newJ*kernelLength + newI] = (filter_t) ifft[j*kernelLength + i];
		}
	}

	delete[] ifft;
}

template<typename T>
void ConvolutionAlgorithm<T>::createKernelVector(
		filter_t* kernelVector,
		filter_t* inputKernel)
{
	int filterLength = getFilterLength();

	Eigen::MatrixXd k(filterLength, filterLength);

	for(int j = 0; j < filterLength; j++) {
		for(int i = 0; i < filterLength; i++) {
			k(j,i) = (double) inputKernel[j*filterLength + i];
		}
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(k, Eigen::ComputeThinU);
	Eigen::MatrixXd u = svd.matrixU();
	Eigen::MatrixXd s = svd.singularValues();

	for(int j = 0; j < filterLength; j++) {
		kernelVector[j] = (filter_t) (u(j,0)*std::sqrt((double)s(0)));
	}
}
