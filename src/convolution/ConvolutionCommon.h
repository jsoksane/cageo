/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ConvolutionCommon.h
 */

#ifndef CONVOLUTIONCOMMON_H_
#define CONVOLUTIONCOMMON_H_

/**
 * An enumeration for different filter models used in the convolutions process.
 */
enum FilterModel {
	EXPONENTIAL, /*!< The exponential filter model.*/
	GAUSSIAN /*!< The Gaussian filter model.*/
};

#endif /* CONVOLUTIONCOMMON_H_ */
