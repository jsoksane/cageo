/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ReduceUtils.h
 */
#ifndef REDUCE_UTILS_H_
#define REDUCE_UTILS_H_

/**
 * \brief Calculate the size for the reduced array that is compatible with the
 * reduceAnd() kernel with given size of data and thread block.
 *
 * \param dataSize The length of the data to be reduced.
 * \param reduction_block_size The size of the thread block used to launch the
 *                             kernels.
 */
int calculate_reduced_size(unsigned int dataSize,
                           unsigned int reduction_block_size);

#endif
