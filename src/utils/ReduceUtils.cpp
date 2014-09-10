/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ReduceUtils.cpp
 */
#include "ReduceUtils.h"

int calculate_reduced_size(unsigned int dataSize,
                           unsigned int reduction_block_size) {
     int reducedSize = (dataSize + 1)/(2 * reduction_block_size);
     if (reducedSize * 2 * reduction_block_size < dataSize) {
         reducedSize++;
     }
     return reducedSize;
};
