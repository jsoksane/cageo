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
 * The code can be found from the document
 * http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
 *
 *
 * \file Reduce.cuh
 *
 */

#ifndef REDUCE_CUH_
#define REDUCE_CUH_

/**
 * \brief A CUDA implementation to reduce boolean values with AND conditional.
 *
 * This kernel will reduce an array of \a N boolean into an array of \a N_r
 * boolean using AND condition in the reduction. The value of \a N_r is
 * coupled to the thread block size \a blockSize used to launch the kernel:
 *
 *                  N
 *     N_r >= -------------,
 *             2 * blockSize
 *
 * The N_r can be calculated using a helper function calculate_reduced_size().
 *
 * The length of the \a g_odata must be at least \a N_r.
 *
 * The kernel is launched with \a N_r thread blocks of size \a blockSize as
 *
 *     reduceAnd<blockSize><<<N_r, blockSize, blockSize * sizeof(bool)>>>(g_idata, g_odata, N).
 *
 * Each thread block reads 2 * \a blockSize values and reduces them into a single
 * value, which is stored in \a g_odata at the position blockIdx.x
 *
 * \param g_idata The input array to be reduced.
 * \param g_odata The output array.
 * \param dataSize The size of the \a g_idata.
 */
template <unsigned int blockSize>
__global__
void reduceAnd(bool *g_idata, bool *g_odata, unsigned int dataSize)
{
    // This is allocated through the kernel launch parameter.
    extern __shared__ bool sdata[];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;

    if(i+blockSize < dataSize) {
		sdata[tid] = g_idata[i] &= g_idata[i + blockSize];
    }
    else if(i < dataSize){
    	sdata[tid] = g_idata[i];
    }
    else {
    	sdata[tid] = true;
    }

    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] &= sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] &= sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] &= sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] &= sdata[tid + 32]; __syncthreads(); }
        if (blockSize >=  32) { sdata[tid] &= sdata[tid + 16]; __syncthreads(); }
        if (blockSize >=  16) { sdata[tid] &= sdata[tid +  8]; __syncthreads(); }
        if (blockSize >=   8) { sdata[tid] &= sdata[tid +  4]; __syncthreads(); }
        if (blockSize >=   4) { sdata[tid] &= sdata[tid +  2]; __syncthreads(); }
        if (blockSize >=   2) { sdata[tid] &= sdata[tid +  1]; __syncthreads(); }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

#endif /* REDUCE_CUH_ */
