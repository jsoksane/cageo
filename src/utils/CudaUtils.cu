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
 * \file CudaUtils.cu
 */

#include <cuda.h>
#include <stdio.h>
#include <sstream>

#include "CudaUtils.cuh"
#include "Utils.h"

namespace CudaUtils {

int deviceCount = 0;
cudaDeviceProp deviceProp;

/**
 * \brief Construct a name for the CUDA device from the PCI address. The name
 * is unique on this host.
 */
std::string constructDeviceName(const cudaDeviceProp &dev) {
    std::stringstream ss;
    ss << dev.name << " - " << dev.pciBusID << ":" << dev.pciDomainID << "." << dev.pciDeviceID;
    return ss.str();
};

/**
 * \brief Load the device properties of the device with id number \a device
 * into global variable CudaUtils::deviceProp. The names and id numbers of the
 * all devices are printed.
 */
void loadDeviceInfo(int device)
{
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);
    if (error_id != cudaSuccess) {
        printf("\tcudaGetDeviceCount returned %d\n-> %s\n", (int) error_id, cudaGetErrorString(error_id) );
        exit(1);
    }
    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) {
    	printf("\tThere is no device supporting CUDA\n");
    	return;
    }
    else {
    	printf("\tFound %d CUDA Capable device(s)\n", deviceCount);
    }

    cudaGetDeviceProperties(&deviceProp, device);

    for(int d = 0; d < deviceCount; d++) {
    	cudaDeviceProp devInfo;
    	cudaGetDeviceProperties(&devInfo, d);
    	printf("\tDevice %d: \"%s\"\n", d, deviceProp.name);
    }

}

/**
 * \brief Calculate the CUDA kernel thread block dimensions based on the
 * dimensions of the input data size.
 *
 * @param dims The dimensions of the 2D data.
 */
void calcGridSize(const dim3& block, const dim3& dims, dim3* grid)
{
	dim3 maxBlockSize;
	maxBlockSize.x = deviceProp.maxThreadsDim[0];
	maxBlockSize.y = deviceProp.maxThreadsDim[1];
	maxBlockSize.z = deviceProp.maxThreadsDim[2];

	int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
	int maxGridSize  = deviceProp.maxGridSize[0];

	if(block.x > 1)
		grid->x = (dims.x + (block.x - 1))/block.x;
	if(block.y > 1)
		grid->y = (dims.y + (block.y - 1))/block.y;
	if(block.z > 1)
		grid->z = (dims.z + (block.z - 1))/block.z;

	bool fail = false;
	if(block.x > maxBlockSize.x || block.y > maxBlockSize.y || block.z > maxBlockSize.z) {
		printf("A thread block dimension is larger than the maximum supported thread block dimension. Consider lowering the thread block dimension. Exiting...\n");
		fail = true;
	}

	if(block.x*block.y*block.z > maxThreadsPerBlock) {
		printf("Thread block size needed (%d) is larger than maximum supported thread block size (%d). Consider lowering the thread block size. Exiting...\n", block.x*block.y*block.z, maxThreadsPerBlock);
		fail = true;
	}

	if(grid->x*grid->y*grid->z > maxGridSize) {
		printf("Grid size needed (%d) is larger than maximum supported grid size (%d). Exiting...\n", grid->x*grid->y*grid->z, maxGridSize);
		fail = true;
	}

	if(fail) {
		exit(EXIT_SUCCESS);
	}
}

size_t getUsedMemory() {
	size_t freeMemory;
	size_t totalMemory;
	cuMemGetInfo(&freeMemory, &totalMemory);

	return totalMemory - freeMemory;
}

size_t getTotalMemory() {
	size_t freeMemory;
	size_t totalMemory;
	cuMemGetInfo(&freeMemory, &totalMemory);

	return totalMemory;
}

}
