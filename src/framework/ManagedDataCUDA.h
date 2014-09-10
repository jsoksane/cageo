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
 * \file ManagedDataCUDA.h
 */

#ifndef MANAGEDDATACUDA_H_
#define MANAGEDDATACUDA_H_

#include "ManagedData.h"

/**
 * \brief NVidia CUDA compatible implementation of the ManagedData class.
 *
 * This class is able to move data between the hard drive, RAM and GPU
 * devices.
 */
template<typename T>
class ManagedDataCUDA : public ManagedData<T> {
public:
    /**
     * \brief Create an empty ManagedDataCUDA object.
     */
	ManagedDataCUDA(): ManagedData<T>() {};
    /**
     * \brief Create an empty ManagedDataCUDA object that will contain
     * \a dataSize values and the data is located in \a storageType.
     */
	ManagedDataCUDA(
			size_t dataSize,
			StorageType storageType) : ManagedData<T>(dataSize, storageType) {};
    /**
     * \brief Create a new ManagedDataCUDA object by copying from given
     * ManagedData object \a model.
     */
	ManagedDataCUDA(
			ManagedData<T>* model,
			StorageType storageType,
			CopyType copyType);
    /**
     * \brief The destructor.
     */
	virtual ~ManagedDataCUDA();

protected:
    /**
     * \brief et all the values in the allocated memory to 0.
     */
	virtual void clearDataImpl() const;
    /**
     * \brief Copy the values from the given array to this object.
     */
	virtual void copyDataImpl(const T* data) const;
    /**
     * \brief Transfer the data from HOST to DEVICE.
     */
	virtual void toDeviceImpl();
    /**
     * \brief Transfer the data from DEVICE to HOST.
     */
	virtual void toHostImpl();
    /**
     * \brief Allocate memory for the data from the DEVICE.
     */
	virtual void allocateDataImpl();
    /**
     * \brief Free the allocated data from the DEVICE.
     */
	virtual void freeDataImpl();

private:
    /**
     * \brief A helper function to construct new ManagedDataCUDA by copying.
     */
	void copyConstructor(
			ManagedData<T>* model,
			StorageType storageType,
			CopyType copyType);
};

template<typename T>
ManagedDataCUDA<T>::ManagedDataCUDA(
		ManagedData<T>* model,
		StorageType storageType,
		CopyType copyType) : ManagedData<T>(model, storageType, copyType)
{
	copyConstructor(model, storageType, copyType);
};

/**
 * If memory is allocated from the device, free it.
 */
template<typename T>
ManagedDataCUDA<T>::~ManagedDataCUDA()
{
	if(this->storageType == DEVICE) {
		CUDA( cudaFree(this->data) );
	}
};

/**
 * If the copyType is SHALLOW_COPY, everything needed is already done in the
 * parent's copyConstructor.
 *
 * If the copyType is DEEP_COPY or AS_TEMPLATE, allocate memory for the data
 * from the device.
 *
 * If the copyType is DEEP_COPY, copy the values from the \a model.
 */
template<typename T>
void ManagedDataCUDA<T>::copyConstructor(
		ManagedData<T>* model,
		StorageType storageType,
		CopyType copyType)
{
	if(copyType == DEEP_COPY || copyType == AS_TEMPLATE) {
		this->allocateData();

		if(copyType == DEEP_COPY) {
			this->copyData(model->getData());
		}
	}
}

/**
 * Clear the data (set all the values to 0).
 */
template<typename T>
void ManagedDataCUDA<T>::clearDataImpl() const {
	assert(this->data != NULL);
	assert(this->dataSize > 0);

	CUDA( cudaMemset(this->data, 0, this->dataSize*sizeof(T)) );
}

/**
 * Copy the values from the array \a data to this object's data array.
 */
template<typename T>
void ManagedDataCUDA<T>::copyDataImpl(const T* data) const {
	assert(data != NULL);
	assert(this->data != NULL);
	assert(this->dataSize > 0);

	CUDA( cudaMemcpy(this->data, data, this->dataSize*sizeof(T), cudaMemcpyDeviceToDevice) );
}

/**
 * Copy the data from the host to the device memory, and free the host memory.
 */
template<typename T>
void ManagedDataCUDA<T>::toDeviceImpl() {
	assert(this->data != NULL);
	assert(this->dataSize > 0);

	T* tmp = NULL;
	CUDA( cudaMalloc((void**) &tmp, this->dataSize*sizeof(T)) );

	TIMING_DEVICE_MEMCPY_START();
	CUDA( cudaMemcpy(tmp, this->data, this->dataSize*sizeof(T), cudaMemcpyHostToDevice) );
	TIMING_DEVICE_MEMCPY_STOP();

	this->freeData();
	this->setData(tmp);
}

/**
 * Copy the data from the device to the host, and free the memory from the
 * devide.
 */
template<typename T>
void ManagedDataCUDA<T>::toHostImpl() {
	assert(this->data != NULL);
	assert(this->dataSize > 0);

	T* tmp = NULL;
	tmp = new T[this->dataSize];

	TIMING_DEVICE_MEMCPY_START();
	CUDA( cudaMemcpy(tmp, this->data, this->dataSize*sizeof(T), cudaMemcpyDeviceToHost) );
	TIMING_DEVICE_MEMCPY_STOP();

	CUDA( cudaFree(this->data) );
	this->data = tmp;
}

/**
 * Allocate device memory for the data.
 */
template<typename T>
void ManagedDataCUDA<T>::allocateDataImpl() {
	assert(this->data == NULL);
	assert(this->dataSize > 0);

	CUDA( cudaMalloc((void**) &this->data, this->dataSize*sizeof(T)) );
}

/**
 * Free the allocated data from the device.
 */
template<typename T>
void ManagedDataCUDA<T>::freeDataImpl() {
	CUDA( cudaFree(this->data) );
}

#endif /* MANAGEDDATACUDA_H_ */
