/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file ManagedData.h
 */

#ifndef MANAGEDDATA_H_
#define MANAGEDDATA_H_

#include <stdlib.h>
#include <assert.h>

#include "SystemUtils.h"
#include "CudaUtils.cuh"
#include "Logging.h"

/// Enumeration for different possible data locations.
enum StorageType {
    HOST,
    DEVICE,
    DISK
};

/**
 * \brief Different options for constructing a new ManagedData object by
 * copying.
 */
enum CopyType {
    AS_TEMPLATE, /*!< Allocate the same amount of values for the new object from the same \a StorageLocation where the original data is stored. */
    SHALLOW_COPY, /*!< The new object points to the same data as the original object. */
    DEEP_COPY /*!< Allocate memory for the new object, and copy the original data. */
};

/// Enumeration that tells whether the data is modified or not.
enum ModificationInfo {
    MODIFIED,
    NOT_MODIFIED
};

/**
 * \brief An abstract class to represent data that can be located in several
 * locations (on a disk, in RAM, in GPU...) and that can be transferred
 * between those locations as needed.
 *
 * This class needs to be subclassed, because the actual data handling is
 * dependent on what kinds of locations are used.
 */
template<typename T>
class ManagedData
{
protected:
    /**
     * \brief The number of data points.
     */
    size_t dataSize;
    /**
     * \brief The location of the data.
     */
    StorageType storageType;

    /**
     * \brief The filename (without the file extension) of the data file.
     */
    std::string fileName;
    /**
     * \brief The path to the directory where the data file is located.
     */
    std::string filePath;
    /**
     * \brief The file extension of the data file.
     */
    std::string fileExtension;
    /**
     * \brief The full filename with absolute path of the data file.
     */
    std::string fullFilePath;

    /**
     * The pointer to the raw data.
     */
    T* data;
public:
    /**
     * \brief Create an empty ManagedData object.
     */
    ManagedData();
    /**
     * \brief Create an empty ManagedData object that will contain \a dataSize
     * values.
     */
    ManagedData(
            size_t dataSize);
    /**
     * \brief Create an empty ManagedData object that will contain \a dataSize
     * values and is located in \a storageType.
     */
    ManagedData(
            size_t dataSize,
            StorageType storageType);
    /**
     * \brief Create a new ManagedData object by copying it from \a model.
     */
    ManagedData(
            ManagedData<T>* model,
            StorageType storageType,
            CopyType copyType);

    /**
     * \brief The destructor.
     */
    virtual ~ManagedData();

    /**
     * \brief Return the pointer to the raw data.
     */
    T* getData() const;
    /**
     * \brief Set the data pointer to point to \a data.
     */
    void setData(T* data);
    /**
     * \brief If the memory is allocated for the data, set all the values to
     * 0.
     */
    void clearData() const;
    /**
     * \brief Copy the values from \a data to this object.
     */
    void copyData(const T* data) const;
    /**
     * \brief Allocate memory for the data.
     */
    void allocateData();
    /**
     * \brief Free the allocated memory and set the data pointer to 0.
     */
    void freeData();

    /**
     * \brief Return the number of data values.
     */
    size_t getSize() { return dataSize; };
    /**
     * \brief Set the size of the data to \a dataSize.
     */
    void setSize(size_t dataSize) { this->dataSize = dataSize; }

    /**
     * \brief Get the location of the data.
     */
    StorageType getStorageType() const;

    /**
     * \brief Transfer the data to the device.
     */
    void toDevice();
    /**
     * \brief Transfer the data to the host.
     */
    void toHost();
    /**
     * \brief Transfer the data to the disk.
     */
    void toDisk(
            const std::string& fileName,
            const std::string& filePath,
            const std::string& fileExtension);

    /**
     * \brief Transfer the data to the disk, if the host memory usage is over
     * the predefined threshold.
     */
    void toDiskIfNeeded(
            const std::string& fileName,
            const std::string& filePath,
            const std::string& fileExtension);

    /**
     * \brief Transfer the data from the device to the host, if the device
     * memory usage is over the predefined threshold.
     */
    void toHostIfNeeded();

    /**
     * \brief Transfer the data to the disk if the device and/or host memory
     * usages are over the predefined thresholds.
     */
    void cacheDataIfNeeded(
            const std::string& fileName,
            const std::string& filePath,
            const std::string& fileExtension,
            ModificationInfo modificationInfo);

    /**
     * \brief Write data to disk.
     */
    void writeDataToDisk(
            const std::string& fileName,
            const std::string& filePath,
            const std::string& fileExtension);

    /**
     * \brief Write data to disk.
     */
    void writeDataToDisk(const std::string& fullFilePath);
    /**
     * \brief Read data from file.
     */
    void readDataFromDisk();
    /**
     * \brief Read data from file.
     */
    void readDataFromDisk(
            const std::string& fileName,
            const std::string& filePath,
            const std::string& fileExtension);
    /**
     * \brief Read data from file.
     */
    void readDataFromDisk(
            const std::string& fullFilePath);

protected:
    /**
     * \brief Set the data values to 0.
     *
     * This needs to be implemented in the subclass, because this operation is
     * dependent on the location of the data.
     */
    virtual void clearDataImpl() const = 0;
    /**
     * \brief Copy the values from the \a data to this object.
     *
     * This needs to be implemented in the subclass, because this operation is
     * dependent on the location of the data.
     */
    virtual void copyDataImpl(const T* data) const = 0;
    /**
     * \brief Transfer the data to the device.
     *
     * This needs to be implemented in the subclass, because this operation is
     * dependent on the location of the data.
     */
    virtual void toDeviceImpl() = 0;
    /**
     * \brief Transfer the data from the device to host.
     *
     * This needs to be implemented in the subclass, because this operation is
     * dependent on the location of the data.
     */
    virtual void toHostImpl() = 0;
    /**
     * \brief Allocate memory for the data from the device.
     *
     * This needs to be implemented in the subclass, because this operation is
     * dependent on the location of the data.
     */
    virtual void allocateDataImpl() = 0;
    /**
     * \brief Free the allocated memory from the device.
     *
     * This needs to be implemented in the subclass, because this operation is
     * dependent on the location of the data.
     */
    virtual void freeDataImpl() = 0;

private:
    /**
     * \brief Set the object as empty.
     */
    void setDefaultValues();
    /**
     * \brief A helper function for copy constructor.
     */
    void copyConstructor(
            ManagedData<T>* model,
            StorageType storageType,
            CopyType copyType);
    /**
     * \brief Free the memory used for the data (if any) and set the data
     * pointer to 0.
     */
    void destructor();

    /**
     * \brief Check whether we need to write data to disk due to host memory
     * being too full.
     */
    bool determineIfStoreOnDiskNeeded();
    /**
     * \brief Check whether the data needs to be transferred from device to
     * host due to device memory being too full.
     */
    bool determineIfStoreInHostNeeded();
};

/*
 * Definitions
 */

/**
 * Construct a ManagedData object with default values.
 */
template<typename T>
ManagedData<T>::ManagedData() {
    setDefaultValues();
}

/**
 * Construct a ManagedData object for a data of the given size. The
 * \a storateType is set to \a HOST.
 */
template<typename T>
ManagedData<T>::ManagedData(size_t dataSize) {
    setDefaultValues();
    this->dataSize = dataSize;
    this->storageType = HOST;
}

/**
 * Construct a ManagedData object for a data of the given size and with given
 * storage type.
 */
template<typename T>
ManagedData<T>::ManagedData(size_t dataSize, StorageType storageType) {
    setDefaultValues();
    this->dataSize = dataSize;
    this->storageType = storageType;
}

/**
 * Construct a ManagedData object by copying.
 */
template<typename T>
ManagedData<T>::ManagedData(
        ManagedData<T>* model,
        StorageType storageType,
        CopyType copyType)
{
    setDefaultValues();
    copyConstructor(model, storageType, copyType);
}

/**
 * Destructor.
 */
template<typename T>
ManagedData<T>::~ManagedData()
{
    destructor();
}

/**
 * Set the default values for the object. By default the data is 0, datasize
 * is 0 and storageType is \a HOST.
 */
template<typename T>
void ManagedData<T>::setDefaultValues() {
    data = 0;
    dataSize = 0;
    storageType = HOST;
}

/**
 * If the copyType is \a DEEP_COPY, transfer to model data to the storage
 * location of this object.
 *
 * The size and storage location of this object is set to the model's values.
 *
 * If the copyType is \a SHALLOW_COPY, set this object's data pointer to point
 * to the model's data. Otherwise, the copying is performed in the subclass.
 */
template<typename T>
void ManagedData<T>::copyConstructor(
        ManagedData<T>* model,
        StorageType storageType,
        CopyType copyType)
{
    assert(model != 0);

    // If we are doing a deep copy, we need to make sure that data for 'model' is stored in 'storageType'
    if(copyType == DEEP_COPY) {
        if(storageType == HOST || storageType == DISK) {
            model->toHost();
        }
        else if(storageType == DEVICE) {
            model->toDevice();
        }
    }

    this->dataSize = model->getSize();
    this->storageType = model->getStorageType();

    if(copyType == SHALLOW_COPY) {
        data = model->getData();
        assert(data != 0);
    }
    else if(copyType == DEEP_COPY) {
        // The allocation and copy will take place in the subclass' constructor
        // (since they are subclass dependent, and baseclass constructor can't
        // yet access the implemented pure virtual functions)
    }
    else if(copyType == AS_TEMPLATE) {
        // same as above
    }
}

/**
 * If the data is located on disk or on host, free it now and set the data
 * pointer to 0.
 *
 * If the data is located on the device, the subclass will handle the freeing
 * of the memory.
 */
template<typename T>
void ManagedData<T>::destructor()
{
    switch(storageType) {
    case DISK:   assert(data == 0); break;
    case HOST:   delete[] data;        break;
    case DEVICE: /* taken care of by subclass */ break;
    }

    data = 0;
}

template<typename T>
T* ManagedData<T>::getData() const
{
    return data;
}

template<typename T>
void ManagedData<T>::setData(T* data)
{
    this->data = data;
}

/**
 * If the data is located on the host, set the values to 0 now.
 *
 * If the data is located on the device, call clearDataImpl() to set the
 * values to zero.
 */
template<typename T>
void ManagedData<T>::clearData() const
{
    assert(data != 0);

    if(storageType == HOST) {
        memset(data, 0, dataSize*sizeof(T));
    }
    else if(storageType == DEVICE) {
        clearDataImpl();
    }
}

/**
 * If the storage location of this object HOST, copy the values from the
 * \a data array to this object's data.
 *
 * If the storage location is DEVICE, call copyDataImpl() to perform the
 * copying.
 */
template<typename T>
void ManagedData<T>::copyData(const T* data) const
{
    assert(this->data != 0);

    if(storageType == HOST) {
        memcpy(this->data, data, dataSize*sizeof(T));
    }
    else if(storageType == DEVICE) {
        TIMING_DEVICE_MEMCPY_START();
        copyDataImpl(data);
        TIMING_DEVICE_MEMCPY_STOP();
    }
}

/**
 * Return the current location of the data.
 */
template<typename T>
StorageType ManagedData<T>::getStorageType() const
{
    return storageType;
}

/**
 * If the data is already in the devide, do nothing. Otherwise, first fetch
 * the data from disk (if needed) to host, and then move it to the device.
 *
 * The \a storageType is updated to \a DEVICE.
 */
template<typename T>
void ManagedData<T>::toDevice()
{
    switch(storageType) {
    case DEVICE: return;
    case DISK:   toHost(); break;
    }

    toDeviceImpl();

    storageType = DEVICE;
}

/**
 * If the data is already on the host, do nothing. If the data is on the
 * device, move it from there to the host memory. If the data is located on
 * disk, allocate memory for the data and read it from the disk.
 *
 * The \a storageType is updated to \a HOST.
 */
template<typename T>
void ManagedData<T>::toHost()
{
    switch(storageType) {
    case HOST:   return;
    case DEVICE: toHostImpl(); break;
    case DISK:   allocateData();
                 readDataFromDisk(); break;
    }

    storageType = HOST;
}

/**
 * If the data is already on disk, do nothing. Otherwise, move it to host
 * memory from the device (if needed), and then write it to disk and free the
 * memory.
 *
 * The \a storageType is updated to \a DISK.
 */
template<typename T>
void ManagedData<T>::toDisk(
        const std::string& fileName,
        const std::string& filePath,
        const std::string& fileExtension)
{
    switch(storageType) {
    case DISK:   return;
    case DEVICE: toHost(); break;
    }

    writeDataToDisk(fileName, filePath, fileExtension);
    freeData();

    storageType = DISK;
}

/** Construct the absolute filename from the given arguments, and write data
 * to that file.
 */
template<typename T>
void ManagedData<T>::writeDataToDisk(
        const std::string& fileName,
        const std::string& filePath,
        const std::string& fileExtension)
{
    std::string fullFilePath;
    SystemUtils::constructFullFilePath(&fullFilePath, fileName, filePath, fileExtension);
    this->fullFilePath = fullFilePath;

    writeDataToDisk(fullFilePath);
}

/**
 * Write the data into a file with given absolute filename.
 */
template<typename T>
void ManagedData<T>::writeDataToDisk(
        const std::string& fullFilePath)
{
    SystemUtils::writeToFile(data, dataSize, fullFilePath);
}

/**
 * Read the data from a predefined file.
 */
template<typename T>
void ManagedData<T>::readDataFromDisk()
{
    readDataFromDisk(fullFilePath);
}

/**
 * Construct the filename from the given arguments, are read the data from
 * that file.
 */
template<typename T>
void ManagedData<T>::readDataFromDisk(
        const std::string& fileName,
        const std::string& filePath,
        const std::string& fileExtension)
{
    std::string fullFilePath;
    SystemUtils::constructFullFilePath(&fullFilePath, fileName, filePath, fileExtension);

    TIMING_IO_READ_START();
    SystemUtils::readFromTempFile(data, fullFilePath);
    TIMING_IO_READ_STOP();
}

/**
 * Read the data from the given absolute filename.
 */
template<typename T>
void ManagedData<T>::readDataFromDisk(
        const std::string& fullFilePath)
{
    assert(fullFilePath.length() > 0);

    TIMING_IO_READ_START();
    SystemUtils::readFromTempFile(data, fullFilePath);
    TIMING_IO_READ_STOP();
}

/**
 * If the storage type is HOST or DISK, space for the data is
 * allocated in RAM.
 *
 * If the storage type is DEVICE, call the allocateDataImpl() to do the
 * allocation.
 */
template<typename T>
void ManagedData<T>::allocateData()
{
    assert(data == 0);

    if(storageType == HOST || storageType == DISK) {
        data = new T[dataSize];
    }
    else if(storageType == DEVICE) {
        allocateDataImpl();
    }
}

/**
 * If the data is stored on disk, simply make sure that the data
 * pointer is zero.
 *
 * If the storage location is HOST, free the allocated memory and set the data
 * pointer to 0.
 *
 * If the storage location is DEVICE, call freeDataImpl() to free the data, and
 * then set the data pointer to 0.
 */
template<typename T>
void ManagedData<T>::freeData()
{
    switch(storageType) {
    case DISK:   assert(data == 0); break;
    case HOST:   delete[] data;        break;
    case DEVICE: freeDataImpl();       break;
    }

    data = 0;
}



/**
 * If the host memory is too full, the data is written to disk and the memory
 * is freed. The need is defined by the \a determineIfStoreOnDiskNeeded function.
 */
template<typename T>
void ManagedData<T>::toDiskIfNeeded(
        const std::string& fileName,
        const std::string& filePath,
        const std::string& fileExtension)
{
    if(storageType == DISK || storageType == DEVICE) {
        return;
    }
    else if(storageType == HOST) {
        bool storeOnDisk = determineIfStoreOnDiskNeeded();
        if(storeOnDisk) {
            toDisk(fileName, filePath, fileExtension);
        }
    }
}

/**
 * If the device memory is too full, the data is transferred to the host
 * memory, and the device memory is free. The need is decided by the
 * \a determineIfStoreInHostNeeded function.
 */
template<typename T>
void ManagedData<T>::toHostIfNeeded()
{
    if(storageType == DISK || storageType == HOST) {
        return;
    }
    else if(storageType == DEVICE) {
        bool storeInHost = determineIfStoreInHostNeeded();
        if(storeInHost) {
            toHost();
        }
    }
}

/**
 * If the data is not modified after reading/writing it to disk, free the host
 * memory. Otherwise, if the data is located on the device, transfer it to
 * host if \a toHostIfNeeded return true. Then, if \a toDiskIfNeeded return true,
 * write the data to disk and free the memory.
 */
template<typename T>
void ManagedData<T>::cacheDataIfNeeded(
        const std::string& fileName,
        const std::string& filePath,
        const std::string& fileExtension,
        ModificationInfo modificationInfo)
{
    if(modificationInfo == NOT_MODIFIED && this->fileName.length() > 0) { // This needs to be redesigned
        freeData();
        storageType = DISK;
        return;
    }

    switch(storageType) {
    case DISK:   return;
    case DEVICE: toHostIfNeeded(); toDiskIfNeeded(fileName, filePath, fileExtension); break;
    case HOST:                     toDiskIfNeeded(fileName, filePath, fileExtension); break;
    }
}

/**
 * Check the current status of the host memory. If it is used more than a
 * predefined percentage \a diskCacheThreshold, return true. Otherwise, return
 * false.
 */
template<typename T>
bool ManagedData<T>::determineIfStoreOnDiskNeeded() {
    size_t memoryUsed  = SystemUtils::getUsedMemory();
    size_t totalMemory = SystemUtils::getTotalMemory();
    size_t size        = dataSize*sizeof(T); // TODO: what if data is compressed?

    //TODO: why these are summed, if the data is already in the memory?
    if((memoryUsed + size)/(double) totalMemory > G::diskCacheThreshold) {
        return true;
    }
    else {
        return false;
    }
}

/**
 * If the device memory is used more than a predefined percentage
 * \a hostCacheThreshold, return true. Otherwise, return false.
 */
template<typename T>
bool ManagedData<T>::determineIfStoreInHostNeeded() {
    size_t memoryUsed  = CudaUtils::getUsedMemory();
    size_t totalMemory = CudaUtils::getTotalMemory();
    size_t size        = dataSize*sizeof(T); // TODO: what if data is compressed?

    double ratio = (memoryUsed + size)/(double) totalMemory;

    if(ratio > G::hostCacheThreshold) {
        return true;
    }
    else {
        return false;
    }
}

#endif /* MANAGEDDATA_H_ */
