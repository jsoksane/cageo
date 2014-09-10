/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file CellGrid.h
 */

#ifndef CELLGRID_H_
#define CELLGRID_H_

#include <stdexcept>

#include <string.h>
#include <string>
#include <assert.h>

#include "ManagedData.h"
#include "ManagedDataFactory.h"
#include "GlobalParameters.h"
#include "SystemUtils.h"
#include "Logging.h"

#include <iostream>

enum BoundaryType {
	BOUNDARY_INNER,
	BOUNDARY_OUTER
};

enum Direction {
	TOP_LEFT = 0,
	TOP,
	TOP_RIGHT,
	LEFT,
	RIGHT,
	BOTTOM_LEFT,
	BOTTOM,
	BOTTOM_RIGHT
};

/**
 * \brief This class provides a dimensions for the array of values of type \a T.
 */
template<typename T>
class CellGrid {

protected:
    /**
     * \brief The raw data object.
     */
	ManagedData<T>* data;

public:
    /**
     * \brief The width of the data.
     */
	int width;
	/**
     * \brief The height of the data.
     */
	int height;
	/**
	 * \brief The number of points in the data grid.
	 */
	int size;

public:
    /**
     * \brief Construct an empty CellGrid.
     */
	CellGrid();

    /**
     * \brief Construct a new CellGrid by copying from another CellGrid
     * object \a model of the same type.
     */
	CellGrid(
			CellGrid<T>& model,
			CopyType copyType);

    /**
     * \brief Construct a new CellGrid by copying from another CellGrid
     * object \a model of different type.
     */
	template<typename U>
	CellGrid(
			CellGrid<U>& model,
			CopyType copyType);

	CellGrid(
			CellGrid<T>& model,
			StorageType storageType,
			CopyType copyType);

	template<typename U>
	CellGrid(
			CellGrid<U>& model,
			StorageType storageType,
			CopyType copyType);

	CellGrid(
			int width,
			int height,
			StorageType storageType);

	CellGrid(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			int width,
			int height);

	CellGrid(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			int width,
			int height,
			StorageType storageType);

	virtual ~CellGrid();

    /**
     * \brief Return the pointer to the raw data.
     */
	T* getData() const;
    /**
     * \brief Return the pointer to the ManagedData object.
     */
	ManagedData<T>* getDataObject();

    /**
     * \brief Set the ManagedData object's data pointer to point to \a data.
     */
	void setData(T* data);

    /**
     * \brief If the memory is allocated for the data, set all the values to
     * 0.
     */
	void clearData();
    /**
     * \brief If the memory is allocated for the data, copy the values from
     * \a data array to ManagedData.
     */
	void copyData(T* data);

    /**
     * \brief Ask the ManagedData to transfer the data to DEVICE.
     */
	void toDevice();
    /**
     * \brief Ask the ManagedData to transfer the data to HOST.
     */
	void toHost();

    /**
     * \brief Ask the ManagedData to transfer the data to DISK.
     */
	void toDisk(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if host memory
     * is getting full.
     */
	void toDiskIfNeeded(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);

    /**
     * \brief Ask the ManagedData to transfer the data to HOST if DEVICE
     * memory is getting full.
     */
	void toHostIfNeeded();

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if the HOST
     * and/or DEVICE is getting too full.
     */
	void cacheDataIfNeeded(
			const std::string& fileName,
			ModificationInfo modificationInfo);

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if the HOST
     * and/or DEVICE is getting too full.
     */
	void cacheDataIfNeeded(
			const std::string& fileName,
			const std::string& fileExtension,
			ModificationInfo modificationInfo);

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if the HOST
     * and/or DEVICE is getting too full.
     */
	void cacheDataIfNeeded(
			const std::string& fileName);

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if the HOST
     * and/or DEVICE is getting too full.
     */
	void cacheDataIfNeeded(
			const std::string& fileName,
			const std::string& fileExtension);

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if the HOST
     * and/or DEVICE is getting too full.
     */
	void cacheDataIfNeeded(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);

    /**
     * \brief Ask the ManagedData to transfer the data to DISK if the HOST
     * and/or DEVICE is getting too full.
     */
	void cacheDataIfNeeded(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			ModificationInfo modificationInfo);

    /**
     * \brief Ask the ManagedData to free data.
     */
	void freeData();

protected:
    /**
     * \brief Set the data pointer to 0.
     */
	void setDefaultValues();

    /**
     * \brief Initialize the object with given geometry.
     */
	void constructorInitDefault(
			int width,
			int height);

    /**
     * \brief Helper function to construct a new CellGrid by copying.
     */
	void constructorCopy(
			CellGrid<T>& model,
			StorageType storageType,
			CopyType copyType);

    /**
     * \brief Helper function to construct a new CellGrid by using the given
     * CellGrid as template.
     */
	template<typename U>
	void constructorAsTemplate(
			CellGrid<U>& model,
			StorageType storageType,
			CopyType copyType);

    /**
     * \brief Helper function to construct a new CellGrid by reading data from
     * file.
     */
	void constructorDataFromFile(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			int width,
			int height,
			StorageType storageType);

    /**
     * \brief Ask ManagedData to free data.
     */
	void allocateData();

    /**
     * \brief Ask ManagedData if the data is needed to store to DISK.
     */
	bool determineIfStoreOnDiskNeeded();
    /**
     * \brief Ask ManagedData if the data is needed to transfer to the HOST.
     */
	bool determineIfStoreInHostNeeded();

    /**
     * \brief Ask ManagedData to read data from file.
     */
	void readDataFromDisk(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);
    /**
     * \brief Ask ManagedData to read data from file.
     */
	void readDataFromDisk();

    /*
	void readPartialDataFromDisk(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);
    */

    /**
     * \brief Ask ManagedData to write data to a file.
     */
	void writeDataToDisk(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);

    /**
     * \brief Ask ManagedData to write data to a file.
     */
	void writeDataToDisk(
			const std::string& fullFilePath);

public:
    /*
	void writePartialDataToDisk(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension);
    */
};

/*
 * Definitions
 */

/**
 * Construct an empty CellGrid object.
 */
template<typename T>
CellGrid<T>::CellGrid()
{
	setDefaultValues();
}

/**
 * Construct a cellGrid object with given arguments.
 */
template<typename T>
CellGrid<T>::CellGrid(
		int width,
		int height,
		StorageType storageType)
{
	constructorInitDefault(width, height);

	data = ManagedDataFactory::create<T>(size, storageType);
	data->allocateData();
}

/**
 * Create a new CellGrid to by copying it from the given CellGrid that is of
 * the same type. The new data will be placed in the same storage location as
 * the original.
 */
template<typename T>
CellGrid<T>::CellGrid(
		CellGrid<T>& model,
		CopyType copyType)
{
	constructorCopy(model, model.getDataObject()->getStorageType(), copyType);
}

/**
 * Create a new CellGrid to by copying it from the given CellGrid that is of
 * different type.
 */
template<typename T>
template<typename U>
CellGrid<T>::CellGrid(
		CellGrid<U>& model,
		CopyType copyType)
{
	constructorAsTemplate(model, model.getDataObject()->getStorageType(), copyType);
}

/**
 * Create a new CellGrid to by copying it from the given CellGrid that is of
 * the same type. The memory for the new data will be allocated from the
 * \a storageType.
 */
template<typename T>
CellGrid<T>::CellGrid(
		CellGrid<T>& model,
		StorageType storageType,
		CopyType copyType)
{
	constructorCopy(model, storageType, copyType);
}

/**
 * Create a new CellGrid to by copying it from the given CellGrid that is of
 * different type. The new data is located in \a storageType.
 */
template<typename T>
template<typename U>
CellGrid<T>::CellGrid(
		CellGrid<U>& model,
		StorageType storageType,
		CopyType copyType)
{
	constructorAsTemplate(model, storageType, copyType);
}

/**
 * Construct a new CellGrid by reading data from the file. The memory for the
 * data is allocated from HOST.
 */
template<typename T>
CellGrid<T>::CellGrid(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension,
		int width,
		int height)
{
	constructorDataFromFile(fileName, filePath, fileExtension, width, height,
			HOST);
}

/**
 * Construct a new CellGrid by reading data from the file. The memory for the
 * data is allocated from \a storageType.
 */
template<typename T>
CellGrid<T>::CellGrid(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension,
		int width,
		int height,
		StorageType storageType)
{
	constructorDataFromFile(fileName, filePath, fileExtension, width, height,
			storageType);
}

/**
 * A helper function to construct CellGrid by reading data from file.
 *
 * First a ManagedData class is created whose storageType is HOST, and the
 * file contents are read into HOST memory. After that, if the \a storageType
 * is DEVICE, the data is transferred to DEVICE.
 */
template<typename T>
void CellGrid<T>::constructorDataFromFile(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension,
		int width,
		int height,
		StorageType storageType)
{
	constructorInitDefault(width, height);

	data = ManagedDataFactory::create<T>(size, HOST);
	data->allocateData();

    readDataFromDisk(fileName, filePath, fileExtension);

	if(storageType == DEVICE) {
		toDevice();
	}
}


template<typename T>
void CellGrid<T>::setDefaultValues()
{
	data = NULL;
}

template<typename T>
void CellGrid<T>::constructorInitDefault(
		int width,
		int height)
{
	setDefaultValues();

	this->width  = width;
	this->height = height;
    this->size = width * height;
}

/**
 * Set this object's values by copying them from the given CellGrid object.
 */
template<typename T>
void CellGrid<T>::constructorCopy(
		CellGrid<T>& model,
		StorageType storageType,
		CopyType copyType)
{
	constructorInitDefault(model.width, model.height);
	data = ManagedDataFactory::create<T>(model.getDataObject(), storageType, copyType);
}

/**
 * Set this object's values by using the given CellGrid object as a template.
 *
 * This object will get the same dimension as the \a model, but the memory is
 * allocated from \a storageType.
 */
template<typename T>
template<typename U>
void CellGrid<T>::constructorAsTemplate(
		CellGrid<U>& model,
		StorageType storageType,
		CopyType copyType) //FIXME useless argument?
{
	// If 'model' is different type than 'this' (eg. T != U), then we can only allow copyType of AS_TEMPLATE
	assert(copyType == AS_TEMPLATE);

	constructorInitDefault(model.width, model.height);

	data = ManagedDataFactory::create<T>(size, storageType);
	data->allocateData();
}

template<typename T>
CellGrid<T>::~CellGrid()
{
	delete data;
    data = NULL;
}

template<typename T>
T* CellGrid<T>::getData() const
{
	return data->getData();
}

template<typename T>
ManagedData<T>* CellGrid<T>::getDataObject()
{
	assert(data != NULL);

	return data;
}

template<typename T>
void CellGrid<T>::setData(T* data)
{
	this->data->setData(data);
}

template<typename T>
void CellGrid<T>::clearData()
{
	assert(data != NULL);

	this->data->clearData();
}

template<typename T>
void CellGrid<T>::copyData(T* data)
{
	this->data->copyData(data);
}

template<typename T>
void CellGrid<T>::toDevice() {
	data->toDevice();
}

template<typename T>
void CellGrid<T>::toHost() {
	data->toHost();
}

template<typename T>
void CellGrid<T>::allocateData()
{
	data->allocateData();
}

template<typename T>
void CellGrid<T>::freeData()
{
	data->freeData();
}

template<typename T>
void CellGrid<T>::toDisk(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension)
{
	std::string fullFilePath;
	SystemUtils::constructFullFilePath(&fullFilePath, fileName, filePath, fileExtension);
	data->setFullFilePath(fullFilePath);
	data->toDisk();
}

template<typename T>
void CellGrid<T>::toDiskIfNeeded(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension)
{
	data->toDiskIfNeeded(fileName, filePath, fileExtension);
}

template<typename T>
void CellGrid<T>::toHostIfNeeded()
{
	data->toHostIfNeeded();
}

template<typename T>
void CellGrid<T>::cacheDataIfNeeded(
		const std::string& fileName)
{
	cacheDataIfNeeded(fileName, G::tempPath, G::defaultExtension);
}

template<typename T>
void CellGrid<T>::cacheDataIfNeeded(
		const std::string& fileName,
		ModificationInfo modificationInfo)
{
	cacheDataIfNeeded(fileName, G::tempPath, G::defaultExtension, modificationInfo);
}

template<typename T>
void CellGrid<T>::cacheDataIfNeeded(
		const std::string& fileName,
		const std::string& fileExtension)
{
	cacheDataIfNeeded(fileName, G::tempPath, fileExtension);
}

template<typename T>
void CellGrid<T>::cacheDataIfNeeded(
		const std::string& fileName,
		const std::string& fileExtension,
		ModificationInfo modificationInfo)
{
	cacheDataIfNeeded(fileName, G::tempPath, fileExtension, modificationInfo);
}

template<typename T>
void CellGrid<T>::cacheDataIfNeeded(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension)
{
	cacheDataIfNeeded(fileName, filePath, fileExtension, MODIFIED);
}

template<typename T>
void CellGrid<T>::cacheDataIfNeeded(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension,
		ModificationInfo modificationInfo)
{
	data->cacheDataIfNeeded(fileName, filePath, fileExtension, modificationInfo);
}

template<typename T>
bool CellGrid<T>::determineIfStoreOnDiskNeeded() {
	return data->determineIfStoreOnDiskNeeded();
}

template<typename T>
bool CellGrid<T>::determineIfStoreInHostNeeded() {
	return data->determineIfStoreInHostNeeded();
}

template<typename T>
void CellGrid<T>::readDataFromDisk(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension)
{
	data->readDataFromDisk(fileName, filePath, fileExtension);
}

template<typename T>
void CellGrid<T>::readDataFromDisk()
{
	data->readDataFromDisk();
}

#endif /* CELLGRID_H_ */
