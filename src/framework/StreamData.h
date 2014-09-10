/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file StreamData.h
 */

#ifndef STREAMDATA_H_
#define STREAMDATA_H_

#include "CellGrid.h"
#include "DemData.h"

/**
 * \brief A grid that contains the id number of the streams.
 */
template<class T>
class StreamData : public CellGrid<T> {
public:
	StreamData();
	StreamData(
			CellGrid<T>& model,
			CopyType copyType);
	template<typename U>
	StreamData(
			CellGrid<U>& model,
			CopyType copyType);
	StreamData(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			int width,
			int height,
            StorageType storageType);
	virtual ~StreamData();
};

typedef StreamData<DemData> StreamData_t;

/*
 * Definitions
 */

template<class T>
StreamData<T>::StreamData() : CellGrid<T>()
{
}

template<class T>
StreamData<T>::StreamData(
		CellGrid<T>& model,
		CopyType copyType)
		: CellGrid<T>(model, copyType)
{
}

template<typename T>
template<typename U>
StreamData<T>::StreamData(
		CellGrid<U>& model,
		CopyType copyType)
		: CellGrid<T>(model, copyType)
{
}

template<class T>
StreamData<T>::StreamData(
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension,
		int width,
		int height,
        StorageType storageType)
: CellGrid<T>(fileName, filePath, fileExtension, width, height, storageType)
{

}

template<typename T>
StreamData<T>::~StreamData()
{
}

#endif /* STREAMDATA_H_ */
