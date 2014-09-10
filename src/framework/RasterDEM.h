/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RasterDEM.h
 */

#ifndef RASTERDEM_H_
#define RASTERDEM_H_

#include "CellGrid.h"
#include "DemData.h"

/**
 * \brief A class to represent the terrain elevation values.
 */
template<class T>
class RasterDEM : public CellGrid<T> {
public:
	RasterDEM() {};
	RasterDEM(
			CellGrid<T>& model,
			CopyType copyType)
	: CellGrid<T>(model, copyType) {};

	RasterDEM(
			CellGrid<T>& model,
			StorageType storageType,
			CopyType copyType)
	: CellGrid<T>(model, storageType, copyType) {};

	RasterDEM(
			int width,
			int height,
			StorageType ptrType)
	 : CellGrid<T>(width, height, ptrType) {};

	RasterDEM(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			int width,
			int height)
	: CellGrid<T>(fileName, filePath, fileExtension, width, height) {};

	RasterDEM(
			const std::string& fileName,
			const std::string& filePath,
			const std::string& fileExtension,
			int width,
			int height,
			StorageType ptrType)
	: CellGrid<T>(fileName, filePath, fileExtension, width, height, ptrType) {};

	virtual ~RasterDEM() {};
};

typedef RasterDEM<DemData> RasterDEM_t;

/*
 * Definitions
 */

#endif /* RASTERDEM_H_ */
