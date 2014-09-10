/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file SystemUtils.h
 */

#ifndef FILEIO_H_
#define FILEIO_H_

#include <stdlib.h>
#include <string>
#include <fstream>
#include <iostream>
#include <ios>

namespace SystemUtils {

enum TempFileInfo {
	TEMP_FILE,
	NO_TEMP_FILE
};

enum TruncateInfo {
	TRUNCATE,
	NO_TRUNCATE
};


size_t getUsedMemory();
size_t getTotalMemory();

void constructFullFilePath(
		std::string* fullFilePath,
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension);

void openFileForReading(
		std::ifstream* file,
		const std::string& fullFilePath);

void openFileForWriting(
		std::fstream* file,
		const std::string& fullFilePath,
		TruncateInfo truncate);

template<typename T>
void writeToFile(
		T* data,
		int dataSize,
		const std::string& fullFilePath)
{
	std::fstream file;
	openFileForWriting(&file, fullFilePath, TRUNCATE);

	file.seekp(0, std::ios::beg);
	file.write((char*) data, dataSize*sizeof(T));

	file.close();
}

/**
 * \brief Read the whole file into given buffer.
 */
template<typename T>
void readFromTempFile(
		T* data,
		const std::string& fullFilePath)
{
	std::ifstream file;
	openFileForReading(&file, fullFilePath);

	file.seekg(0, std::ios::end);
	size_t fileSize = file.tellg();
	file.seekg(0, std::ios::beg);

	file.read((char*) data, fileSize);

	file.close();
}

/**
 * \brief Read from the data file the part (block) of data that is defined by
 * the parameters.
 *
 * @param data The (allocated) array of data.
 * @param fullFilePath The absolute filename.
 */
template<typename T>
void readFromSharedFile(
		T* data,
		int width,
		int height,
		const std::string& fullFilePath)
{
	std::ifstream file;
	openFileForReading(&file, fullFilePath);

	for(int row = 0; row < height; row++) {
		file.seekg(((int64_t) row * width) * sizeof(T), std::ios::beg);
		file.read((char*) (data + row * width), width * sizeof(T));
	}

	file.close();
}

}


#endif /* FILEIO_H_ */
