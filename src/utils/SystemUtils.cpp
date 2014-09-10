/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file SystemUtils.cpp
 */

#include <sigar.h>
#include <assert.h>
//#include <unistd.h>
#include <sstream>

#include "SystemUtils.h"

namespace SystemUtils {

//size_t getUsedMemory() {
//	size_t totalPages = sysconf(_SC_PHYS_PAGES);
//	size_t pageSize   = sysconf(_SC_PAGE_SIZE);
//	size_t freePages  = sysconf(_SC_AVPHYS_PAGES);
//
//	size_t totalMemoryAvailable = totalPages*pageSize;
//	size_t totalMemoryFree = freePages*pageSize;
//
//	return totalMemoryAvailable - totalMemoryFree;
//}
//
//size_t getTotalMemory() {
//	size_t totalPages = sysconf(_SC_PHYS_PAGES);
//	size_t pageSize   = sysconf(_SC_PAGE_SIZE);
//
//	return totalPages*pageSize;
//}

size_t getUsedMemory() {
	sigar_t* sigar = NULL;
	assert(SIGAR_OK == sigar_open(&sigar));

	sigar_mem_t mem;
	sigar_mem_get(sigar, &mem);

	sigar_close(sigar);

	return (size_t) mem.actual_used;
}

size_t getTotalMemory() {
	sigar_t* sigar = NULL;
	assert(SIGAR_OK == sigar_open(&sigar));

	sigar_mem_t mem;
	sigar_mem_get(sigar, &mem);

	sigar_close(sigar);

	return (size_t) mem.total;
}

void constructFullFilePath(
		std::string* fullFilePath,
		const std::string& fileName,
		const std::string& filePath,
		const std::string& fileExtension)
{
	std::stringstream ss;
	if(filePath.length() > 0) {
		ss << filePath << "/";
	}
	ss << fileName;

	ss << fileExtension;

	*fullFilePath = ss.str();
}

void openFileForReading(
		std::ifstream* file,
		const std::string& fullFilePath)
{
	file->open(fullFilePath.c_str(), std::ios_base::in|std::ios_base::binary|std::ios_base::ate);

	if(!file->is_open()) {
		printf("Unable to open file \"%s\" for reading", fullFilePath.c_str());
		exit(1);
	}
}

// TODO: Change so that this function can create directories which don't exist
void openFileForWriting(
		std::fstream* file,
		const std::string& fullFilePath,
		TruncateInfo truncate)
{
	std::ios_base::openmode mode = std::ios::binary|std::ios::out|std::ios::in;
	if(truncate == TRUNCATE) {
		mode |= std::ios::trunc;
	}

	file->open(fullFilePath.c_str(), mode);
	if(!file->is_open()) { // File doesn't exist
		file->open(fullFilePath.c_str(), mode|std::ios::trunc); // Create and open file
		if(!file->is_open()) {
			printf("Unable to open file \"%s\" for writing", fullFilePath.c_str());
			exit(1);
		}
	}
}

}


