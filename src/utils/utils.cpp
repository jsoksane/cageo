/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 * \file utils.cpp
 */
#include "Utils.h"
#include <stdlib.h>
#include <fstream>
#include <string.h>
#include <errno.h>

#include "Timer.h"

//Align a to nearest higher multiple of b
int alignUp(int a, int b) {
    return (a % b != 0) ?  (a - a % b + b) : a;
}

int isPowerOfTwo(unsigned int x)
{
	while (((x % 2) == 0) && x > 1) /* While x is even and > 1 */
		x /= 2;
	return (x == 1);
}

void openFileForReading(ifstream* file, const string& fullFilePath)
{
	file->open(fullFilePath.c_str(), ios_base::in|ios_base::binary|ios_base::ate);

	if(!file->is_open()) {
		printf("Unable to open file \"%s\" for reading", fullFilePath.c_str());
		exit(1);
	}
}

void printProgress(const char* message, const int i, const int partsPerRank, const int x, const int y) {
#ifdef PRINT_PROGRESS
	printf("%s %.2f%% (%d,%d)\n", message, 100.0f*((float) (i+1)/partsPerRank), x, y);
#endif
}

void writeToFileBinary(void* h_data, int dataSize, string fileName) {
	ofstream file(fileName.c_str(), ios::binary);
	if(!file.is_open()) {
		cerr << "Unable to open file \"" << fileName << "\"" << endl;
		exit(1);
	}

	file.write((char*)h_data, dataSize);
	file.close();
}

void loadHeaderData(string fileName, DemInfo* demInfo)
{
	ifstream file(fileName.c_str());

	if(!file.is_open())
	{
		cerr << "Unable to open file \"" << fileName << "\"" << endl;
		exit(1);
	}

	string tag;

	while(true) {
		file >> tag;

		if(file.eof())
			break;

		if(tag.compare("ncols") == 0) {
			file >> demInfo->fileCols;
		}
		else if(tag.compare("nrows") == 0) {
			file >> demInfo->fileRows;
		}
		else if(tag.compare("xllcorner") == 0)
			file >> demInfo->xllCorner;
		else if(tag.compare("yllcorner") == 0)
			file >> demInfo->yllCorner;
		else if(tag.compare("cellsize") == 0)
			file >> demInfo->cellSize;
		else if(tag.compare("NODATA_value") == 0)
			file >> demInfo->noDataValue;
		else if(tag.compare("byteorder") == 0 || tag.compare("BYTEORDER") == 0) {
			file >> tag;
			if(tag.compare("MSBFIRST") == 0)
				demInfo->byteOrder = MSBFIRST;
			else if(tag.compare("LSBFIRST") == 0)
				demInfo->byteOrder = LSBFIRST;
			else if(tag.compare("MSBFIRST") == 0)
				demInfo->byteOrder = VMS_FFLOAT;
			else {
				cout << "Unrecognized byte order value: " << tag << endl;
				file.close();
				exit(1);
			}
		}
		else {
			cout << "Unrecognized tag value: " << tag << endl;
			file.close();
			exit(1);
		}
	}

	file.close();
}
