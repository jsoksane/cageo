/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file GlobalParameters.cpp
 */

#include "GlobalParameters.h"

namespace G {

Framework framework = CUDA;

OutputType   outputType   = OUTPUT_TYPE_JPEG;
OutputOption outputOption = OUTPUT_OPTION_NONE;
std::string  outputName = "result";

double diskCacheThreshold = 0.9;
double hostCacheThreshold = 0.9;

int monteCarloIterations = 10;
int iterMC = 0;

std::string demName("dem10");
std::string demPath("DEM");

std::string streamsName("streams");
std::string streamsPath("DEM");

std::string defaultExtension(".bin");
std::string tempPath;

int inputWidth  = 0;
int inputHeight = 0;

double noDataValueDEM = 0.0;
double noDataValueStream = 0.0;

FilterModel filterModel = GAUSSIAN;

int bevelRadius = 2;

float practicalRange = 60;
float cellSize = 10;

float errorModelMean = 0.0f;
float errorModelStandardDeviation = 1.0f;

}
