/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file GlobalParameters.h
 */

#ifndef GLOBALPARAMETERS_H_
#define GLOBALPARAMETERS_H_

#include <string>
#include "ConvolutionCommon.h"

/**
 * \brief The namespace for global values.
 */
namespace G {

/**
 * \brief An enumeration of different frameworks available.
 */
enum Framework {
	CUDA
};

/**
 * \brief An enumeration for different types of output.
 */
enum OutputType {
	OUTPUT_TYPE_BINARY, /*!< output in binary format . */
	OUTPUT_TYPE_JPEG /*!< output in jpg format. */
};

/**
 * \brief An enumeration to describe how often the output is saved.
 */
enum OutputOption {
	OUTPUT_OPTION_NONE, /*!< does not save any results. */
	OUTPUT_OPTION_END, /*!< saves the final result. */
	OUTPUT_OPTION_FRAMES /*!< saves first frame as such, second frame
				  averaged with the first one, third averaged
				  with the first and second and so on. */

};

/**
 * \brief A variable telling which OutputType is selected.
 */
extern OutputType   outputType;
/**
 * \brief A variable telling which OutputOption is selected.
 */
extern OutputOption outputOption;
/**
 * \brief The name of the output file.
 */
extern std::string  outputName;

/**
 * \brief A variable telling which Framework is selected.
 */
extern Framework framework;

/**
 * \brief If the host memory usage exceeds this value, the ManagedData objects
 * may start to store the data to disk.
 */
extern double diskCacheThreshold;
/**
 * \brief If the device memory usage exceeds this value, the ManagedData
 * objects may start to transfer to data to the host memory.
 */
extern double hostCacheThreshold;

/**
 * \brief How many Monte Carlo iterations to perform.
 */
extern int monteCarloIterations;
/**
 * \brief The number of Monte Carlo iterations performed.
 */
extern int iterMC;

/**
 * \brief The name of the DEM file.
 */
extern std::string demName;
/**
 * \brief The path of the DEM file.
 */
extern std::string demPath;

/**
 * \brief The name of the stream data file.
 */
extern std::string streamsName;
/**
 * \brief The path of the stream data file.
 */
extern std::string streamsPath;

/**
 * \brief The default extension of the cache files.
 */
extern std::string defaultExtension;
/**
 * \brief The path to store cache files.
 */
extern std::string tempPath;

/**
 * \brief The width of the global data.
 */
extern int inputWidth;
/**
 * \brief The height of the global data.
 */
extern int inputHeight;

/**
 * \brief The value used to represent that a cell does not have DEM value.
 */
extern double noDataValueDEM;
/**
 * \brief The value used to represent that a cell in stream data grid is not in
 * a stream.
 */
extern double noDataValueStream;

/**
 * \brief A variable telling which FilterModel is selected.
 */
extern FilterModel filterModel;

/**
 * \brief The radius of the beveling of the stream edges after burning.
 */
extern int bevelRadius;

/**
 * \brief The practical range for the convolution filter.
 */
extern float practicalRange;
/**
 * \brief The size of the cell in the input data.
 */
extern float cellSize;

/**
 * \brief The mean value of the random number distribution.
 */
extern float errorModelMean;
/**
 * \brief The standard deviation of the random number distribution.
 */
extern float errorModelStandardDeviation;

}


#endif /* GLOBALPARAMETERS_H_ */
