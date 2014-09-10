/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file StreamBurningAlgorithm.h
 */

#ifndef STREAMBURNINGALGORITHM_H_
#define STREAMBURNINGALGORITHM_H_

#include <list>

#include "AbstractAlgorithm.h"
#include "StreamBurningCommon.h"
#include "RasterDEM.h"
#include "StreamData.h"
#include <cmath>

/**
 * \brief Abstract class to represent an algorithm which burns the stream
 * information to the elevation data.
 */
template<typename T, typename U>
class StreamBurningAlgorithm : public AbstractAlgorithm {
protected:
    /**
     * \brief Translation vector to closest neighbour cells.
     */
	ManagedData<CoordOffset>* circularSearchArray;
    /**
     * \brief The radius of the beveling of the stream edges.
     */
	int bevelRadius;
    /**
     * \brief The value used to inform that the cell does not have elevation
     * data.
     */
	T   noDataValueDEM;
    /**
     * \brief The value used to inform that the cell does not have stream
     * value.
     */
	T   noDataValueStream;

public:
    /**
     * \brief The constructor.
     */
	StreamBurningAlgorithm();
    /**
     * \brief The destructor that deletes the circularSearchArray.
     */
	virtual ~StreamBurningAlgorithm() { delete circularSearchArray; };

    /**
     * \brief Execute the algorithm.
     */
	void execute(
			RasterDEM<T>*  demWithError,
			StreamData<U>* streams);

    /**
     * \brief Burn the stream information from \a streams to \a dem using the
     * \a output as the output data.
     *
     * This function must be implemented when subclassing.
     */
	virtual void performStreamBurning(
			CellGrid<T>* output,
			CellGrid<T>* dem,
			CellGrid<U>* streams) = 0;

    /**
     * \brief Set the bevel radius.
     */
	void setBevelRadius(int bevelRadius) {
		this->bevelRadius = bevelRadius;
	}

    /**
     * \brief Set the value for the \a noDataValueDEM.
     */
	void setNoDataValueDEM(T noDataValue) {
		this->noDataValueDEM = noDataValue;
	}

    /**
     * \brief Set the value for the \a noDataValueStream.
     */
	void setNoDataValueStream(T noDataValueStream) {
		this->noDataValueStream = noDataValueStream;
	}

protected:
	void generateCircularSearchArray(
			CoordOffset* circularSearchArray,
			int bevelRadius);
};

typedef StreamBurningAlgorithm<DemData, DemData> StreamBurningAlgorithm_t;

template<typename T, typename U>
StreamBurningAlgorithm<T, U>::StreamBurningAlgorithm()
{
	circularSearchArray = NULL;
	bevelRadius         = 2;
	noDataValueDEM      = 0;
	noDataValueStream   = 0;
}

template<typename T, typename U>
void StreamBurningAlgorithm<T, U>::execute(
			RasterDEM<T>*  demWithError,
			StreamData<U>* streams)
{
	LOG_TRACE("STREAM BURNING");

	if(circularSearchArray == NULL) {
		LOG_TRACE_ALGORITHM("Generating circular search array");

		circularSearchArray = ManagedDataFactory::create<CoordOffset>(MAX_CIRCULAR_ARRAY_SIZE, HOST);
		circularSearchArray->allocateData();
		generateCircularSearchArray(circularSearchArray->getData(), bevelRadius);
	}

    LOG_TRACE_ALGORITHM("Processing");

    performStreamBurning(demWithError, demWithError, streams);
}

/**
 * \brief Generate a list of translation vectors that are shorter than
 * \a bevelRadius. The list is ordered from shortest vector to longest.
 */
template<typename T, typename U>
void StreamBurningAlgorithm<T, U>::generateCircularSearchArray(
		CoordOffset* circularSearchArray,
		int bevelRadius)
{
	int radius2 = bevelRadius*bevelRadius;

	std::list<CoordDist> l;
	for (int y = -bevelRadius; y <= bevelRadius; y++) {
		int yy = y * y;
		for (int x = -bevelRadius; x <= bevelRadius; x++) {
			if (y == 0 && x == 0)
				continue;

			int distance2 = x * x + yy;
			if (distance2 <= radius2) {
				CoordDist s = {distance2, {(short)x, (short)y}};
				l.push_back(s);
			}
		}
	}
	l.sort(coordDistCompare);

    int i = 0;
    for (std::list<CoordDist>::const_iterator it = l.begin(); it != l.end(); ++it, i++) {
        circularSearchArray[i] = it->p;
    }
}

#endif /* STREAMBURNINGALGORITHM_H_ */
