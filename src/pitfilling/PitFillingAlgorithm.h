/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file PitFillingAlgorithm.h
 */

#ifndef PITFILLINGALGORITHM_H_
#define PITFILLINGALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "RasterDEM.h"
#include "ProcLaterClass.h"

/**
 * \brief A class to represent the algorithm that elevates the depressions on the 2D elevation data with burned streams so that the flow from every point eventually ends up to either a stream or at the edge of the 2D DEM data.
 *
 * This class cannot be instantiated directly. Instead one must subclass it and
 * implement the \a performPitFilling function.
 */
template<typename T>
class PitFillingAlgorithm : public AbstractAlgorithm {
public:
	/**
	 * \brief A counter that counts how many times the algorithm has been
	 * iterated.
	 */
	int iter;
	/**
	 * \brief A value that is used to represents the elevation for cells
	 * that are located in streams or at the border of the DEM data.
	 */
	T noDataValueDEM;

public:
	PitFillingAlgorithm();
	virtual ~PitFillingAlgorithm() {};

	/**
	 * \brief Execute the algorithm with given DEM data with burned streams.
	 *
	 * Algorithm produces the elevated DEM data so that the water flow from
	 * every point leads either to a stream or to the edge of the border.
	 *
	 * In the end this function deletes the objects pointed by the pointers
	 * in demPitsFilledList.
	 *
	 * \attention The demPitsFilledList should be a list of null pointers
	 * because the algorithm creates new objects for every pointer without
	 * deleting any previously created objects.
	 *
	 * @param demPitsFilledList The list of SpillClass_t pointers
	 * @param demWithStreamsList The list of RasterDEM_t pointers pointing
	 *                           to RasterDEM_t objects with burned stream
	 *                           data
	 */
	void execute(
			RasterDEM<T>* demPitsFilled,
			RasterDEM<T>* demWithStreams);

	/**
	 * \brief Set the value that is used as noDataValueDEM.
	 */
	void setNoDataValueDem(T noDataValueDEM) {
		this->noDataValueDEM = noDataValueDEM;
	}

protected:
	virtual void performPitFilling(
			RasterDEM<T>* spill,
			RasterDEM<T>* dem) = 0;
};

typedef PitFillingAlgorithm<DemData> PitFillingAlgorithm_t;

template<typename T>
PitFillingAlgorithm<T>::PitFillingAlgorithm()
{
	iter = 0;
	noDataValueDEM = 0;
}

template<typename T>
void PitFillingAlgorithm<T>::execute(
		RasterDEM<T>* demPitsFilled,
		RasterDEM<T>* demWithStreams)
{
	LOG_TRACE("PIT FILLING");

    performPitFilling(demPitsFilled, demWithStreams);
    
    demPitsFilled->cacheDataIfNeeded("pfSpill", NOT_MODIFIED);
}


#endif /* PITFILLINGALGORITHM_H_ */
