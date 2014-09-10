/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RealisticDemAlgorithm.h
 */

#ifndef REALISTICDEMALGORITHM_H_
#define REALISTICDEMALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "KernelUtilsFactory.h"
#include "RasterDEM.h"
#include "RandomFieldClass.h"
#include "Logging.h"

/**
 * \brief An algorithm to add a random field to the DEM and remove possible
 * negative values.
 */
template<typename T, typename U>
class RealisticDemAlgorithm : public AbstractAlgorithm {

public:
	RealisticDemAlgorithm();
	virtual ~RealisticDemAlgorithm() {};

    /**
     * \brief Add dem and randomField and store the result to demWithError.
     * Remove negative values after the addition.
     */
	void execute(RasterDEM<T>* dem,
                 RandomFieldClass<U>* randomField,
                 RasterDEM<T>* demWithError);
};

typedef RealisticDemAlgorithm<DemData, DemData> RealisticDemAlgorithm_t;

template<typename T, typename U>
RealisticDemAlgorithm<T, U>::RealisticDemAlgorithm()
{
}

template<typename T, typename U>
void RealisticDemAlgorithm<T, U>::execute(
		RasterDEM<T>* dem,
		RandomFieldClass<U>* randomField,
		RasterDEM<T>* demWithError)
{
	LOG_TRACE("REALISTIC DEM");

	RemoveNegativeElevationsAlgorithm<T>* removeNegativeElevation = KernelUtilsFactory::createRemoveNegativeElevationsAlgorithm<T>();
	AddSurfacesAlgorithm<T, U>*           addSurfacesAlgorithm    = KernelUtilsFactory::createAddSurfacesAlgorithm<T, U>();

	removeNegativeElevation->setNoDataValue((T) G::noDataValueDEM);
	addSurfacesAlgorithm   ->setNoDataValue((T) G::noDataValueDEM);

    LOG_TRACE_ALGORITHM("Processing");

    addSurfacesAlgorithm   ->execute(dem, randomField, demWithError);
    removeNegativeElevation->execute(demWithError);

	delete removeNegativeElevation;
	delete addSurfacesAlgorithm;
}

#endif /* REALISTICDEMALGORITHM_H_ */
