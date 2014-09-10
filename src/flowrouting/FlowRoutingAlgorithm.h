/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowRoutingAlgorithm.h
 */

#ifndef FLOWROUTINGALGORITHM_H_
#define FLOWROUTINGALGORITHM_H_

#include "FlowRoutingCommon.h"
#include "AbstractAlgorithm.h"
#include "FlowDirClass.h"
#include "FlatDistClass.h"
#include "RasterDEM.h"

/**
 * \brief The algorithm that determines the flow directions to each cell.
 *
 * This class cannot be used directly, but must be subclassed, and an
 * implementation for the abstract function performFlowRouting() must be
 * provided.
 */
template<typename T, typename U>
class FlowRoutingAlgorithm : public AbstractAlgorithm {
public:
	U noDataValueDEM;

public:
	FlowRoutingAlgorithm() {};
	virtual ~FlowRoutingAlgorithm() {};

	void setNoDataValueDEM(U noDataValueDEM) {
		this->noDataValueDEM = noDataValueDEM;
	}

	void execute(
			FlowDirClass<T>*  flowDir,
			RasterDEM<U>*     demPitsFilled);

protected:
    /**
     * \brief Determine the flow directions to cells that have neighbours with
     * lower elevation values.
     */
	virtual void performFlowRouting(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem) = 0;

    /**
     * \brief Determine the flow directions to cells that are "flat", i.e.
     * that do not have neighbour with lower elevation. The function
     * performFlowRouting() must be called prior to this function.
     */
	virtual void performFlowRoutingOnFlatSurfaces(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem) = 0;
};

typedef FlowRoutingAlgorithm<FlowDirDataType, DemData> FlowRoutingAlgorithm_t;

template<typename T, typename U>
void FlowRoutingAlgorithm<T, U>::execute(
		FlowDirClass<T>*  flowDir,
		RasterDEM<U>*     demPitsFilled)
{
	LOG_TRACE("FLOW ROUTING");

    LOG_TRACE_ALGORITHM("Processing");

    FlatDist_t* flatDist = new FlatDist_t(*demPitsFilled, AS_TEMPLATE);

    performFlowRouting(flowDir, flatDist, demPitsFilled);

    demPitsFilled->cacheDataIfNeeded("pfSpill"); // TODO: Here is an example of stuff that might be written to disk unnecessarily

	// Handle flat surfaces
    LOG_TRACE_ALGORITHM("processing");

    performFlowRoutingOnFlatSurfaces(flowDir, flatDist, demPitsFilled);

    demPitsFilled     ->cacheDataIfNeeded("pfSpill"); // TODO: Here is an example of stuff that might be written to disk unnecessarily
    flowDir ->cacheDataIfNeeded("fdFlowDir");
    flatDist->cacheDataIfNeeded("fdFlatDist");

    delete flatDist;
}


#endif /* FLOWROUTINGALGORITHM_H_ */
