/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowTracingAlgorithm.h
 */

#ifndef FLOWTRACINGALGORITHM_H_
#define FLOWTRACINGALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "FlowTraceDataClass.h"
#include "FlowDirClass.h"
#include "StreamData.h"

/**
 * \brief The algorithm that tracks the flow directions and connects each cell
 * to either a stream or at the edge of the DEM.
 *
 * This class cannot be used directly, but must be subclassed, and an
 * implementation for the abstract function performFlowTracing() must be
 * provided.
 */
template<typename T, typename U, typename V>
class FlowTracingAlgorithm : public AbstractAlgorithm {
public:
	FlowTracingAlgorithm() {};
	virtual ~FlowTracingAlgorithm() {};

	void execute(
			FlowTraceDataClass<T>* flowTraceData,
			FlowDirClass<U>*       flowDir,
			StreamData<V>*         streams);

protected:
	virtual void performFlowTracing(
			FlowTraceDataClass<T>* flowTrace,
			FlowDirClass<U>*       flowDir,
			StreamData<V>*         streams) = 0;

	virtual void performConnectFlowPath(
			FlowTraceDataClass<T>* flowTrace) = 0;
};

typedef FlowTracingAlgorithm<FlowTraceDataType, FlowDirDataType, DemData> FlowTracingAlgorithm_t;

template<typename T, typename U, typename V>
void FlowTracingAlgorithm<T, U, V>::execute(
		FlowTraceDataClass<T>* flowTraceData,
		FlowDirClass<U>*       flowDir,
		StreamData<V>*         streams)
{
	LOG_TRACE("FLOW TRACING");

    LOG_TRACE_ALGORITHM("Processing");

    performFlowTracing(flowTraceData, flowDir, streams);

    streams->cacheDataIfNeeded("streams");

	// Flow tracing (rest)
    performConnectFlowPath(flowTraceData);
    flowTraceData->cacheDataIfNeeded("ftFlowTrace");
}

#endif /* FLOWTRACINGALGORITHM_H_ */
