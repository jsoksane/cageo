/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file CatchmentBorderAlgorithm.h
 */

#ifndef CATCHMENTBORDERALGORITHM_H_
#define CATCHMENTBORDERALGORITHM_H_

#include "AbstractAlgorithm.h"
#include "KernelUtilsFactory.h"
#include "CatchmentBorderClass.h"
#include "FlowTraceDataClass.h"

/**
 * \brief An algorithm to extract the drainage are borders from the given
 * FlowTraceDataClass instance.
 */
template<typename T, typename U>
class CatchmentBorderAlgorithm : public AbstractAlgorithm {
public:
	CatchmentBorderAlgorithm() {};
	virtual ~CatchmentBorderAlgorithm() {};

    /**
     * \brief Extract the drainage border from the \a flowTraceData and add it
     * to the \a * monteCarloBorder. This function calls the abstract
     * performExtractDrainageBorder() function, which must be implemented when
     * subclassing.
     */
	void execute(
			CatchmentBorderClass<T>* monteCarloBorder,
			FlowTraceDataClass<U>* flowTraceData);

protected:
    /**
     * \brief Implementation-specific function to extract the drainage
     * borders.
     */
	virtual void performExtractDrainageBorder(
			CatchmentBorderClass<T>* catchmentBorder,
			FlowTraceDataClass<U>* flowTrace) = 0;
};

typedef CatchmentBorderAlgorithm<CatchmentBorderType, FlowTraceDataType> CatchmentBorderAlgorithm_t;

template<typename T, typename U>
void CatchmentBorderAlgorithm<T, U>::execute(
		CatchmentBorderClass<T>* monteCarloBorder,
		FlowTraceDataClass<U>* flowTraceData)
{
	LOG_TRACE("BORDER EXTRACTION");

	AddSurfacesAlgorithm<T, T>* addSurfacesAlgorithm = KernelUtilsFactory::createAddSurfacesAlgorithm<T, T>();
	addSurfacesAlgorithm->setNoDataValue(-1);

    LOG_TRACE_ALGORITHM("Processing");

    CatchmentBorder_t catchmentBorder(*monteCarloBorder, AS_TEMPLATE);

    performExtractDrainageBorder(&catchmentBorder, flowTraceData);

    addSurfacesAlgorithm->execute(monteCarloBorder, &catchmentBorder, monteCarloBorder);

    monteCarloBorder->cacheDataIfNeeded("monteCarloBorder");

	delete addSurfacesAlgorithm;
}


#endif /* CATCHMENTBORDERALGORITHM_H_ */
