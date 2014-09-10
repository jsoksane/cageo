/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file CatchmentBorderAlgorithmCPUOpenMP.h
 */

#ifndef CATCHMENTBORDERALGORITHMCPUOPENMP_H_
#define CATCHMENTBORDERALGORITHMCPUOPENMP_H_

#include "CatchmentBorderAlgorithm.h"
#include "FlowTracingCommon.h"

template<typename T, typename U>
class CatchmentBorderAlgorithm_CPU_OpenMP : public CatchmentBorderAlgorithm<T, U> {
public:
	CatchmentBorderAlgorithm_CPU_OpenMP() {};
	virtual ~CatchmentBorderAlgorithm_CPU_OpenMP() {};

	void performExtractDrainageBorder(
			CatchmentBorderClass<T>* catchmentBorder,
			FlowTraceDataClass<U>* flowTrace);
};

template<typename T, typename U>
void CatchmentBorderAlgorithm_CPU_OpenMP<T, U>::performExtractDrainageBorder(
		CatchmentBorderClass<T>* catchmentBorder,
		FlowTraceDataClass<U>* flowTrace)
{
	flowTrace->toHost();
	catchmentBorder->toHost();

	T* catchmentBorderData = catchmentBorder->getData();
	U* flowTraceData = flowTrace->getData();

#pragma omp parallel for
	for(int y = 0; y < flowTrace->height; y++) {
		for(int x = 0; x < flowTrace->width; x++)
		{
			int index = y*flowTrace->width + x;

			U centerValue = flowTraceData[index];

			if(centerValue != FLOWTRACE_LAKE)
			{
				for(int n = 0; n < 8; n++) {
					int nx, ny, dx, dy;

					switch(n) {
						case 0: dx =  0; dy = -1; break;
						case 1: dx =  1; dy = -1; break;
						case 2: dx =  1; dy =  0; break;
						case 3: dx =  1; dy =  1; break;
						case 4: dx =  0; dy =  1; break;
						case 5: dx = -1; dy =  1; break;
						case 6: dx = -1; dy =  0; break;
						case 7: dx = -1; dy = -1; break;
					}

					nx = x + dx;
					ny = y + dy;

					if(nx < 0 || nx >= flowTrace->width || ny < 0 ||
ny >= flowTrace->height) {
						continue;
					}

					int nIndex = ny*flowTrace->width + nx;
					if(centerValue < flowTraceData[nIndex] && flowTraceData[nIndex] != FLOWTRACE_LAKE) {
						catchmentBorderData[index] = 1;
						break;
					}
				}
			}
		}
	}
}

#endif /* CATCHMENTBORDERALGORITHMCPUOPENMP_H_ */
