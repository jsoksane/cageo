/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowTracingAlgorithmCPUOpenMP.h
 */

#ifndef FLOWTRACINGALGORITHMCPUOPENMP_H_
#define FLOWTRACINGALGORITHMCPUOPENMP_H_

#include "FlowTracingAlgorithm.h"
#include "FlowTracingCommon.h"

template<typename T, typename U, typename V>
class FlowTracingAlgorithm_CPU_OpenMP : public FlowTracingAlgorithm<T, U, V> {
public:
	FlowTracingAlgorithm_CPU_OpenMP() {};
	virtual ~FlowTracingAlgorithm_CPU_OpenMP() {};

	void performFlowTracing(
			FlowTraceDataClass<T>* flowTrace,
			FlowDirClass<U>*       flowDir,
			StreamData<V>*         streams);

	void performConnectFlowPath(
			FlowTraceDataClass<T>* flowTrace);
};

template<typename T, typename U, typename V>
void FlowTracingAlgorithm_CPU_OpenMP<T, U, V>::performFlowTracing(
		FlowTraceDataClass<T>* flowTrace,
		FlowDirClass<U>* flowDir,
		StreamData<V>* streams)
{
	flowTrace->toHost();
	flowDir->toHost();
	streams->toHost();

	flowTrace->clearData();

	T* flowTraceData = flowTrace->getData();
	U* flowDirData   = flowDir->getData();
	V* streamsData   = streams->getData();

	std::vector<int> indexPath;

	for(int y = 0; y < flowTrace->height; y++) {
		for(int x = 0; x < flowTrace->width; x++) {
			int curX = x;
			int curY = y;
			int index = y * flowTrace->width + x;

			if(streamsData[index] >= 0.0) {
				flowTraceData[index] = FLOWTRACE_LAKE;
				continue;
			}
			if(flowTraceData[index] < 0) {
				continue;
			}

			int iter = 0;
			while(true) {
				if(iter > 100000) {
					printf("Infinite loop?\n");
//					bool* infloop = new bool[flowTrace.localSizePad]();
//					for(int i = 0; i < indexPath.size(); i++) {
//						infloop[indexPath[i]] = true;
//					}
					break;
				}

				indexPath.push_back(index);

				curX  += flowDirData[index].x;
				curY  += flowDirData[index].y;
				index = curY * flowTrace->width + curX;

				// If outside border
				if(curX < 0 || curX >= flowTrace->width ||
				   curY < 0 || curY >= flowTrace->height) {
					for(int i = 0; i < (int)indexPath.size(); i++) {
						flowTraceData[indexPath[i]] = FLOWTRACE_OUTSIDE;
					}
					break;
				}
				// If found a lake/stream
				if(streamsData[index] >= 0.0) {
					for(int i = 0; i < (int)indexPath.size(); i++) {
						flowTraceData[indexPath[i]] = -streamsData[index];
					}
					break;
				}
				// If cell has already been traced
				if(flowTraceData[index] < 0) {
					for(int i = 0; i < (int)indexPath.size(); i++) {
						flowTraceData[indexPath[i]] = flowTraceData[index];
					}
					break;
				}

				iter++;
			}

			indexPath.clear();

		}
	}
}

template<typename T, typename U, typename V>
void FlowTracingAlgorithm_CPU_OpenMP<T, U, V>::performConnectFlowPath(
		FlowTraceDataClass<T>* flowTrace)
{
	flowTrace->toHost();
	T* flowTraceData = flowTrace->getData();

	for(int y = 0; y < flowTrace->height; y++) {
		for(int x = 0; x < flowTrace->width; x++) {
			int index = y * flowTrace->width + x;

			if(flowTraceData[index] >= 0) {
				int newIndex = flowTraceData[index];
				if(flowTraceData[newIndex] < 0) {
					flowTraceData[index] = flowTraceData[newIndex];
				}
			}
		}
	}
}


#endif /* FLOWTRACINGALGORITHMCPUOPENMP_H_ */
