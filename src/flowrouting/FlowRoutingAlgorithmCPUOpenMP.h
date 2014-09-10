/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file FlowRoutingAlgorithmCPUOpenMP.h
 */

#ifndef FLOWROUTINGALGORITHMCPUOPENMP_H_
#define FLOWROUTINGALGORITHMCPUOPENMP_H_

#include "FlowRoutingAlgorithm.h"

typedef std::vector<Coord> FlatCellList;

template<typename T, typename U>
class FlowRoutingAlgorithm_CPU_OpenMP : public FlowRoutingAlgorithm<T, U> {

protected:
	FlatCellList* flatCellLists;

public:
	FlowRoutingAlgorithm_CPU_OpenMP();
	virtual ~FlowRoutingAlgorithm_CPU_OpenMP();

	void performFlowRouting(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem);

	void performFlowRoutingOnFlatSurfaces(
			FlowDirClass<T>* flowDir,
			FlatDist_t*      flatDist,
			CellGrid<U>*     dem);
};

template<typename T, typename U>
FlowRoutingAlgorithm_CPU_OpenMP<T, U>::FlowRoutingAlgorithm_CPU_OpenMP()
{
	flatCellLists = new FlatCellList[1];
}

template<typename T, typename U>
FlowRoutingAlgorithm_CPU_OpenMP<T, U>::~FlowRoutingAlgorithm_CPU_OpenMP()
{
	delete[] flatCellLists;
}

template<typename T, typename U>
void FlowRoutingAlgorithm_CPU_OpenMP<T, U>::performFlowRouting(
		FlowDirClass<T>* flowDir,
		FlatDist_t* flatDist,
		CellGrid<U>* dem)
{
	flowDir->toHost();
	flatDist->toHost();
	dem->toHost();

	flowDir->clearData();
	flatDist->clearData();

	T*            flowDirData  = flowDir->getData();
	FlatDistType* flatDistData = flatDist->getData();
	U*            demData      = dem->getData();
	FlatCellList* flatCellList = &flatCellLists[0];

#pragma omp parallel for
	for(int y = 0; y < dem->height; y++) {
		for(int x = 0; x < dem->width; x++) {
			int dxi = 0, dyi = 0;

			if(x == 0)
				dxi = -1;
			else if(x == dem->width - 1)
				dxi =  1;
			if(y == 0)
				dyi = -1;
			else if(y == dem->height - 1)
				dyi =  1;

			int index = y * dem->width + x;

			if(dxi != 0 || dyi != 0) {
				flowDirData[index] = (T) {dxi, dyi};
				flatDistData[index] = -1;
				continue;
			}
			if(demData[index] == STREAM) {
				flatDistData[index] = -2;
				continue;
			}

			U steepest = 0.0;
			int steepestDx = 0;
			int steepestDy = 0;

			for(int n = 0; n < 8; n++) {
				int nx, ny;
				int dx = 0, dy = 0;

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

				if(nx < 0 || nx >= dem->width || ny < 0 || ny >=
dem->height) {
					continue;
				}

				U steepness = demData[index] - demData[ny*dem->width + nx];

				if(steepness > 0.0) {
					if(n%2 == 1) {
						steepness = steepness/MATH_SQRTTWO;
					}
					if(steepness > steepest) {
						steepest = steepness;
						steepestDx = dx;
						steepestDy = dy;
					}
				}
			}

			if(steepest > 0.0f) {
				flowDirData[index] = (T) {steepestDx, steepestDy};
				flatDistData[index] = -1;
			}
			else {
				#pragma omp critical
				{
					Coord c = {x, y};
					flatCellList->push_back(c);
				}
			}
		}
	}
}

template<typename T, typename U>
void FlowRoutingAlgorithm_CPU_OpenMP<T, U>::performFlowRoutingOnFlatSurfaces(
		FlowDirClass<T>* flowDir,
		FlatDist_t* flatDist,
		CellGrid<U>* dem)
{
	flowDir->toHost();
	flatDist->toHost();
	dem->toHost();

	T*            flowDirData  = flowDir->getData();
	FlatDistType* flatDistData = flatDist->getData();
	U*            demData      = dem->getData();
	FlatCellList& flatCellList = flatCellLists[0];

	int threadCount = omp_get_max_threads();

	FlatCellList* dst_list = new FlatCellList[threadCount];

	bool moreToDo   = true;
	int idToLookFor = -1;

#pragma omp parallel shared(moreToDo, flatCellList, idToLookFor, dst_list)
	while(moreToDo)
	{
		// Make sure all threads are inside the while-loop before we change 'moreToDo' to 'false'
		#pragma omp barrier

		moreToDo     = false;
		int listSize = flatCellList.size();
		int tid      = omp_get_thread_num();

		#pragma omp barrier

		#pragma omp for
		for(int i = 0; i < listSize; i++)
		{
			Coord c           = flatCellList[i];
			int index         = c.y*dem->width + c.x;
			bool givenFlowDir = false;

			for(int n = 0; n < 8; n++) {
				int nx, ny;
				int dx = 0, dy = 0;

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

				nx = c.x + dx;
				ny = c.y + dy;

				if(nx < 0 || nx >= dem->width || ny < 0 || ny >=
dem->height) {
					continue;
				}

				int nIndex = ny*dem->width + nx;
				if(flatDistData[nIndex] == idToLookFor && demData[nIndex] <= demData[index])
				{
					flatDistData[index]  = (idToLookFor == -1) ? 1 : idToLookFor + 1;
					flowDirData[index].x = dx;
					flowDirData[index].y = dy;
					givenFlowDir         = true;
					moreToDo             = true;
				}
			}
			if(!givenFlowDir) {
				dst_list[tid].push_back(c);
			}
		}

		#pragma omp barrier
		#pragma omp master
		{
			flatCellList.clear();
			for(int i = 0; i < threadCount; i++) {
				flatCellList.insert(flatCellList.end(), dst_list[i].begin(), dst_list[i].end());
				dst_list[i].clear();
			}
			++idToLookFor;
			if(idToLookFor == 0) {
				idToLookFor = 1;
			}
		}
	}

	flatCellList.clear();
	delete[] dst_list;
}

#endif /* FLOWROUTINGALGORITHMCPUOPENMP_H_ */
