/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file PitFillingAlgorithmCPUOpenMP.h
 */

#ifndef PITFILLINGALGORITHMCPUOPENMP_H_
#define PITFILLINGALGORITHMCPUOPENMP_H_

#include "PitFillingAlgorithm.h"
#include "thrust/pair.h"
#include "thrust/extrema.h"
#include "LockQueue.h"

/**
 * \brief An OpenMP implementation of the PitFillingAlgorithm.
 */
template<typename T>
class PitFillingAlgorithm_CPU_OpenMP : public PitFillingAlgorithm<T> {
public:
	PitFillingAlgorithm_CPU_OpenMP() {};
	virtual ~PitFillingAlgorithm_CPU_OpenMP() {};

    /**
     * \brief Perform the pit filling with OpenMP-parallelized algorithm.
     */
	void performPitFilling(
				RasterDEM<T>* spill,
				RasterDEM<T>* dem);
};

#define PREC 1000

/**
 * \brief A simple struct for 2D cell coordinates.
 */
struct Cell {
        /**
         * \brief The cell indices.
         */
        unsigned short x, y;
        /**
         * \brief The constructor.
         */
        Cell(unsigned short x, unsigned short y) : x(x), y(y) { }
};

typedef LockQueue<Cell> CellQueue;

template<typename T>
inline
size_t toIndex(T elevation, T minElevation) {
        return (size_t) (PREC*(elevation - minElevation));
}

template<typename T>
inline
T toElev(size_t index, T minElevation) {
        return (((T) index)/PREC + minElevation);
}

template<typename T>
void PitFillingAlgorithm_CPU_OpenMP<T>::performPitFilling(
		RasterDEM<T>* spill,
		RasterDEM<T>* dem)
{
	spill->toHost();
	dem->toHost();

    ProcLater_t* procLater = new ProcLater_t(*dem, AS_TEMPLATE);

	T* spillData        = spill->getData();
	T* demData          = dem->getData();

	bool* procLaterData = procLater->getData();
	bool* processed     = NULL;

	if(this->iter == 0) {
		#pragma omp parallel for
		for(int y = 0; y < dem->height; y++) {
			for(int x = 0; x < dem->width; x++)
			{
				int index = y*dem->width + x;

				if(demData[index] == this->noDataValueDEM) {
					continue;
				}

				T elev = demData[index];
				size_t qIndex = toIndex(elev, (T) 0);

				// TODO: This is a temp solution. If the if-statement is true, then it means the elevation was rounded down to 0,
				// which means it will be treated as a stream(!). Instead of having an elevaiton of 0, make it 1/PREC (by setting qIndex = 1)
				if(elev != 0.0 && qIndex == 0) {
					qIndex = 1;
				}

				demData[index] = toElev(qIndex, (T) 0);
			}
		}

		memcpy(spillData, demData, dem->size*sizeof(T));
	}

	thrust::pair<T*, T*> minMax;
	minMax = thrust::minmax_element(spillData, spillData + spill->size);

	T minElev = *minMax.first;
	T maxElev = *minMax.second;

	size_t arraySize = (size_t) ((maxElev - minElev)*PREC + 1);
	CellQueue* Q = new CellQueue[arraySize]();

	if(this->iter == 0) {
		processed = new bool[dem->size]();

		#pragma omp parallel for
		for(int y = 0; y < dem->height; y++) {
			for(int x = 0; x < dem->width; x++)
			{
				int index = y*dem->width + x;

				if(spillData[index] == this->noDataValueDEM) {
					continue;
				}

				if(x == 0 || x == dem->width  - 1 ||
				   y == 0 || y == dem->height - 1 ||
				   spillData[index] == STREAM)
				{
					T elev = spillData[index];
					size_t qIndex = toIndex(elev, minElev);
					Q[qIndex].push(Cell(x, y));
					processed[index] = true;
				}
			}
		}
	}
	else {
		#pragma omp parallel for
		for(int y = 0; y < dem->height; y++) {
			for(int x = 0; x < dem->width; x++)
			{
				int index = y*dem->width + x;

				if(procLaterData[index])
				{
					T elev = spillData[index];
					size_t qIndex = toIndex(elev, minElev);
					Q[qIndex].push(Cell(x, y));
				}
			}
		}
	}

	for(int z = 0; z < arraySize; z++) {

		T cSpill = toElev((size_t) z, minElev); // Convert index to elevation

		while(!Q[z].empty()) {
			Cell c = Q[z].front();
			Q[z].pop();

			int cx = c.x;
			int cy = c.y;

			// If elevation in array does not match the expected elevation (inferred from 'z')
			// This happens when the same coordinate has been added to a queue multiple times
			// The correct modification has happened earlier, which means we can ignore this one
			// A workaround would be to keep track of which coordinates have already been added to a queue,
			// however, that requires an allocation of an array of size N, which I find to be a worse solution
			if(cSpill != spillData[cy*dem->width + cx]) {
				continue;
			}

			for(int n = 0; n < 8; n++)
			{
				int dx, dy;
				int nx, ny;

				switch(n) {
					case 0: dx = -1; dy = -1; break;
					case 1: dx =  0; dy = -1; break;
					case 2: dx =  1; dy = -1; break;
					case 3: dx = -1; dy =  0; break;
					case 4: dx =  1; dy =  0; break;
					case 5: dx = -1; dy =  1; break;
					case 6: dx =  0; dy =  1; break;
					case 7: dx =  1; dy =  1; break;
				}

				nx = cx + dx;
				ny = cy + dy;

				if(nx < 0 || nx >= dem->width || ny < 0 || ny >=
dem->height) {
					continue;
				}

				int nIndex = ny*dem->width + nx;
				T nSpill = spillData[nIndex];

				if(nSpill == this->noDataValueDEM) {
					continue;
				}

				if(this->iter == 0) {
					if(processed[nIndex]) {
						continue;
					}
					else if(nSpill < cSpill){
						spillData[nIndex] = cSpill;
					}

					int nZOrig = toIndex(spillData[nIndex], minElev);
					Q[nZOrig].push(Cell(nx, ny));
					processed[nIndex] = true;
				}
				else {
					T nSpillOrig = demData[nIndex];
					int nZ = toIndex(nSpill, minElev);
					if(z < nZ && nSpill != nSpillOrig)
					{
						spillData[nIndex] = std::max(cSpill, nSpillOrig);
						nZ = toIndex(spillData[nIndex], minElev);
						Q[nZ].push(Cell(nx, ny));
					}
				}
			}
		}
	}

	delete[] processed;
	delete[] Q;
    delete procLater;
}

#endif /* PITFILLINGALGORITHMCPUOPENMP_H_ */
