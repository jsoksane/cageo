/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file LockQueue.h
 */

#ifndef LOCKQUEUE_H_
#define LOCKQUEUE_H_

#include <stdlib.h>
#include <queue>
#include <omp.h>

/**
 * \brief A queue that is thread-safe inside OpenMP-parallelized loop.
 */
template<typename T>
class LockQueue : public std::queue<T> {
public:
	LockQueue() : std::queue<T>::queue() {};
	virtual ~LockQueue() {};

	void push(const T& x) {
		#pragma omp critical
		{
			std::queue<T>::push(x);
		}
	}

	T& front() {
		T* ret;
		#pragma omp critical
		{
			ret = &std::queue<T>::front();
		}
		return *ret;
	}

	void pop() {
		#pragma omp critical
		{
			std::queue<T>::pop();
		}
	}

	bool empty() const {
		bool ret;
		#pragma omp critical
		{
			ret = std::queue<T>::empty();
		}
		return ret;
	}

	T* frontAndPopIfNotEmpty() {
		T* ret;
		#pragma omp critical
		{
			if(!std::queue<T>::empty()) {
				ret = &std::queue<T>::front();
				std::queue<T>::pop();
			}
			else {
				ret = NULL;
			}
		}
		return ret;
	}
};

#endif /* LOCKQUEUE_H_ */
