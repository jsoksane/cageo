/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file StreamDataFactory.h
 */

#ifndef STREAMDATAFACTORY_H_
#define STREAMDATAFACTORY_H_

#include <assert.h>

#include "AbstractFactory.h"
#include "StreamData.h"

/**
 * \brief A factory that constructs StreamData objects for the current
 * framework.
 */
class StreamDataFactory : public AbstractFactory {
public:
	StreamDataFactory();
	virtual ~StreamDataFactory();

    /**
     * \brief Create and return a new StreamData object from the given
     * datafile.
     */
	static StreamData<DemData>* create(
			const std::string &demName,
            const std::string &demPath,
            const std::string &demSuffix,
            int width,
            int height);
};

/**
 * \brief The location of the data is automatically selected based on the
 * selected framework.
 */
StreamData<DemData>* StreamDataFactory::create(const std::string & demName,
                                       const std::string & demPath,
                                       const std::string & demSuffix,
                                       int width,
                                       int height) {
    switch(defaultImpl) {
    case CPU:
    case CPU_OpenMP: return new StreamData<DemData>(demName, demPath, demSuffix,
                                             width, height, HOST);
    case GPU:
    case GPU_CUDA: return new StreamData<DemData>(demName, demPath, demSuffix,
                                           width, height, DEVICE);
	default:         assert(false); return NULL;
    }
}

#endif /* STREAMDATAFACTORY_H_ */
