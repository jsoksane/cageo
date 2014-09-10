/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file RasterDEMFactory.h
 */

#ifndef RASTERDEMFACTORY_H_
#define RASTERDEMFACTORY_H_

#include <assert.h>

#include "AbstractFactory.h"
#include "DemData.h"

/**
 * \brief A factory that constructs RasterDEM objects for the current
 * framework.
 */
class RasterDEMFactory : public AbstractFactory {
public:
	RasterDEMFactory();
	virtual ~RasterDEMFactory();

    /**
     * \brief Create and return a new RasterDEM object using \a model as a
     * template.
     */
	template<typename T>
	static RasterDEM<DemData>* create(
			CellGrid<T>& model,
			CopyType copyType);

    /**
     * \brief Read elevation data from the file and return a new RasterDEM
     * object based on the read data.
     */
	static RasterDEM<DemData>* create(
			const std::string &demName,
            const std::string &demPath,
            const std::string &demSuffix,
            int width,
            int height);
};

template<typename T>
RasterDEM<DemData>* RasterDEMFactory::create(
		CellGrid<T>& model,
		CopyType copyType)
{
	switch(defaultImpl) {
	case CPU:
	case CPU_OpenMP: return new RasterDEM<DemData>(model, HOST, copyType);

	case GPU:
	case GPU_CUDA:   return new RasterDEM<DemData>(model, DEVICE, copyType);
	default:         assert(false); return NULL;
	}
}

RasterDEM<DemData>* RasterDEMFactory::create(const std::string & demName,
                                       const std::string & demPath,
                                       const std::string & demSuffix,
                                       int width,
                                       int height) {
    switch(defaultImpl) {
    case CPU:
    case CPU_OpenMP: return new RasterDEM<DemData>(demName, demPath, demSuffix,
                                             width, height, HOST);
    case GPU:
    case GPU_CUDA: return new RasterDEM<DemData>(demName, demPath, demSuffix,
                                           width, height, DEVICE);
	default:         assert(false); return NULL;
    }
}

#endif /* RASTERDEMFACTORY_H_ */
