/**
 * Copyright 2014 Finnish Geodetic Institute
 *
 * Programmers: David Eränen and Ville Mäkinen
 *
 * This file is part of the drainage program. It is released under the GNU
 * Lesser General Public Licence version 3.
 *
 *
 * \file main.cpp
 */
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <iostream>
#include <iomanip>

#include <time.h>

#include "Logging.h"

#include "CellGrid.h"
#include "RasterDEM.h"
#include "StreamData.h"
#include "RasterDEMFactory.h"
#include "StreamDataFactory.h"

#include "RandomFieldClass.h"
#include "RandomFieldAlgorithm.h"
#include "RandomFieldAlgorithmFactory.h"
#include "ConvolutionAlgorithmFactory.h"
#include "RealisticDemAlgorithm.h"
#include "StreamBurningAlgorithmFactory.h"
#include "PitFillingAlgorithmFactory.h"
#include "FlowRoutingAlgorithmFactory.h"
#include "FlowTracingAlgorithmFactory.h"
#include "CatchmentBorderAlgorithmFactory.h"
#include "ToByteArrayAlgorithmFactory.h"
#include "FlowDirClass.h"
#include "FlowTraceDataClass.h"
#include "CatchmentBorderClass.h"

#include "DemData.h"
#include "DemInfo.h"

#include "CudaUtils.cuh"
#include "SystemUtils.h"
#include "Utils.h"
#include "Timer.h"
#include "TimerTree.h"

#include "jpge.h"

void parseArgs(int argc, char** argv);

int main(int argc, char** argv) {

	parseArgs(argc, argv);

    if (AbstractFactory::defaultImpl == AbstractFactory::GPU ||
        AbstractFactory::defaultImpl == AbstractFactory::GPU_CUDA) {
        int localDeviceCount, myDevice;
        cudaGetDeviceCount(&localDeviceCount);
        myDevice = 0; // Device no. 0 is the fastest.
        cudaSetDevice(myDevice);
        cudaGetDeviceProperties(&CudaUtils::deviceProp, myDevice);
        std::cout << "Acquiring device " << CudaUtils::constructDeviceName(CudaUtils::deviceProp) << std::endl << std::flush;
        // Initial synchronize
        cudaDeviceSynchronize();
    } else {
        std::cout << "Using OpenMP version" << std::endl;
    }

	if(G::streamsPath.length() == 0) {
		G::streamsPath = G::demPath;
	}

	G::tempPath = G::demPath + "/temp";

	if(G::bevelRadius > MAX_BEVEL_RADIUS) {
        printf("Bevel radius (%d) is larger than MAX_BEVEL_RADIUS (%d). Please change MAX_BEVEL_RADIUS and recompile.\n",
            G::bevelRadius, MAX_BEVEL_RADIUS);
		exit(1);
	}

	DemInfo demInfo, streamInfo;
	loadHeaderData(G::demPath + "/" + G::demName + ".hdr", &demInfo);
	loadHeaderData(G::demPath + "/" + G::streamsName + ".hdr", &streamInfo);

	bool headerInfoError = false;
	if(demInfo.fileCols != streamInfo.fileCols) {
        printf("The sizes of the DEM and stream data don't match! Exiting...\n");
		headerInfoError = true;
	}
	if(demInfo.cellSize != streamInfo.cellSize) {
        printf("The cell size of the DEM and stream data don't match! Exiting...\n");
		headerInfoError = true;
	}
	if(headerInfoError) {
		exit(1);
	}

	G::inputWidth  = demInfo.fileCols;
	G::inputHeight = demInfo.fileRows;
	G::cellSize    = demInfo.cellSize;

	G::noDataValueDEM    = demInfo.noDataValue;
	G::noDataValueStream = streamInfo.noDataValue;

	int filterRadius = (int) std::ceil( (G::practicalRange/G::cellSize)*2.5f );

    printf("DEM size: %d x %d\n", G::inputWidth, G::inputHeight);
    printf("Kernel size: %d x %d (%s)\n\n",
            filterRadius*2 + 1,
            G::filterModel == GAUSSIAN ? 1 : filterRadius*2 + 1,
            G::filterModel == GAUSSIAN ? "Gaussian" : "Exponential");

	Timer mcTimer;
    Timer randomFieldTimer, convolutionTimer, realisticDemTimer,
        streamBurningTimer, pitFillTimer, flowRoutingTimer, flowTracingTimer,
        catchmentBorderTimer;
	Timer toByteArrayTimer, assembleDataTimer, jpegTimer;

	Timer globalTimer;
	globalTimer.startTimer("Total algorithm time");

	// Random field
	RandomFieldAlgorithm_t* randomFieldAlgorithm = RandomFieldAlgorithmFactory::create<RandomFieldType>();
	randomFieldAlgorithm->setMean(G::errorModelMean);
	randomFieldAlgorithm->setStandardDeviation(G::errorModelStandardDeviation);

	// Convolution
	ConvolutionAlgorithm_t* convolutionAlgorithm = ConvolutionAlgorithmFactory::create<RandomFieldType>();
	convolutionAlgorithm->setFilterRadius(filterRadius);
	convolutionAlgorithm->setFilterModel(G::filterModel);
	convolutionAlgorithm->createFilter(G::practicalRange, G::cellSize);

    // Creating realistic dem model
    RealisticDemAlgorithm_t* realisticDemAlgorithm = new RealisticDemAlgorithm<DemData, RandomFieldType>();

	// Stream burning
	StreamBurningAlgorithm_t* streamBurningAlgorithm = StreamBurningAlgorithmFactory::create<DemData, DemData>();
	streamBurningAlgorithm->setBevelRadius(G::bevelRadius);
	streamBurningAlgorithm->setNoDataValueDEM(G::noDataValueDEM);
	streamBurningAlgorithm->setNoDataValueStream(G::noDataValueStream);

	// Pit filling
	PitFillingAlgorithm_t* pitfillingAlgorithm = PitFillingAlgorithmFactory::create<DemData>();
	pitfillingAlgorithm->setNoDataValueDem(G::noDataValueDEM);

	// Flow routing
	FlowRoutingAlgorithm_t* flowRoutingAlgorithm =
			FlowRoutingAlgorithmFactory::create<FlowDirDataType, DemData>();
	flowRoutingAlgorithm->setNoDataValueDEM(G::noDataValueDEM);

	// Flow tracing
	FlowTracingAlgorithm_t* flowTracingAlgorithm =
			FlowTracingAlgorithmFactory::create<FlowTraceDataType, FlowDirDataType, DemData>();

	// Catchment border extraction
	CatchmentBorderAlgorithm_t* catchmentBorderAlgorithm =
			CatchmentBorderAlgorithmFactory::create<CatchmentBorderType, FlowTraceDataType>();

	// TODO: Make CPU version of this
	ToByteArrayAlgorithm<float>* toByteArrayAlgorithm = ToByteArrayAlgorithmFactory::create<float>();

    // Data objects that stay the same during the Monte Carlo iterations.
    RasterDEM_t* dem = RasterDEMFactory::create(G::demName,
                                                G::demPath,
                                                ".bin",
                                                G::inputWidth,
                                                G::inputHeight);
    StreamData_t* streams = StreamDataFactory::create(G::streamsName,
                                                      G::demPath,
                                                      ".bin",
                                                      G::inputWidth,
                                                      G::inputHeight);
    CatchmentBorder_t* monteCarloBorder = new CatchmentBorder_t(*dem,
                                                                AS_TEMPLATE);
	for(int iterMC = 0; iterMC < G::monteCarloIterations; iterMC++)
	{
		LOG_TRACE("*** Monte Carlo iteration " << iterMC << " ***");
		TIMING_START(mcTimer, "Monte Carlo iteration");

        // Temporary working array
        RandomField_t* rand = new RandomField_t(*dem, AS_TEMPLATE);

		// RANDOM FIELD
        time_t t;
        time(&t);
		TIMING_START(randomFieldTimer, "Random field");
            randomFieldAlgorithm->execute(rand, (int) t + iterMC);
		TIMING_END_PRINT(randomFieldTimer);

		// CONVOLUTION
		TIMING_START(convolutionTimer, "Convolution");
			convolutionAlgorithm->execute(rand);
		TIMING_END_PRINT(convolutionTimer);

        RasterDEM_t* rDem = (RasterDEM_t*) rand;

        // CREATE REALISTIC DEM MODEL
		TIMING_START(realisticDemTimer, "Realistic DEM");
            realisticDemAlgorithm->execute(dem, rand, rDem);
		TIMING_END_PRINT(realisticDemTimer);

        dem->cacheDataIfNeeded("dem");

		// STREAM BURNING
		TIMING_START(streamBurningTimer, "Stream burning");
			streamBurningAlgorithm->execute(rDem, streams);
		TIMING_END_PRINT(streamBurningTimer);

        streams->cacheDataIfNeeded("streams");

		// PIT FILLING
        RasterDEM_t* demPitsFilled = new RasterDEM_t(*rDem, AS_TEMPLATE);

		TIMING_START(pitFillTimer, "Pit filling");
			pitfillingAlgorithm->execute(demPitsFilled, rDem);
		TIMING_END_PRINT(pitFillTimer);
        delete rDem;

		// FLOW ROUTING
        FlowDir_t* flowDir = new FlowDir_t(*dem, AS_TEMPLATE);
		TIMING_START(flowRoutingTimer, "Flow routing");
			flowRoutingAlgorithm->execute(flowDir, demPitsFilled);
		TIMING_END_PRINT(flowRoutingTimer);
        delete demPitsFilled;

		// FLOW TRACING
        FlowTraceData_t* flowTraceData = new FlowTraceData_t(*flowDir, AS_TEMPLATE);

		TIMING_START(flowTracingTimer, "Flow tracing");
			flowTracingAlgorithm->execute(flowTraceData, flowDir,
                streams);
		TIMING_END_PRINT(flowTracingTimer);
        delete flowDir;

//		// EXTRACT CATCHMENT BORDER (and add it to Monte Carlo result)
		TIMING_START(catchmentBorderTimer, "Catchment border extraction");
			catchmentBorderAlgorithm->execute(monteCarloBorder, flowTraceData);
		TIMING_END_PRINT(catchmentBorderTimer);
        delete flowTraceData;

		/*
		 * HANDLING FILE OUTPUT BELOW
		 */

		if((G::outputOption == G::OUTPUT_OPTION_END && iterMC + 1 == G::monteCarloIterations) || G::outputOption == G::OUTPUT_OPTION_FRAMES)
		{
			std::stringstream fileName;
			fileName << G::outputName;
			if(G::outputOption == G::OUTPUT_OPTION_FRAMES) {
				fileName << "_" << std::setfill('0') << std::setw(4) << iterMC;
			}

			if(G::outputType == G::OUTPUT_TYPE_JPEG) {
				CellGrid<Byte_t>* byteArray = new CellGrid<Byte_t>(*monteCarloBorder, AS_TEMPLATE);

				TIMING_START(toByteArrayTimer, "Converting to byte array");
					toByteArrayAlgorithm->execute(byteArray, (CellGrid<CatchmentBorderType>*) monteCarloBorder);
				TIMING_END_PRINT(toByteArrayTimer);

				TIMING_START(jpegTimer, "JPEG Encode time");
                    fileName << ".jpg";
                    byteArray->toHost();
                    jpge::compress_image_to_jpeg_file(
                            fileName.str().c_str(), G::inputWidth, G::inputHeight, 1,
                            (const jpge::uint8*) byteArray->getData(), jpge::params());
				TIMING_END_PRINT(jpegTimer);
                delete byteArray;

			} else if (G::outputType == G::OUTPUT_TYPE_BINARY) {
                monteCarloBorder->toHost();
                CatchmentBorderType *data = monteCarloBorder->getData();

				fileName << ".bin";
				SystemUtils::writeToFile(data, G::inputWidth*G::inputHeight, fileName.str());
			}
		}
		TIMING_END_PRINT(mcTimer);
	}

    delete dem;
    delete streams;
    delete monteCarloBorder;

	TIMING_END(globalTimer);
    std::cerr << "\n\n***********\n\n";
	TIMING_PRINT_TOTAL(globalTimer);
	TIMING_PRINT_AVERAGE(mcTimer);
	TIMING_PRINT_AVERAGE(randomFieldTimer);
	TIMING_PRINT_AVERAGE(convolutionTimer);
    TIMING_PRINT_AVERAGE(realisticDemTimer);
	TIMING_PRINT_AVERAGE(streamBurningTimer);
	TIMING_PRINT_AVERAGE(pitFillTimer);
	TIMING_PRINT_AVERAGE(flowRoutingTimer);
	TIMING_PRINT_AVERAGE(flowTracingTimer);
	TIMING_PRINT_AVERAGE(catchmentBorderTimer);

	TIMING_DEVICE_MEMCPY_PRINT_TOTAL();
	TIMING_SLACK_PRINT_TOTAL();
	TIMING_IO_READ_PRINT_TOTAL();

    TimerTree tt(&globalTimer);
    tt.setChild(&randomFieldTimer);
    tt.setChild(&convolutionTimer);
    tt.setChild(&realisticDemTimer);
    tt.setChild(&streamBurningTimer);
    tt.setChild(&pitFillTimer);
    tt.setChild(&flowRoutingTimer);
    tt.setChild(&flowTracingTimer);
    tt.setChild(&catchmentBorderTimer);
    tt.summary();

	cudaDeviceReset();
	return 0;
}

void parseArgs(int argc, char** argv) {
	std::string filterType;
	std::string outputFile, outputOption;
	std::string visualization;

	bool useGPU = true, useCPU = false;
	namespace po = boost::program_options;

	po::options_description desc("Options");
	desc.add_options()
	    ("help", "Produce help message")
	    ("dem",
	    		po::value<std::string>(&G::demName)->default_value(G::demName),
	    		"Name of the input DEM")
	    ("stream",
	    		po::value<std::string>(&G::streamsName)->default_value(G::streamsName),
	    		"Name of the input stream network")
	    ("dem-path",
	    		po::value<std::string>(&G::demPath)->default_value(G::demPath),
	    		"Path to the input DEM")
		("stream-path",
				po::value<std::string>(&G::streamsPath)->default_value(G::streamsPath),
				"Path to the input stream network")
	    ("filter-type",
	    		po::value<std::string>(&filterType), "Spatial autocorrelation model for the random field ('Gaussian' or 'Exponential')")
	    ("practical-range",
	    		po::value<float>(&G::practicalRange)->default_value(G::practicalRange), "Practical range of the spatial autocorrelation model for the random field")
		("error-mean",
				po::value<float>(&G::errorModelMean)->default_value(G::errorModelMean),
				"Mean of the random field")
		("error-standard-deviation",
				po::value<float>(&G::errorModelStandardDeviation)->default_value(G::errorModelStandardDeviation),
				"Standard deviation of the random field")
		("iterations,i",
				po::value<int>(&G::monteCarloIterations)->default_value(G::monteCarloIterations),
				"Number of Monte Carlo iterations")
		("gpu",
				po::value<bool>(&useGPU)->default_value(true)->implicit_value(true),
				"Run algorithm on GPU")
		("cpu",
				po::value<bool>(&useCPU)->default_value(false)->implicit_value(true),
				"Run algorithm on CPU")
		("output-file,o",
				po::value<std::string>(&outputFile),
				"Write result into file (e.g. '-o result.jpg' will write a jpeg file of the result). File formats currently supported are *.bin and *.jpg")
		("output-option",
				po::value<std::string>(&outputOption),
				"Specify how and when the output file is written to disk.\n"
				"Options currently supported:"
				"\n\tend         (Write final output)"
				"\n\tframes      (Write output each iteration into differently named files (e.g. file_001, file_002, ...))")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")) {
	    cout << desc << "\n";
	    exit(EXIT_SUCCESS);
	}

	if(vm.count("output-file")) {
		boost::filesystem::path path(outputFile);
		std::string name = path.stem().string();
		std::string ext  = path.extension().string();
		if(ext.compare(".bin") == 0) {
			G::outputType = G::OUTPUT_TYPE_BINARY;
		}
		else if(ext.compare(".jpg") == 0 || ext.compare("") == 0) {
			G::outputType = G::OUTPUT_TYPE_JPEG;
		}
		else {
			printf("Unknown output extension provided: %s. Exiting...\n", ext.c_str());
			exit(EXIT_FAILURE);
		}

		if(name.compare("") != 0) {
			G::outputName = name;
		}

		G::outputOption = G::OUTPUT_OPTION_END; // If output file given, then output final result
	}

	if(vm.count("output-option")) {
		if(outputOption.compare("end") == 0) {
			G::outputOption = G::OUTPUT_OPTION_END;
		}
		else if(outputOption.compare("frames") == 0) {
			G::outputOption = G::OUTPUT_OPTION_FRAMES;
		}
		else {
			printf("Unknown output option provided: %s. Exiting...\n", outputOption.c_str());
			exit(EXIT_FAILURE);
		}
	}

	if(vm.count("filter-type"))
	{
		if(filterType.compare("Exponential") == 0) {
			G::filterModel = EXPONENTIAL;
		}
		else if(filterType.compare("Gaussian") == 0) {
			G::filterModel = GAUSSIAN;
		}
		else {
			printf("Unknown filter type provided: %s. Exiting...\n", filterType.c_str());
			exit(EXIT_FAILURE);
		}
	}

	if(useCPU) {
		useGPU = false;
	}

	if(useGPU) {
		AbstractFactory::defaultImpl = AbstractFactory::GPU;
	}
	else if(useCPU) {
		AbstractFactory::defaultImpl = AbstractFactory::CPU;
	}
}
