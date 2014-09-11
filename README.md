A full graphics processing unit implementation of uncertainty-aware drainage basin delineation
==============================================================================================

David Eränen (davideranen@hotmail.com) (a), Juha Oksanen (b), Jan Westerholm (a), Tapani Sarjakoski (b)

(a) Åbo Akademi University, Department of Information Technologies, Joukahainengatan 3-5 a, FI-20520 Åbo, Finland
(b) Finnish Geodetic Institute, Department of Geoinformatics and Cartography, P.O. Box 15, FI-02431 Masala, Finland

Corresponding author: Juha Oksanen, juha.oksanen@fgi.fi


Abstract
--------

Terrain analysis based on modern, high-resolution Digital Elevation Models (DEMs) has become quite time consuming because of the large amounts of data involved. Additionally, when the propagation of uncertainties during the analysis process is investigated using the Monte Carlo method, the run time of the algorithm can increase by a factor of between 100 and 1000, depending on the desired accuracy of the result. This increase in run time constitutes a large barrier when we expect the use of uncertainty-aware terrain analysis become more general. In this paper, we evaluate the use of Graphics Processing Units (GPUs) in uncertainty-aware drainage basin delineation. All computations are run on a GPU, including the creation of the realization of a stationary DEM uncertainty model, stream burning, pit filling, flow direction calculation, and the actual delineation of the drainage basins. On average, our GPU version is approximately 11 times faster than a sequential, one-core CPU version performing the same task.

Keywords: geospatial analysis, data uncertainty, Monte Carlo, DEM, GPU, CUDA


Citation
--------

Eränen, D., Oksanen, J., Westerholm, J. and T. Sarjakoski (in press). A full graphics processing unit implementation of uncertainty-aware drainage basin delineation. Computers & Geosciences. DOI: 10.1016/j.cageo.2014.08.012


Installation
------------

For compiling and running the program:

cd *root directory of project*

make

./drainage


Dependencies
------------

<li>cuda (http://www.nvidia.com/getcuda)</li>
<li>boost (http://www.boost.org/)</li>
-eigen3 (http://eigen.tuxfamily.org/index.php?title=Main_Page)
-fftw3 (https://github.com/FFTW/fftw3)
-sigar (https://github.com/hyperic/sigar)

Edit the Makefile and give the correct paths to these dependencies. By default, the software will be compiled in release configuration (-O3).

The software is developed/tested in Linux environment with gcc 4.6.3, cuda 5.5, boost 1.53.0, fftw3 3.3.3 and sigar 1.6.4.

Sample data
-----------

Samples contain data from the Topographic database and DEM10 (6/2012) by the National Land Survey of Finland, 
http://www.maanmittauslaitos.fi/avoindata_lisenssi_versio1_20120501.

./DEM/dem10	Digital elevation model in 10 m grid
		http://www.maanmittauslaitos.fi/en/digituotteet/elevation-model-10-m

./DEM/streams	Rasterized streams from the Topographic database
		http://www.maanmittauslaitos.fi/en/digituotteet/topographic-database


Running the algorithm
---------------------

For getting help about all options:

	./drainage --help

Sample run used for getting the results presented in the article's Figure 1:

    ./drainage --dem-path=DEM --dem=dem10 --stream-path=DEM --stream=streams --filter-type='Gaussian' --practical-range=60 --iterations=100 --gpu


License
-------

This program is released under GNU Lesser General Public License. For more information, see files COPYING and COPYING.LESSER.

This program contains jpeg encoder library jpge, which is released under Public domain.
