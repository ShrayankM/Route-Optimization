# Route Optimization
## Implementation Detailed
Module 1 - Downloading Tiles and Generating area of study

Get area of study file from https://vedas.sac.gov.in/energymap/ for Natural Gas Pipeline and Road Data from https://wiki.openstreetmap.org/wiki/India/National_Highway_Network. Mark gas pipeline manually whereas road data is available directly. 

Vedas Gas Pipeline Area of Interest [Download 2 files Vector(1) - points-lines for gas lines, Vector(2) - polygon shape covering area of interest. Download both GeoJson and KML version]

OpenStreetMap Area of Interest [Directly available National Highways data. Check Wikidata code for respective highway if available download using overpass turbo https://overpass-turbo.eu/ ] Polygon download pending for highways * Running query on overpass turbo =  set wikidata and go to region of interest before building the query (can cause errors) *

Manually Downloading and Selecting tiles for the study area
Use USGS Earth explorer https://earthexplorer.usgs.gov/  to get exact coordinates of the study area by uploading a KML file. Use Copernicus Open Access Hub https://scihub.copernicus.eu/ .Details - Enter Sensing Period (Feb - May), check Mission Sentinel-2, select Product Type (S2MSI2A), Mark the polygon according to coordinates from USGS Earth Explorer. * Some tiles may be offline and can be retrieved in a couple of hours*
	
	Downloading Tiles with Code
Use SentinelSat Api https://github.com/sentinelsat/sentinelsat to download products according to their products ids that are available from Copernicus Open Access Hub. 
		Code [ download_data.ipynb ]

Module 2 - Mosaicing Incomplete Single Tile
Some available tiles have some data missing which needs to be filled with the help of the same corresponding section of the tile. This mosaicing happens band wise i.e. band-2 cross section with corresponding band-2 cross section. Similarly combine all bands to get one complete tile for further processing. GDAL - https://gdal.org/programs/gdal_merge.html module is used to achieve this task.
	Code [ mosaicing.ipynb ]

Module 3 - Mosaicing Tiles of Single Area
Combine corresponding bands of different tiles i.e. band-2 of all tiles, band-3 of all tiles, and so on for all available tiles. The same libraries as the one used in the previous module will be used for this module.
	Code [ mosaicing.ipynb ] modification of code required

Module 4 - Resolution conversion from 20m to 10m
As the resolution of some bands is higher than others and hence does not match the process of layer stacking cannot be done, so convert the 20m bands to 10m. 
Code [ downscale.ipynb ]
	
	Not Compulsory to do
	Convert from JP2 format to TIFF format 
Sentinel-2 products are available in JP2 format and hence for further ease of processing we can initially convert them to TIFF.
Code [ create_tif.ipynb ]

Module 5 - Optimum Index Factor (OIF) Calculation
The Optimum Index Factor (OIF) is a statistical value that can be used to select the optimum combination of three bands in a satellite image with which you want to create a color composite. The optimum combination of bands out of all possible 3-band combinations is the one with the highest amount of 'information' (= highest sum of standard deviations), with the least amount of duplication (lowest correlation among band pairs). http://spatial-analyst.net/ILWIS/htm/ilwisapp/optimum_index_factor_functionality_algorithm.htm
	
Try combinations of 3, 4, 6, 9 bands higher the value better the combination of bands 
Code [ oif_calculation.ipynb ]

Module 6 - Layer Stacking
Combining the best bands depending on OIF values and according to need of combinations https://gisgeography.com/sentinel-2-bands-combinations/

Resolution must be the same to do layer stacking.
Code [ layer_stacking.ipynb ]



Module 7 - Clipping with Mask Layer
As sentinel-2 data tiles can become too large to process we can clip the main generated raster file using the Area of Interest (AOI) shapefile.

Code [ clipping.ipynb ]

Module 8 - Clipping with Extent
As sentinel-2 data tiles can become too large to process we can clip the main generated raster file using already available smaller rasters.
	
	Code [ clipping.ipynb ]

Module 9 - Generating LULC Mask using QGIS
Semi Automatic Classification (SCP) Plugin avaliable in Quantum GIS is used for the purpose of creating LULC mask for any a specific study area. Initially we select various Regions of Interset (ROIs) for a specific class such as Water, Dense-forest etc. After providing various ROIs as training input to SCP Spectral Angle Mapping Algorithm is used to do LULC classification of the entire study area. The entire study area mask generation is done using the QGIS open-source tool. [https://www.qgis.org/en/site/]

Module 10 - Patches Generation
64 x 64 overlapped patches were generated from the original raster and the generated LULC mask. Initially extents are acquired from the original raster using Rasterio python library. Using the main generated extents GDALs python library (gdal_translate function) is used to generate the individual masks.
Raster patches from the original TIFF format are converted to PNG format for model training. Mask patches are kept in the original TIFF format.
Generated patches are split into Train, Test and Validation datasets in ratio 70:30 percent. 
	
	Code [ create_patches.ipynb, patches_convert.ipynb, split_data.ipynb ]

Module 11 - Convolution Neural Network Models Training
Tensorflow (TensorFlow) and segmentation_models (GitHub - qubvel/segmentation_models: Segmentation models with pretrained backbones. Keras and TensorFlow Keras.) is used for models building, the process of data augmentation and training models.
Tensorflow’s ImageDataGenerator Module is used for the process of Data-augmentation.
Segmentation_models github repository provides various pre-built CNN architectures along with their backbones which are used to create 10 CNN models. 
Total of 10 CNN models are trained for a total of 7 LULC classes namely [ Water, Denseforest, Sparseforest, Barrenland, Urbanland, Farmland, Fallowland ]
Batch-size (16), Optimizer (Adam), Learning Rate (0.001), Epochs (150)
	
	Code [ train_segmentation.ipynb, train_run.sh ]

Module 12 - Testing CNN Models
All the models are tested on a batch size of 16 images from the test set. Initially Confusion Matrices (CF) for the individual models are created which give values such as TP, TN, FP and FN. Confusion matrices are used to calculate various metrics such as Overall Accuracy (OA), Precision (PR), Recall (RE), F1-Score and Matthews Co-relation co-efficient (MCC). LULC classwise comparison of individual models is also done in tabular format.
The entire study area is too big to be tested directly so non-overlapping patches of the study area are generated and prediction is done on each and every individual patch. Once prediction is done the individual patches are mosaicked back to form the complete study area.

	Code [ segmentation_testing.ipynb, segmentation_testing_v2.ipynb ]

Module 13 - DEM of Study Area (Digital Elevation Model)
Digital Elevation Model (DEM) for the study area is downloaded using USGS Earth Explorer (https://earthexplorer.usgs.gov/). Downloaded DEM tile is 30M resolution and raster is 10M so initially 30M DEM is converted to 10M resolution.

Code [ dem-upscale.ipynb ] 

Module 14 - AHP (Analytical Hierarchy Process)
Multi-Criteria Decision Making (MCDM) method AHP is used to rank the multiple LULC classes for the order of their importance such that the consistency ratio is less than 0.10.
	
	Code [ ahp.ipynb ]

Module 15 - Route Optimization
Create source and destination shape file using QGIS software, 2 individual shape files. (source.shp and destination.shp) source and destination shapes are in co-ordinates system so need for conversion to pixels to identify start-pixel and end-pixel. Extent of the raster is used to calculate the exact start and end pixels by traversing the entire the raster.
Accumulated cost surface calculation using generated MASK and DEM data.
Queen’s Pattern (8 Neighbors) for every pixel is used to determine the neigbours.
Dijkstra’s Algorithm implemented using Priority Queues is written from scratch, A* Algorithm is also implemented using various Heuristics such as Euclidean, Manhattan, Diagonal, Chebyshev as well as a New Proposed Heuristic.
Once Accumulated cost surface is generated using the above algorithms the entire optimized route is backtracked from destination pixel to source pixel and stored in path.txt file.
The generated backtracked path is in pixel form which is converted back to the co-ordinates system to get the shape file for the route by using extents.
Optimized route length is calculated in the final step to check distance reduction.

	Code [ dijk-route.ipynb, a-star-route-diag.ipynb ]

Module 16 - Route Analysis
Original route distance and analysis such as LULC road passing percentage, pixel percentage and road percentage is done first.
Difference between GAS Pipeline and ROAD route original distance calculation.
All found routes are analysed to find LULC classification.

	Code [ route-analysis.ipynb ] 
`


	
	
