# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import re

from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr

import numpy as np
from osgeo import ogr, gdal
import rasterio
import json
import rasterio
from rasterio.enums import Resampling

from copy import deepcopy
from osgeo import ogr
import json

from osgeo import ogr, gdal
import os
import subprocess
import rasterio

import math
from queue import PriorityQueue
from queue import Queue
from copy import deepcopy

import shapefile

# AHP Mapping
ahp_map = {

'urban': 0.29,
'farms': 0.239,
'dense-forest': 0.207,
'water': 0.13,
'fallow': 0.067,
'sparse-forest': 0.049,
'barren-land': 0.019,
'unclassified':7,
'unclassified2': 8,

}


# CLASS NAME Map
class_map = {

    -999: 'unclassified2',
    0: 'unclassified',
    1: 'water',
    2: 'dense-forest',
    3: 'sparse-forest',
    4: 'barren-land',
    5: 'urban',
    6: 'farms',
    7: 'fallow',

}


# Set to WEIGHT Mapping
def set_weights(c):
    c_str = class_map.get(c)
    return ahp_map[c_str]


# Dijkstar's Condition check (STOP Condition)
def condition_check(start, end):
    if (start[0] == end[0]) and (start[1] == end[1]):
        return False
    return True

# ANISTROPIC COST ACCUMULATION
def c_anist_cost(i, j, x, y, mu = 10, wt = 2):

    mu_sqr = mu * mu
    h_diff = dem_data[i][j] - dem_data[x][y]
    h_sqr = h_diff * h_diff
    c_dv = (mask_copy[i][j] + mask_copy[x][y]) / 2
    cst = np.sqrt(mu_sqr + h_sqr) * (c_dv + math.atan(h_diff / mu) * wt) + distance_matrix[i][j]
    return cst

# GET NEIGHBORS
def get_neigh_cost(i, j, lt_iw, lt_jh):
    arr = []

    #(1) col - 1, row - 1
    if (j - 1 >= 0) and (i - 1 >= 0):
        if (distance_matrix[i - 1][j - 1] > distance_matrix[i][j] + c_anist_cost(i, j, i - 1, j - 1)):
            distance_matrix[i - 1][j - 1] = distance_matrix[i][j] + c_anist_cost(i, j, i - 1, j - 1) 
            parent[i - 1][j - 1] = i, j
        arr.append([distance_matrix[i - 1][j - 1], [i - 1, j - 1], [i, j]])
    # else:
    #     arr.append(math.inf)
            
    
    #(2) col, row - 1
    if (i - 1 >= 0):
        if (distance_matrix[i - 1][j] > distance_matrix[i][j] + c_anist_cost(i, j, i - 1, j)):
            distance_matrix[i - 1][j] = distance_matrix[i][j] + c_anist_cost(i, j, i - 1, j)
            parent[i - 1][j] = i, j
        arr.append([distance_matrix[i - 1][j], [i - 1, j], [i, j]])
    # else:
    #     arr.append(math.inf)
    
    #(3) col + 1, row - 1
    if (j + 1 < lt_jh) and (i - 1 >= 0):
        if (distance_matrix[i - 1][j + 1] > distance_matrix[i][j] + c_anist_cost(i, j, i - 1, j + 1)):
            distance_matrix[i - 1][j + 1] = distance_matrix[i][j] + c_anist_cost(i, j, i - 1, j + 1)
            parent[i - 1][j + 1] = i, j
        arr.append([distance_matrix[i - 1][j + 1], [i - 1, j + 1], [i, j]])
    # else:
    #     arr.append(math.inf)
    
    #(4) col - 1, row
    if (j - 1 >= 0):
        if (distance_matrix[i][j - 1] > distance_matrix[i][j] + c_anist_cost(i, j, i, j - 1)):
            distance_matrix[i][j - 1] = distance_matrix[i][j] + c_anist_cost(i, j, i, j - 1)
            parent[i][j - 1] = i, j
        arr.append([distance_matrix[i][j - 1], [i, j - 1], [i, j]])
    # else:
    #     arr.append(math.inf)
    
    #(5) col + 1, row
    if (j + 1 < lt_jh):
        if (distance_matrix[i][j + 1] > distance_matrix[i][j] + c_anist_cost(i, j, i, j + 1)):
            distance_matrix[i][j + 1] = distance_matrix[i][j] + c_anist_cost(i, j, i, j + 1)
            parent[i][j + 1] = i, j
        arr.append([distance_matrix[i][j + 1], [i, j + 1], [i, j]])
    # else:
    #     arr.append(math.inf)
    
    #(6) col - 1, row + 1
    if (j - 1 >= 0) and (i + 1 < lt_iw):
        if (distance_matrix[i + 1][j - 1] > distance_matrix[i][j] + c_anist_cost(i, j, i + 1, j - 1)):
            distance_matrix[i + 1][j - 1] = distance_matrix[i][j] + c_anist_cost(i, j, i + 1, j - 1)
            parent[i + 1][j - 1] = i, j
        arr.append([distance_matrix[i + 1][j - 1], [i + 1, j - 1], [i, j]])
    # else:
    #     arr.append(math.inf)
    
    #(7) col, row + 1
    if (i + 1 < lt_iw):
        if (distance_matrix[i + 1][j] > distance_matrix[i][j] + c_anist_cost(i, j, i + 1, j)):
            distance_matrix[i + 1][j] = distance_matrix[i][j] + c_anist_cost(i, j, i + 1, j)
            parent[i + 1][j] = i, j
        arr.append([distance_matrix[i + 1][j], [i + 1, j], [i, j]])
    # else:
    #     arr.append(math.inf)
    
    #(8) col + 1, row + 1
    if (j + 1 < lt_jh) and (i + 1 < lt_iw):
        if (distance_matrix[i + 1][j + 1] > distance_matrix[i][j] + c_anist_cost(i, j, i + 1, j + 1)):
            distance_matrix[i + 1][j + 1] = distance_matrix[i][j] + c_anist_cost(i, j, i + 1, j + 1)
            parent[i + 1][j + 1] = i, j
        arr.append([distance_matrix[i + 1][j + 1], [i + 1, j + 1], [i, j]])
    # else:
    #     arr.append(math.inf)

    return arr


# Study Area LOAD
area_cover = 'mask.tif'
area = rasterio.open(area_cover, count = 1)
area = np.array(area.read(1))

# DEM LOAD
area_dem = 'dem_clipped.tif'
dem = rasterio.open(area_dem, count = 1)
dem = np.array(dem.read(1))
print(dem.shape)

# Upscaling to 10M
# whole-numbers indicate upscaling, fractions indicate downscaling
upscale_factor = 3
with rasterio.open('dem_clipped.tif') as dataset:

    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * upscale_factor),
            int(dataset.width * upscale_factor)
        ),
        resampling=Resampling.bilinear
    )

    # scale image transform
    transform = dataset.transform * dataset.transform.scale(
        (dataset.width / data.shape[-1]),
        (dataset.height / data.shape[-2])
    )

area = np.swapaxes(area, 1, 0)
up_dem = data[0]
print(up_dem.shape, area.shape)

dem_data, mask_data = up_dem, area
dem_data = dem_data.T

print(dem_data.shape, mask_data.shape)
width, height = 10500, 8163

dem_data, mask_data = dem_data[:width,:height], mask_data[:width, :height]
print(dem_data.shape, mask_data.shape)

mask_copy = deepcopy(mask_data)
set_weights_vtr = np.vectorize(set_weights, otypes=[np.float])
mask_copy = set_weights_vtr(mask_copy)

# print(np.unique(mask_copy, return_counts = True))

# SOURCE CO-ORDINATES LOAD
file = ogr.Open("source/source-point.shp")
source_shp = file.GetLayer(0)
feature = source_shp.GetFeature(0)
source_shp = feature.ExportToJson()
source_shp = json.loads(source_shp)
start_ext = source_shp['geometry']['coordinates']

# DESTINATION CO-ORDINATES LOAD
file = ogr.Open("destination/destination-point.shp")
destination_shp = file.GetLayer(0)
feature = destination_shp.GetFeature(0)
destination_shp = feature.ExportToJson()
destination_shp = json.loads(destination_shp)
end_ext = destination_shp['geometry']['coordinates']

print(start_ext, end_ext)

# GET Raster Extent
path = 'raster.tif' 
data = rasterio.open(path)
print(data.bounds)

extent = data.bounds
left, bottom, right, top = extent[0], extent[1], extent[2], extent[3]
print(left, bottom, right, top)

# width = round(right - left)
# height = round(top - bottom)

# print("Width and Height of Raster")
# print(width, height)

# Setting Start-Pixel and End-Pixel 
start_pixel, end_pixel = [14424, 17699], [54850, 83443]
start_pixel = np.array(start_pixel)
start_pixel = start_pixel / 10

start_pixel = list(np.rint(start_pixel))
start_pixel = list(np.array(start_pixel, dtype = 'int'))

end_pixel = np.array(end_pixel)
end_pixel = end_pixel / 10

end_pixel = list(np.rint(end_pixel))
end_pixel = list(np.array(end_pixel, dtype = 'int'))

mask_copy_t = deepcopy(mask_copy)
mask_copy = mask_copy * 2000

# print(np.unique(mask_copy, return_counts = True))
print(start_pixel, end_pixel)

dim = mask_copy.shape
print(dim)
print(mask_copy.shape, dem_data.shape)


# MAIN Pixels
start_pixel, end_pixel = [1770, 1442], [8344, 5485]

# DUMMY PIXELS
# start_pixel, end_pixel = [1770, 1442], [1823, 1300]

print(start_pixel, end_pixel)
print(width, height)


wd, ht = int(width / 10), int(height / 10)
lt_iw, lt_jh = wd, ht

#SHORTEST PATH FASTER ALGORITHM
distance_matrix = np.full((dim), math.inf)
inQueue = np.zeros((dim))
parent = np.full((dim), None)

i, j = start_pixel
parent[i][j] = [-1, -1]

i, j = start_pixel
distance_matrix[i][j] = 0
inQueue[i][j] = 1

Q = Queue()
Q.put(start_pixel)

cnt = 10000 * 10
stop_flag = False
last = None

while not Q.empty():
    if cnt == 0:
        break
    cnt = cnt - 1

    if stop_flag:
        break 

    i, j = Q.get()
    last = [i, j]
    inQueue[i][j] = 0

    ngbors = get_neigh_cost(i, j, lt_iw, lt_jh)
    for n in ngbors:
        a, b = n[1]

        if a == end_pixel[0] and b == end_pixel[1]:
            stop_flag = True

        if inQueue[a][b]:
            continue
        inQueue[a][b] = 1
        Q.put(n[1])


end_pixel = last
print(distance_matrix[end_pixel[0]][end_pixel[1]])

# PARENT PATH TRACE
path = []
pr = parent[end_pixel[0]][end_pixel[1]]

while (pr[0] != -1) and (pr[1] != -1):
    path.append(pr)
    with open('routes-shape/spf/path.txt', 'a') as f:
        f.write(str(pr) + '\n')

    pr = parent[pr[0]][pr[1]]

path_list = []
with open('routes-shape/spf/path.txt', 'r') as f:
    for point in f:
        path_list.append(point)

len(path_list)
path_list.reverse()

# CO-ORDINATES CONVERSION TO PIXEL
ordinates_dict = {}
for i in range(len(path_list)):
    point = path_list[i].replace('\n', '').split(' ')
    pi, pj = int(point[0].split(',')[0].split('(')[1]), int(point[1].split(',')[0].split(')')[0])

    ext_i, ext_j = left + (pi * 10), top - (pj * 10)

    ordinates_dict[pi, pj] = [ext_i, ext_j]

ordinates_list = []
for key, value in ordinates_dict.items():
    ordinates_list.append(value)

w = shapefile.Writer('routes-shape/spf/shapefiles/test/multipoint')
w.field('name', 'C')

w.multipoint(ordinates_list) 
w.record('multipoint1')

w.close()

road_lenght = 0
index = 1
for _ in ordinates_list[1:]:
    i, j = ordinates_list[index][0], ordinates_list[index][1]
    x, y = ordinates_list[index - 1][0], ordinates_list[index - 1][1]

    if (x - i == 10.0) and (y - j == -10.0):
        road_lenght += math.sqrt(2 * 100)
    else:
        road_lenght += 10 

print(f'Current Road-length = {road_lenght / 1000} kms')
