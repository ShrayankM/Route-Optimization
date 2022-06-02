# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os.path
import re

import timeit

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

# Routes-paths Mapping
route_paths = {
    1: 'a-star-man',
    2: 'a-star-euc',
    3: 'a-star-diag',
    4: 'a-star-chesb',
    5: 'a-star-new',
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
    cst = np.sqrt(mu_sqr + h_sqr) * (c_dv + math.atan(h_diff / mu) * wt) + acc_cost[i][j]
    return cst

# HEURISTIC FUNCTIONS
# Manhattan Distance
def h_cost_manhattan(i, j):
    xa, ya, xb, yb = end_pixel[0], end_pixel[1], i, j
    return abs(xa - xb) + abs(ya - yb)

# Euclidean Distance
def h_cost_euclidean(i, j):
    xa, ya, xb, yb = end_pixel[0], end_pixel[1], i, j
    return math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)

# Diagonal Distance
def h_cost_diagonal(i, j):
    xa, ya, xb, yb = end_pixel[0], end_pixel[1], i, j
    return abs(xa - xb) + abs(ya - yb) + (math.sqrt(2) - 2) * min(abs(xa - xb), abs(ya - yb))

# Chebyshev Distance
def h_cost_chebyshev(i, j):
    xa, ya, xb, yb = end_pixel[0], end_pixel[1], i, j
    return max(abs(xa - xb), abs(ya - yb))

# New-heuristic function
def h_cost_new(i, j):
    xa, ya, xb, yb = end_pixel[0], end_pixel[1], i, j
    return h_cost_euclidean(i, j) + h_cost_chebyshev(i, j) * 2

# Heuristic Mapping
heuristic_dict = {
    1: h_cost_manhattan,
    2: h_cost_euclidean,
    3: h_cost_diagonal,
    4: h_cost_chebyshev,
    5: h_cost_new
}

# GET NEIGHBORS
def get_neigh_cost(i, j, flag, lt_iw, lt_jh):
    arr = []

    heuristic_func = heuristic_dict.get(flag)


    #(1) col - 1, row - 1
    if (j - 1 >= 0) and (i - 1 >= 0):
        g = c_anist_cost(i, j, i - 1, j - 1)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i - 1][j - 1] > (g + h)):
            parent[i - 1][j - 1] = i, j
            acc_cost[i - 1][j - 1] = (g + h)
        arr.append([acc_cost[i - 1][j - 1], [i - 1, j - 1], [i, j]])
    else:
        arr.append(math.inf)

    #(2) col, row - 1
    if (i - 1 >= 0):
        g = c_anist_cost(i, j, i - 1, j)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i - 1][j] > (g + h)):
            parent[i - 1][j] = i, j
            acc_cost[i - 1][j] = (g + h)
        arr.append([acc_cost[i - 1][j], [i - 1, j], [i, j]])
    else:
        arr.append(math.inf)

    #(3) col + 1, row - 1
    if (j + 1 < lt_jh) and (i - 1 >= 0):
        g = c_anist_cost(i, j, i - 1, j + 1)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i - 1][j + 1] > (g + h)):
            parent[i - 1][j + 1] = i, j
            acc_cost[i - 1][j + 1] = (g + h)
        arr.append([acc_cost[i - 1][j + 1], [i - 1, j + 1], [i, j]])
    else:
        arr.append(math.inf)

    #(4) col - 1, row
    if (j - 1 >= 0):
        g = c_anist_cost(i, j, i, j - 1)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i][j - 1] > (g + h)):
            parent[i][j - 1] = i, j
            acc_cost[i][j - 1] = (g + h)
        arr.append([acc_cost[i][j - 1], [i, j - 1], [i, j]])
    else:
        arr.append(math.inf)

    #(5) col + 1, row
    if (j + 1 < lt_jh):
        g = c_anist_cost(i, j, i, j + 1)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i][j + 1] > (g + h)):
            parent[i][j + 1] = i, j
            acc_cost[i][j + 1] = (g + h)
        arr.append([acc_cost[i][j + 1], [i, j + 1], [i, j]])
    else:
        arr.append(math.inf)

    #(6) col - 1, row + 1
    if (j - 1 >= 0) and (i + 1 < lt_iw):
        g = c_anist_cost(i, j, i + 1, j - 1)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i + 1][j - 1] > (g + h)):
            parent[i + 1][j - 1] = i, j
            acc_cost[i + 1][j - 1] = (g + h)
        arr.append([acc_cost[i + 1][j - 1], [i + 1, j - 1], [i, j]])
    else:
        arr.append(math.inf)

    #(7) col, row + 1
    if (i + 1 < lt_iw):
        g = c_anist_cost(i, j, i + 1, j)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i + 1][j] > (g + h)):
            parent[i + 1][j] = i, j
            acc_cost[i + 1][j] = (g + h)
        arr.append([acc_cost[i + 1][j], [i + 1, j], [i, j]])
    else:
        arr.append(math.inf)

    #(8) col + 1, row + 1
    if (j + 1 < lt_jh) and (i + 1 < lt_iw):
        g = c_anist_cost(i, j, i + 1, j + 1)
        h = heuristic_func(i - 1, j - 1)

        if (acc_cost[i + 1][j + 1] > (g + h)):
            parent[i + 1][j + 1] = i, j
            acc_cost[i + 1][j + 1] = (g + h)
        arr.append([acc_cost[i + 1][j + 1], [i + 1, j + 1], [i, j]])
    else:
        arr.append(math.inf)
    

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
width, height = 20976, 20793

dem_data, mask_data = dem_data[:width,:height], mask_data[:width, :height]
print(dem_data.shape, mask_data.shape)

mask_copy = deepcopy(mask_data)
set_weights_vtr = np.vectorize(set_weights)
mask_copy = set_weights_vtr(mask_copy)

# print(np.unique(mask_copy, return_counts = True))

# SOURCE CO-ORDINATES LOAD
file = ogr.Open("source/source.shp")
source_shp = file.GetLayer(0)
feature = source_shp.GetFeature(0)
source_shp = feature.ExportToJson()
source_shp = json.loads(source_shp)
start_ext = source_shp['geometry']['coordinates']

# DESTINATION CO-ORDINATES LOAD
file = ogr.Open("destination/destination.shp")
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
start_pixel, end_pixel = [89827, 52184], [149597, 133916]
start_pixel = np.array(start_pixel)
start_pixel = start_pixel / 10

start_pixel = list(np.rint(start_pixel))
start_pixel = list(np.array(start_pixel, dtype = 'int'))

end_pixel = np.array(end_pixel)
end_pixel = end_pixel / 10

end_pixel = list(np.rint(end_pixel))
end_pixel = list(np.array(end_pixel, dtype = 'int'))

mask_copy_t = deepcopy(mask_copy)
mask_copy = mask_copy * 5000

# print(np.unique(mask_copy, return_counts = True))
print(start_pixel, end_pixel)

dim = mask_copy.shape
print(dim)
print(mask_copy.shape, dem_data.shape)


# MAIN PIXELS
# start_pixel, end_pixel = [8983, 5218], [14960, 13392]

# DUMMY PIXELS
start_pixel, end_pixel = [8983, 5218], [8755, 4856]
print(start_pixel, end_pixel)
print(width, height)

# LOOPING A*
for f in range(1, 6):

    start = timeit.default_timer()
    FLAG = f
    # A* main   
    #--------------------#####----------------------#
    Q = PriorityQueue()
    s_pixel, e_pixel = start_pixel, end_pixel

    # mask_copy.shape, dem_data.shape

    visited = np.zeros((dim))
    acc_cost = np.full((dim), math.inf)

    acc_cost[s_pixel[0]][s_pixel[1]] = 0
    visited[s_pixel[0]][s_pixel[1]] = 1

    parent = np.full((dim), None)
    parent[s_pixel[0]][s_pixel[1]] = -1, -1

    while (condition_check(s_pixel, e_pixel)):
    
        i, j = s_pixel
        print(i, j)
        # h-cost(0) - indicates Euclidean distance
        # h-cost(1) - indicates Manhattan distance

        neighbours_cost = get_neigh_cost(i, j, FLAG)
        for nc in neighbours_cost:
            if nc == math.inf:
                continue
            else:
                Q.put(nc)
        
        bst = Q.get()
        m, n = bst[1][0], bst[1][1]

        while True:
            if visited[m][n] == 0:
                break
            bst = Q.get()
            m, n = bst[1][0], bst[1][1]

        # set-visited
        visited[m][n] = 1
        # path.append([m, n])
        # print(m, n)

        parent[m][n] = bst[2][0], bst[2][1]

        s_pixel = [m, n]
    stop = timeit.default_timer()
    print(f'Execution time = {(stop - start) / 3600} hrs')
    #--------------------#####----------------------#
    # -- Path saving code
    #--------------------#####----------------------#
    print(acc_cost[end_pixel[0]][end_pixel[1]])

    # os.mkdir(f'paths/a-stars/P{FLAG}')
    # !mkdir f'paths/P{FLAG}'
    
    mkdir(f'paths/{route_paths[FLAG]}/')
    path = []
    pr = parent[end_pixel[0]][end_pixel[1]]

    # cnt = 15000
    while (pr[0] != -1) and (pr[1] != -1):
        path.append(pr)
        # path.append('-')
        with open(f'paths/{route_paths[FLAG]}/path.txt', 'a') as f:
            f.write(str(pr) + '\n')

        # cnt = cnt - 1
        # if cnt == 0:
        #     break

        pr = parent[pr[0]][pr[1]]
    
    path_list = []
    with open(f'paths/{route_paths[FLAG]}/path.txt', 'r') as f:
        for point in f:
            path_list.append(point)
    
    path_list.reverse()
    ordinates_dict = {}

    for i in range(len(path_list)):
        point = path_list[i].replace('\n', '').split(' ')
        pi, pj = int(point[0].split(',')[0].split('(')[1]), int(point[1].split(',')[0].split(')')[0])

        ext_i, ext_j = left + (pi * 10), top - (pj * 10)

        # ordinates_dict[(pi, pj)] = extent_matrix[pi][pj]
        ordinates_dict[pi, pj] = [ext_i, ext_j]
    
    # ordinates_dict
    ordinates_list = []
    for key, value in ordinates_dict.items():
        ordinates_list.append(value)
    
    # import shapefile
    w = shapefile.Writer(f'routes-shape/{route_paths[FLAG]}/shapefiles/test/multipoint')
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
    #--------------------#####----------------------#
