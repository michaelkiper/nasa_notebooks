from visuals import WAVELENGTHS
from pca_helper import *
import os
import numpy as np
import pickle
import os
from osgeo import gdal
import sys
from spectral.io import envi

def pixel_geo_coords(dx,dy, raster):
    px = raster.GetGeoTransform()[0]
    py = raster.GetGeoTransform()[3]
    rx = raster.GetGeoTransform()[1]
    ry = raster.GetGeoTransform()[5]
    x = dx*rx + px
    y = dy*ry + py
    return x,y

import concurrent

def pixel_geo_coords(dx,dy, raster):
    px = raster.GetGeoTransform()[0]
    py = raster.GetGeoTransform()[3]
    rx = raster.GetGeoTransform()[1]
    ry = raster.GetGeoTransform()[5]
    x = dx*rx + px
    y = dy*ry + py
    return x,y

def get_coords(file):
    raster = gdal.Open(file)
    # raster_envi = envi.open(file + '.hdr').open_memmap(interleave='bip')

    step_size = 50
    valid_array = []
    index_coords = []
    utm_coords = []
    for x in range(0, raster.RasterXSize-step_size, step_size):
        for y in range(0, raster.RasterYSize-step_size, step_size):
            arr = raster.ReadAsArray(xoff=x, yoff=y, xsize=step_size, ysize=step_size)
            # arr = raster_envi[y:y+step_size,x:x+step_size,:].copy()
            try:
                if abs(sum(sum(arr))) != step_size*step_size:

                    index_coords.append({'xoff':x, 'yoff':y, 'step':step_size})
                    upper_left_coords, lower_right_coords = pixel_geo_coords(x, y, raster), pixel_geo_coords(x+step_size, y+step_size, raster)
                    utm_coords.append((upper_left_coords, lower_right_coords))
                    valid_array.append(arr)
            except Exception:
                raise Exception
                
    return index_coords, valid_array


def thread_rfl(map_indices):
    start_index, end_index = map_indices['indices']
    index_coords = map_indices['index_coords']
    valid_array = map_indices['valid_array']
    # gdals = map_indices['gdals']
    file_names = map_indices['file_names']
    
    dates = ['20220224', '20220228', '20220308', '20220316', '20220322', '20220405', '20220412', '20220420', '20220429', '20220503', '20220511', '20220517', '20220529']
    this_complete_pixels = []
    step_size = 50
    
    complete_pixels = dict(zip(dates, [np.array([]) for d in dates]))
    for i, coords in enumerate(index_coords[start_index: end_index]):
        i = i + start_index
        
        v = valid_array[i]
        v = np.transpose(v, (1,0))
        
        valid_indices = np.where(v == 0)
        
        rfls = []
        for j, date in enumerate(dates):
            rfl = envi.open(file_names[j] + '.hdr').open_memmap(interleave='bip')[coords['yoff']:coords['yoff']+step_size,coords['xoff']:coords['xoff']+step_size,:].copy()
            # print('Obtined RFL:', date, i)
            rfl = np.transpose(rfl, (1,0,2))
            # temp_rfl = rfl.copy()

            
            rfl = rfl[valid_indices]

            if len(complete_pixels[date]) == 0:
                complete_pixels[date] = rfl
            else:
                complete_pixels[date] = np.concatenate([complete_pixels[date], rfl], axis=0)
            rfls.append(rfl)

        print(str(i)+"/"+str(len(index_coords)))
        
    return complete_pixels


def get_rfls(file):
    
    index_coords, valid_array = get_coords(file)
    print("Obtained Coords")
    
    dates = ['20220224', '20220228', '20220308', '20220316', '20220322', '20220405', '20220412', '20220420', '20220429', '20220503', '20220511', '20220517', '20220529']
    file_names = [f"/beegfs/scratch/makiper/Mosaics/flight_products/{date}/box_mosaics/box_rfl_phase_l2a" for date in dates]
    complete_pixels = []

    # gdals = []
    # for date in dates:
    #     gdals.append(gdal.Open(f"/beegfs/scratch/makiper/Mosaics/flight_products/{date}/box_mosaics/box_rfl_phase_l2a"))
        
    with concurrent.futures.ThreadPoolExecutor() as executer:
        index_step = 2
        indices = [(index, index+index_step if index+index_step < len(index_coords) else len(index_coords)) for index in range(0, len(index_coords), index_step)]
        print(indices)
        # enter_list = [{'indices':i, 'index_coords':index_coords, 'valid_array':valid_array, 'gdals':gdals} for i in indices]
        enter_list = [{'indices':i, 'index_coords':index_coords, 'valid_array':valid_array, 'file_names':file_names} for i in indices]
        
        results = executer.map(thread_rfl, enter_list)

        for result in results:
            complete_pixels.append(result)
        
    return complete_pixels        
        

file = "/home/makiper/Clustering/v4/cover_raster_unsupervised_test.tif"
raster = gdal.Open(file)

step_size = 50
valid_array = []
index_coords = []
utm_coords = []
for x in range(0, raster.RasterXSize-step_size, step_size):
    for y in range(0, raster.RasterYSize-step_size, step_size):
        arr = raster.ReadAsArray(xoff=x, yoff=y, xsize=step_size, ysize=step_size)
        try:
            if abs(sum(sum(arr))) != step_size*step_size:
                # TODO: DFS alg on extracting exact shape
                
                index_coords.append({'xoff':x, 'yoff':y, 'step':step_size})
                upper_left_coords, lower_right_coords = pixel_geo_coords(x, y, raster), pixel_geo_coords(x+step_size, y+step_size, raster)
                utm_coords.append((upper_left_coords, lower_right_coords))
                valid_array.append(arr)
        except Exception:
            raise Exception
    
y, dy = index_coords[0]['yoff'], index_coords[0]['yoff']+index_coords[0]['step']
x, dx = index_coords[0]['xoff'], index_coords[0]['xoff']+index_coords[0]['step']

file = "/beegfs/scratch/makiper/Mosaics/flight_products/20220224/box_mosaics/box_rfl_phase_l2a"
rfl = envi.open(file + '.hdr').open_memmap(interleave='bip')[y:dy, x:dx, :].copy()
rfl = np.transpose(rfl, (1,0,2))


static_lots_rfls = get_rfls("/home/makiper/Clustering/v4/cover_raster_unsupervised_test.tif")

total_rfls = {}

for item in static_lots_rfls:
    for date in list(static_lots_rfls[0].keys()):
        if date not in total_rfls.keys():
            total_rfls[date] = item[date]
        else:
            total_rfls[date] = np.concatenate([total_rfls[date], item[date]], axis = 0)
            
# bad_pix = dict(zip([date for date in total_rfls.keys()], [[] for d in total_rfls.keys()]))
# for date in total_rfls.keys():
#     this_date = []
#     for i, rfl in enumerate(total_rfls[date]):
#         if -9999 in list(rfl):
#             bad_pix[date].append(i)
#             this_date.append(i)
#     print(date, len(this_date))

# total_bad_pix = []
# for d in bad_pix.keys():
#     these_i = bad_pix[d]
#     for p in these_i:
#         if p not in total_bad_pix:
#             total_bad_pix.append(p)
# total_bad_pix.sort()


# total_rfls_corrected = {}
# for date in total_rfls.keys():
#     bad_indices = bad_pix[date]
#     good_indces = [i for i in range(total_rfls['20220224'].shape[0]) if i not in total_bad_pix]
#     total_rfls_corrected[date] = np.array(total_rfls[date])[good_indces]
    

with open("/home/makiper/Clustering/v3/unsupervised_test_rfls.pickle", "wb") as f:
    pickle.dump(total_rfls, f)


