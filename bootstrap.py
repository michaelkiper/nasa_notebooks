from visuals import WAVELENGTHS
from visuals import *
from pca_helper import *
from pca_items.intrinsic_dimension import IntrinsicDimensionRMT as ID
import os
import numpy as np
import pickle
import pandas as pd
import argparse
import concurrent


def geocoords(file):
    transform = file.GetGeoTransform()

    x_origin = transform[0]
    y_origin = transform[3]
    pixel_width = transform[1]
    pixel_height = -transform[5]
    return x_origin, y_origin, pixel_width, pixel_height

def geotransform(geocoords, x, y):
    x_origin, y_origin, pixel_width, pixel_height = geocoords
    return int((x - x_origin)/pixel_width), int((y_origin - y)/pixel_height)

gdals = []
all_gcoords = []
dates = ['20220224', '20220228', '20220308', '20220316', '20220322', '20220405', '20220412', '20220420', '20220429', '20220503', '20220511', '20220517', '20220529']
for date in dates:
    f = gdal.Open(f"/beegfs/scratch/makiper/Mosaics/flight_products/{date}/box_mosaics/box_rfl_phase")
    gcoords = geocoords(f)
    gdals.append(f)
    all_gcoords.append(gcoords)

def get_rfls(dataframe):
    num_coords = len(dataframe.index)
    step = 3
    
    date_dict = dict(zip(dates, [[] for d in dates]))

    for pix in range(num_coords):
        this_x_utm, this_y_utm = dataframe.iloc[pix]['center_x_utm'], dataframe.iloc[pix]['center_y_utm']
        print("%: ", pix/num_coords*100)

        for i, date in enumerate(dates):
            f = gdals[i]
            gcoords = all_gcoords[i]

            x, y = geotransform(gcoords, this_x_utm, this_y_utm)


            rfl = f.ReadAsArray(xoff=x-1, yoff=y+1, xsize=step, ysize=step)
            rfl = index_reshape(rfl)

            for j in range(step):
                for k in range(step):
                    date_dict[date].append(rfl[j][k])
    return date_dict
                    



def indicies_choice(c:int):
    band_ranges = [(1340, 1460), (1800, 2050), 2450, 400]
    wl = np.array(WAVELENGTHS)
    
    if c == 1:
        # take out band noise bands
        cond_1 = ((wl > band_ranges[0][0]) & (wl < band_ranges[0][1]))
        cond_2 = ((wl > band_ranges[1][0]) & (wl < band_ranges[1][1]))
        cond_3 = (wl > band_ranges[2])
        cond_4 = (wl < band_ranges[3])
        indicies = np.where(cond_1 | cond_2 | cond_3 | cond_4)
        return indicies
    
    elif c == 2:
        cond_1_2 = ((wl > 1760) & (wl < 1840))
        cond_2_2 = ((wl > 1480) & (wl < 1760))
        cond_3_2 = ((wl > 1930) & (wl < 2450))
        cond_4_2 = (wl < 1280)
        indicies_2 = np.where(cond_1_2 | cond_2_2 | cond_3_2 | cond_4_2)
        return indicies_2


def noise_matrix_choice(c:int, indicies):
    if c == 1:
        noise = open("/home/makiper/Notebooks/SHIFT_noise_M_allbands.txt", "rb").read()
        noise = str(noise).replace("b","").replace("'", "")
        noise = np.array([float(i) for i in str(noise).split(",")])

        # take out band noise bands
        corrected_noise = np.delete(noise, indicies)
        return corrected_noise
    
    elif c == 2:
        noise = np.genfromtxt('/home/makiper/Notebooks/RMT_debugging_N.csv', delimiter=',')
        corrected_noise = np.array([noise[i][i] for i in range(len(noise))])
        return corrected_noise
    
    
def backup_id(series_list, corrected_noise):
    ids = []
    for i, series in enumerate(series_list):
        print(f"Starting Series: {i+1}")
        this_mat = np.array(series)

        this_noise = corrected_noise
        for j in range(i):
            this_noise = np.concatenate([this_noise, corrected_noise], axis=0)    
        this_n_est = {'noise_covariance':np.diag(this_noise), 'estimation_method': 'MultipleRegressionClassicNoiseEstimator'} 

        instrinsic_dimension = ID()
        _id = instrinsic_dimension(this_mat, this_n_est)
        print(f"Series: {i+1} ID: {_id['intrinsic_dimension']}")
        ids.append(_id['intrinsic_dimension'])
    return ids


def get_series(s, c, indicies):
    if s.lower() == 'parking_lot':
        parking_lot_dict = pickle.load(open("/home/makiper/Notebooks/outliers/parking_lot_items.pickle", 'rb'))
        
        parking_lot_reflectances = {}
        for date in list(parking_lot_dict.keys()):
            valid_pixels = [parking_lot_dict[date]['pixels'][p] for p in range(len(parking_lot_dict[date]['pixels'])) if p not in parking_lot_dict[date]['outliers']]
            parking_lot_reflectances[date] = valid_pixels
            
        num_dates = len(parking_lot_dict.keys())
        parking_lot_dates = [create_raw_time_series(parking_lot_reflectances, i, remove_bands=False) for i in range(num_dates)] # water bands already removed
        parking_lot_series = []
        remove_bands = True

        for dates, mean_freq in parking_lot_dates:
            master = None
            for d in dates:
                ds = parking_lot_dict[d]['pixels']
                if remove_bands:
                    if c == 1:
                        ds = remove_bands_f(ds)
                    elif c == 2:
                        ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX

                if master is None:
                    master = ds
                else:
                    master = np.concatenate([master, ds], axis=1)
            parking_lot_series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
            
        return parking_lot_series
    
    elif s.lower() == 'roof_top':
        roof_top_dict = pickle.load(open("/home/makiper/Notebooks/outliers/roof_top_items.pickle", 'rb'))
        
        roof_top_reflectances = {}
        for date in list(roof_top_dict.keys()):
            valid_pixels = [roof_top_dict[date]['pixels'][p] for p in range(len(roof_top_dict[date]['pixels'])) if p not in roof_top_dict[date]['outliers']]
            roof_top_reflectances[date] = valid_pixels
            
        num_dates = len(roof_top_reflectances.keys())
        roof_top_dates = [create_raw_time_series(roof_top_reflectances, i, remove_bands=False) for i in range(num_dates)] # water bands already removed
            
        roof_top_series = []
        remove_bands = True

        for dates, mean_freq in roof_top_dates:
            master = None
            for d in dates:
                ds = roof_top_dict[d]['pixels']
                if remove_bands:
                    if c == 1:
                        ds = remove_bands_f(ds)
                    elif c == 2:
                        ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX

                if master is None:
                    master = ds
                else:
                    master = np.concatenate([master, ds], axis=1)
            roof_top_series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
            
        return roof_top_series
        
    elif s.lower() == 'rock':
        rock_dict = pickle.load(open("/home/makiper/Notebooks/outliers/rock_items.pickle", 'rb'))
        
        rock_reflectances = {}
        for date in list(rock_dict.keys()):
            valid_pixels = [rock_dict[date]['pixels'][p] for p in range(len(rock_dict[date]['pixels'])) if p not in rock_dict[date]['outliers']]
            rock_reflectances[date] = valid_pixels
            
        num_dates = len(rock_dict.keys())
        rock_dates = [create_raw_time_series(rock_reflectances, i, remove_bands=False) for i in range(num_dates)] # water bands already removed
        
        rock_series = []
        remove_bands = True

        for dates, mean_freq in rock_dates:
            master = None
            for d in dates:
                ds = rock_dict[d]['pixels']
                if remove_bands:
                    if c == 1:
                        ds = remove_bands_f(ds)
                    elif c == 2:
                        ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX

                if master is None:
                    master = ds
                else:
                    master = np.concatenate([master, ds], axis=1)
            rock_series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
            
        return rock_series
    
    elif s.lower() == 'grassland':
        veg_data = pd.read_csv('/home/makiper/Notebooks/export_for_mk.csv')
        veg_types = np.unique(veg_data['Plot Type'])
        grassland = veg_data[veg_data['Plot Type'] == 'Grassland']
        
        grassland_rfls = get_rfls(grassland)
        num_dates = len(grassland_rfls.keys())
        grassland_dates = [create_raw_time_series(grassland_rfls, i, remove_bands=False) for i in range(num_dates)] # water bands already removed
        
        grassland_series = []
        remove_bands = True

        for dates, mean_freq in grassland_dates:
            master = None
            for d in dates:
                ds = np.array(grassland_rfls[d])
                if remove_bands:
                    if c == 1:
                        ds = remove_bands_f(ds)
                    elif c == 2:
                        ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX

                if master is None:
                    master = ds
                else:
                    master = np.concatenate([master, ds], axis=1)
            grassland_series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
        
        return grassland_series
            
    elif s.lower() == 'shrub':
        veg_data = pd.read_csv('/home/makiper/Notebooks/export_for_mk.csv')
        veg_types = np.unique(veg_data['Plot Type'])
        shrub = veg_data[veg_data['Plot Type'] == 'Shrub']
        
        shrub_rfls = get_rfls(shrub)
        num_dates = len(shrub_rfls.keys())
        shrub_dates = [create_raw_time_series(shrub_rfls, i, remove_bands=False) for i in range(num_dates)] # water bands already removed
        
        shrub_series = []
        remove_bands = True

        for dates, mean_freq in shrub_dates:
            master = None
            for d in dates:
                ds = np.array(shrub_rfls[d])
                if remove_bands:
                    if c == 1:
                        ds = remove_bands_f(ds)
                    elif c == 2:
                        ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX

                if master is None:
                    master = ds
                else:
                    master = np.concatenate([master, ds], axis=1)
            shrub_series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
            
        return shrub_series
    
    elif s.lower() == 'tree':
        veg_data = pd.read_csv('/home/makiper/Notebooks/export_for_mk.csv')
        veg_types = np.unique(veg_data['Plot Type'])
        tree = veg_data[veg_data['Plot Type'] == 'Tree']
        
        tree_rfls = get_rfls(tree)
        num_dates = len(tree_rfls.keys())
        tree_dates = [create_raw_time_series(tree_rfls, i, remove_bands=False) for i in range(num_dates)] # water bands already removed
        
        tree_series = []
        remove_bands = True

        for dates, mean_freq in tree_dates:
            master = None
            for d in dates:
                ds = np.array(tree_rfls[d])
                if remove_bands:
                    if c == 1:
                        ds = remove_bands_f(ds)
                    elif c == 2:
                        ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX

                if master is None:
                    master = ds
                else:
                    master = np.concatenate([master, ds], axis=1)
            tree_series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
        
        return tree_series
        

def main():
    parser = argparse.ArgumentParser(
        description='Efficiently compute the intrinsic dimension of a series of matrices. Returns a pickle file.')
    parser.add_argument('series_type', type=str, help='type of series to generate')
    parser.add_argument('noise_choice', type=int, help='choise of noise matrix. either 1 or 2', default=2)
    parser.add_argument('-output_path', type=str, default=".")
    args = parser.parse_args()
    
    c = int(args.noise_choice)
    
    if not os.path.isdir(args.output_path):
        print("Could not find output path, changing to CWD")
        args.output_path = "."
    
    num_bootstrap = 30
    batch_size = 500
    
    indicies = indicies_choice(c)
    noise = noise_matrix_choice(c, indicies)
    
    series = get_series(args.series_type, c, indicies)
    
    # remove NaNs/-9999
    nan_inds = []
    for i in range(series[-1]['data'].shape[0]):
        if -9999 in list(series[-1]['data'][i]):
            nan_inds.append(i)
    nan_inds
    
    for i, s in enumerate(series):
        this_mat = np.delete(np.array(s['data']), nan_inds, axis=0)
        series[i]['data'] = this_mat
    
    total_ids = []
    for r in range(num_bootstrap):
        sample_indicies = [np.random.randint(0, series[0]['data'].shape[0]) for i in range(batch_size)] #sample with replacement
        data = [d['data'][sample_indicies] for d in series]
        
        this_id = backup_id(data, noise)
        total_ids.append(this_id)
        
    print("Finished Bootstrap")
    
    out_path = os.path.join(args.output_path, str(args.series_type)+"_bootstrap.pickle")
    
    try:
        with open(out_path, 'wb') as f:
            pickle.dump(total_ids, f)
            
    except Exception:
        with open(str(args.series_type)+"_bootstrap.pickle", 'wb') as f:
            pickle.dump(total_ids, f)


if __name__ == "__main__":
    main()

