# RECOMMENDATION: EMBARRASSINGLY PARALLELIZE THE GIVEN UL_BR COORDS TO SPEED UP PROCESS

from pca_items.intrinsic_dimension import IntrinsicDimensionRMT as ID # this is a personal python file, not a library

from spectral.io import envi
from concurrent.futures import ThreadPoolExecutor
from osgeo import gdal
import pickle
import numpy as np
import argparse


def main():
    """
    Note:
    • 'reflectance_file' is a .pickle file with a list with lists of the reflectance files to refernce
        • i.e. [[xyz_2018.tif], [xyz_2018.tif, xyz_2019.tif], [xyz_2018.tif, xyz_2019.tif, xyz_2019.tif], ...]
        • all files in 'reflectance_file' should have same index spanse
        
    Returns:
    • List of dictionaries.
        • Index of each dictionary corresponds to the index of the reflectance_file
        • None values for ID are stored in the dictionary as -9999
        • the key to each dictionary is the ID, the values are coords (x, x+x_step, y, y+y_step)
    """
    parser = argparse.ArgumentParser(description='Spatial kernal for calculating the intrinsic dimensionality of a hyperspectral scene.')
    
    parser.add_argument('out_path', type=str) #specify the out file name too
    parser.add_argument('kernal_size', type=int, nargs=2) # x y
    parser.add_argument('reflectance_file', type=str)
    parser.add_argument('noise_file', type=bool, default=True) # pickle with noise array - !!TAKE OUT BAD REFLECTANCE VALUES PRIOR!!
    parser.add_argument('-normalize', type=bool, default=True)
    parser.add_argument('-valid_rfl_indices', type=str) # pickle with an np.ndarray of the reflectance wavelength values to use
    parser.add_argument('-ul_br', type=int, nargs=2) # 4 seperate integers of the index coordinate range (NOT UTM) i.e. 1000 500 900 510 (y1, x1, y2, x2)
    parser.add_argument('-replace_nan', type=str, nargs=1, choices=["median", "mean"])
    args = parser.parse_args()
    
    if ".pickle" not in args.out_path:
        raise Exception("'out_path' needs to end in .pickle")
    
    reflectance_file = pickle.load(open(args.reflectance_file, 'rb'))
    
    noise = pickle.load(open(args.noise_file, 'rb'))
    
    if args.valid_rfl_indices is not None:
        good_indices = pickle.load(open(args.valid_rfl_indices, 'rb'))
        if type(good_indices) == list:
            good_indices = np.array(good_indices)
        if type(good_indices) != np.ndarray:
            raise Exception("Valid Reflectance values need to be as a list or numpy.array")
    
    file = reflectance_file[0][0]
    raster = gdal.Open(file)
    max_y = raster.RasterYSize
    max_x = raster.RasterXSize
    if args.ul_br is not None:
        if args.ul_br[2] >= raster.RasterYSize:
            args.ul_br[2] = raster.RasterYSize - args.kernal_size[1]
            print("Trimming bottom coord - lower than possible extent")
        if args.ul_br[3] >= raster.RasterXSize:
            args.ul_br[2] = raster.RasterYSize - args.kernal_size[0]
            print("Trimming right coord - lower than possible extent")
    elif args.ul_br is None:
        args.ul_br = [0, 0, raster.RasterYSize-args.kernal_size[1], raster.RasterXSize-args.kernal_size[0]]
        
    # return list of dictionaries for space
    return_list = [{} for _ in range(len(reflectance_file))]
    
    # now run the ID, multithreading it
    x_step, y_step = args.kernal_size[0], args.kernal_size[1]
    for x in range(args.ul_br[1], args.ul_br[3], x_step):
        for y in range(args.ul_br[0], args.ul_br[2], y_step):
            
            file_reflectances = []
            for file_list in reflectance_file: 
                
                reflectances = []
                
                with ThreadPoolExecutor() as executer:
                    enter_list = [(i, file, x, x+x_step, y, y+y_step) for i, file in enumerate(file_list)]
                    results = executer.map(get_rfl, enter_list)

                    for result in results:
                        reflectances.append(result)
                reflectances = sorted(reflectances, key=lambda x: x[0])
                reflectances = [r for i,r in reflectances]
                
                complete_rfls = None:
                for rfl in :
                    if complete_rfls is None:
                        complete_rfls = rfl
                    else:
                        complete_rfls = np.concatenate([complete_rfls, rfl], axis=2)
                if args.valid_rfl_indices is not None:
                    complete_rfls = complete_rfls[:, :, good_indices]
                complete_rfls = np.reshape(complete_rfls, (complete_rfls.shape[0]*complete_rfls.shape[1], complete_rfls.shape[2]))
                
                if args.normalize:
                    complete_rfls = np.array([rfl/(sum([i**2 for i in rfl])**0.5) for rfl in complete_rfls])
                
                # now replace nan values if specified
                if np.isnan(complete_rfls).any() and args.replace_nan is None:
                    file_reflectances.append(np.array([]))
                    continue
                elif np.isnan(complete_rfls).any():
                    nan_indices = np.array([i for i in range(len(complete_rfls)) if True in np.isnan(complete_rfls)[i]])
                    if args.replace_nan.lower() == "median":
                        complete_rfls[nan_indices] = np.nanmedian(complete_rfls, axis=0)
                    elif args.replace_nan.lower() == "mean":
                        complete_rfls[nan_indices] = np.nanmean(complete_rfls, axis=0)
                        
                file_reflectances.append(complete_rfls)
                        
                    
            # now continue on with threading the ID
            id_values = []
            with ThreadPoolExecutor() as executer:
                enter_list = [(i, rfls, noise) for i, rfls in enumerate(file_reflectances)]
                results = executer.map(get_id, enter_list)
                
                for result in results:
                    id_values.append(result)
            id_values = sorted(id_values, key=lambda x: x[0])
            id_values = [_id for i, _id in id_values]
            
            #append these values to the hash table
            for i in range(len(id_values)):
                this_dict = return_list[i]
                this_id = id_values[i]
                
                if this_id not in list(this_dict.keys()):
                    this_dict[this_id] = [(x, x+x_step, y, y+y_step)]
                else:
                    this_dict[this_id].append((x, x+x_step, y, y+y_step))
                    
                return_list[i] = this_dict
                
    pickle.dump(return_list, open(, "wb"))


def get_rfl(_input):
    i, file, x_start, x_end, y_start, y_end = _input
    if ".hdr" not in file:
        file = file + ".hdr"
        
    rfl = envi.open(file).open_memmap(interleave='bip')[y_start:y_end, x_start:x_end, :].copy()
    rfl = np.transpose(rfl, (1,0,2))
    return (i, rfl)


def get_id(_input):
    i, rfls, noise = _input
    if len(rfls) == 0:
        return (i, -9999)
    
    needed_concatenations = int(rfls.shape[1]/noise.shape[0])
    this_noise = noise
    for j in range(needed_concatenations):
        this_noise = np.concatenate([this_noise, noise], axis=0)    
    this_n_est = {'noise_covariance':np.diag(this_noise), 'estimation_method': 'MultipleRegressionClassicNoiseEstimator'}
    
    instrinsic_dimension = ID()
    _id = instrinsic_dimension(rfls, this_n_est)
    return (i, _id['intrinsic_dimension'])


if __name__ == "__main__":
    main()