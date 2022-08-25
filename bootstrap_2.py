import argparse
from pca_items.intrinsic_dimension import IntrinsicDimensionRMT as ID
from visuals import WAVELENGTHS
from pca_helper import *
import os
import numpy as np
import pickle
import concurrent


def thread_id(s):
    i, series = s[0], s[1]
    print("Starting Series: ", i+1)
    this_mat = np.array(series['data'])

    this_noise = corrected_noise
    for j in range(i):
        this_noise = np.concatenate([this_noise, corrected_noise], axis=0)    
    this_n_est = {'noise_covariance':np.diag(this_noise), 'estimation_method': 'MultipleRegressionClassicNoiseEstimator'} 
    
    instrinsic_dimension = ID()
    _id = instrinsic_dimension(this_mat, this_n_est)
    print(f"Series: {i+1} ID: {_id['intrinsic_dimension']}")
    return (i, _id)


def thread_series(series_dict):
    id_series = []
    with concurrent.futures.ThreadPoolExecutor() as executer:
        series = [(i, s) for i, s in enumerate(series_dict)]

        results = executer.map(thread_id, series)

        for result in results:
            id_series.append(result)
            
        s = sorted(id_series, key=lambda x: x[0])
        r = [j for i,j in s]
        return r


# wl = np.array(WAVELENGTHS)
# cond_1_2 = ((wl > 1760) & (wl < 1840))
# cond_2_2 = ((wl > 1480) & (wl < 1760))
# cond_3_2 = ((wl > 1930) & (wl < 2450))
# cond_4_2 = (wl < 1280)
# indicies = np.where(cond_1_2 | cond_2_2 | cond_3_2 | cond_4_2)

# noise = open("/home/makiper/Notebooks/SHIFT_noise_M_allbands_2.txt", "rb").read()
# noise = str(noise).replace("b","").replace("'", "")
# noise = np.array([float(i) for i in str(noise).split(",") if i != ''])
# band_ranges = [(1340, 1460), (1800, 2050), 2450, 400]
# wl = np.array(WAVELENGTHS)
# cond_1 = ((wl > band_ranges[0][0]) & (wl < band_ranges[0][1]))
# cond_2 = ((wl > band_ranges[1][0]) & (wl < band_ranges[1][1]))
# cond_3 = (wl > band_ranges[2])
# cond_4 = (wl < band_ranges[3])
# indicies = np.where(cond_1 | cond_2 | cond_3 | cond_4)
# corrected_noise = np.delete(noise, indicies)
with open("/home/makiper/Notebooks/mining_noise_2.pickle", "rb") as f:
    m_noise = pickle.load(f)
corrected_noise = np.diag(m_noise)


def create_series(dates_list, series_dict):
    
    # return the bad indices too
    bad_indices = []
    for date in series_dict.keys():
        for i, rfl in enumerate(series_dict[date]):
            if i not in bad_indices and -9999 in list(rfl):
                bad_indices.append(i)
    bad_indices.sort()
    
    series = []
    remove_bands = True

    for dates, mean_freq in dates_list:
        master = None
        for d in dates:
            ds = series_dict[d]
            if remove_bands:
                ds = remove_bands_f(ds) 
                # ds = ds[:,indicies[0]] # IMPORTANT REMOVE THIS WHEN NOT RUNNING ON NWE NOISE MATRIX
                
                norm = True # probably best to nromalize the vectors
                if norm:
                    ds = np.array([j/(sum([i**2 for i in j])**0.5) for j in ds])

            if master is None:
                master = ds
            else:
                master = np.concatenate([master, ds], axis=1)
        series.append({'dates':dates, 'mean_frequency':mean_freq, 'data':master})
    
    return series, bad_indices


def nan_indicies(series):
    nan_inds = []
    for i in range(series[-1]['data'].shape[0]):
        if -9999 in list(series[-1]['data'][i]):
            nan_inds.append(i)
    return nan_inds


def remove_nans(series, nan_inds):
    # nan_inds = nan_indicies(series)
    
    for s in series:
        s['data'] = np.delete(np.array(s['data']), nan_inds, axis=0)
        
    return series


# noise = np.genfromtxt('/home/makiper/Notebooks/RMT_debugging_N.csv', delimiter=',')
# corrected_noise = np.array([noise[i][i] for i in range(len(noise))])


with open("/home/makiper/Notebooks/true_parking_lot_rfls.pickle", "rb") as f:
    temp_dict = pickle.load(f)

num_dates = len(temp_dict.keys())
series_dates = [create_raw_time_series(temp_dict, i, remove_bands=False) for i in range(num_dates)] # water bands already removed

    
def main():
    parser = argparse.ArgumentParser(
        description='Efficiently compute the intrinsic dimension of a series of matrices. Returns a pickle file.')
    parser.add_argument('series_pickle', type=str, help='path to pickle file containing a list of matrices to compute')
    parser.add_argument('-out_base', type=str, help='output file path', default='.')
    parser.add_argument('-type', type=str, help='identifier for creating output file')
    args = parser.parse_args()
    
    if not os.path.isdir(args.out_base):
        print(f"Could not find out_base: '{args.out_base}' - changing out_base to CWD")
        args.out_base = "."
    
    if not os.path.isfile(args.series_pickle):
        raise Exception("'series_pickle' file not found: ", args.series_pickle)
        
    with open(args.series_pickle, "rb") as f:
        series_dict = pickle.load(f)
                
    if args.type is not None:
        output_path = os.path.join(args.out_base, args.type + "_mining_noise.pickle")
    else:
        output_path = os.path.join(args.out_base, str(dt.now().strftime("%m/%d/%Y_%H:%M:%S")) + "_mining_noise.pickle")
            
    series, bad_indices = create_series(series_dates, series_dict)
    series_corrected = remove_nans(series, bad_indices)
    print("\n SHAPE:", series_corrected[0]['data'].shape,"\n")
    
    # now bootstrap
    num_batches = 30
    batch_size = 500
    # batch_size = 100000
    
    batch_ids = []
    for b in range(num_batches):
        sample_indicies = [np.random.randint(0, series_corrected[0]['data'].shape[0]) for i in range(batch_size)]
        
        this_s = []
        for s in series_corrected:
            data = np.array([s['data'][i] for i in sample_indicies])
            this_s.append({'dates':s['dates'], 'mean_frequency':s['mean_frequency'], 'data':data})
            
        output_ids = thread_series(this_s)
        output_ids = [i['intrinsic_dimension'] for i in output_ids]
        batch_ids.append(output_ids)
    # output_ids = backup_id(series)
    
    print("Finished ID")
    
    with open(output_path, "wb") as f:
        pickle.dump(batch_ids, f)
    

if __name__ == "__main__":
    main()
    
    
        
    

