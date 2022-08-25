import numpy as np
import dateutil.parser as dparser
from datetime import timedelta
from visuals import *

def create_raw_time_series(reflectances:dict, n_dates:int, remove_bands=True):
    keys = list(reflectances.keys())
    
    if n_dates < 0:
        raise Exception("Cannot have negative 'n_dates'")
        
    if n_dates > len(keys):
        print("Maximum number of dates is "+str(len(keys))+". Converting n_dates to "+str(len(keys))+".")
        n_dates = len(keys)
    
    # concatinate the lists
    master = None
    dates_used = []
    datetimes = []
    
    # find the lowest mean distances
    datetime_list = [dparser.parse(i) for i in keys]
    min_date = min(datetime_list)

    datetimes.append(min_date)

    datetime_list.pop(0)

    print("  Real Date  |  Desired Optimal Date  |  Difference")
    print(min_date, "|", min_date, "|", 0)
    if n_dates >= 1:
        sep = ((max(datetime_list)-min_date).days/n_dates)

        optimal_separated_dates = [min_date + timedelta(days=(int((sep*(i+1))))) for i in range(n_dates)]

        # now find the dates that are closest to these            
        for op_date in optimal_separated_dates:

            op_real_date_index = 0

            for i, d in enumerate(datetime_list):
                old_delta = abs((op_date - datetime_list[op_real_date_index]).days)
                new_delta = abs((op_date - d).days)

                if new_delta < old_delta:
                    op_real_date_index = i

            # pop from datetime_list once used
            op_real_date = datetime_list[op_real_date_index]
            print(op_real_date, "|", op_date, "|", abs((op_real_date-op_date).days))

            datetimes.append(op_real_date)

            datetime_list.pop(op_real_date_index)
                
    # print("\nLeft Out: ", datetime_list, "\n")

    # now concatinate all of the dates to the master list
    match = [dparser.parse(d) in datetimes for d in keys]
    dates_used = np.array(keys)[match]

#     for date in dates_used:
#         ds = np.array(reflectances[date])
        
#         if remove_bands:
#             ds = remove_bands_f(ds)
            
#         if master is None:
#             master = ds
#         else:
#             ax = ds.shape[-1]
#             master = np.concatenate([master, ds], axis=1)
                
#     print("Shape: ",master.shape, "\n")
    
    mean_frequency = np.mean([abs((datetimes[i]-datetimes[i+1]).days) for i in range(len(datetimes)-1)])
    
    # reutrn_dict = {"dates": dates_used, "data":master, "mean_frequency": mean_frequency}
    # return reutrn_dict
    return dates_used, mean_frequency

def remove_bands_f(ds:np.array):
    band_ranges = [(1340, 1460), (1800, 2050), 2450, 400]
    wl = np.array(WAVELENGTHS)

    cond_1 = ((wl > band_ranges[0][0]) & (wl < band_ranges[0][1]))
    cond_2 = ((wl > band_ranges[1][0]) & (wl < band_ranges[1][1]))
    cond_3 = (wl > band_ranges[2])
    cond_4 = (wl < band_ranges[3])
    indicies = np.where(cond_1 | cond_2 | cond_3 | cond_4)[0]
    
    new_ds = []
    for x in range(ds.shape[0]):
        # row = []
        # for y in range(ds.shape[1]):
        #     row.append(list(np.delete(ds[x][y], indicies)))
        row = list(np.delete(ds[x], indicies))
        new_ds.append(row)
    return np.array(new_ds)


