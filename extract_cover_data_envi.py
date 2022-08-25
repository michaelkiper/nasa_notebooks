


import gdal
import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
import subprocess
import glob
from spectral.io import envi
import ray


def main():
    parser = argparse.ArgumentParser(
        description='efficiently extract data from a vector file and multiple accompanying rasters')

    parser.add_argument('out_base', type=str)
    parser.add_argument('-all_shape_dir',type=str)
    parser.add_argument('-shape_dirs', nargs='+', type=str)
    parser.add_argument('-shp_attribute', type=str, default='id')
    parser.add_argument('-max_samples_per_class', type=int, default=20000)
    parser.add_argument('-source_files', nargs='+', type=str)
    args = parser.parse_args()

    args.source_files = [\
                         # '/beegfs/scratch/brodrick/col/mosaics/built_mosaic/min_phase_refl',
                         # '/beegfs/scratch/brodrick/col/mosaics/built_mosaic/min_phase_tch_me',
                         # '/beegfs/scratch/brodrick/col/mosaics/built_mosaic/min_phase_shade',
                         # '/beegfs/scratch/brodrick/col/mosaics/built_mosaic/min_phase_wtrl'
                         "/beegfs/scratch/makiper/Mosaics/flight_products/20220228/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220228/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220308/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220316/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220322/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220405/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220412/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220420/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220429/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220503/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220511/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220517/box_mosaics/box_rfl_phase_l2a",
                         # "/beegfs/scratch/makiper/Mosaics/flight_products/20220529/box_mosaics/box_rfl_phase_l2a",
                         # "/home/makiper/Data/surface_types/raster_series.vrt"
                        ]

    if args.all_shape_dir is not None:
        args.shape_dirs = glob.glob(os.path.join(args.all_shape_dir,'*'))

    shuffle_coords = True
    if args.max_samples_per_class == -1:
        args.max_samples_per_class=1e15
        shuffle_coords = False


    # Open / check all raster files.  Check is very cursory.
    print("Src files:", args.source_files)
    envi_sets = []
    for _f in range(len(args.source_files)):
        envi_ds = envi.open(args.source_files[_f] + '.hdr', image=args.source_files[_f])
        dat = envi_ds.open_memmap(interleave='bil',writeable=False)
        envi_sets.append(dat)
        print(envi_sets[-1].shape)


    n_features = 0
    for _f in range(len(envi_sets)):
        print(args.source_files[_f])
        assert envi_sets[_f] is not None, 'Invalid input file'
        if (envi_sets[_f].shape[-1] != envi_sets[0].shape[-1]):
        # if (envi_sets[_f].shape[-1] != envi_sets[1].shape[-1]):
            print('Raster X Size does not match, terminiating')
            quit()
        if (envi_sets[_f].shape[0] != envi_sets[0].shape[0]):
        # if (envi_sets[_f].shape[0] != envi_sets[1].shape[0]):
            print('Raster Y Size does not match, terminiating')
            quit()
        n_features += envi_sets[_f].shape[1]


    # trans = gdal.Open(args.source_files[1],gdal.GA_ReadOnly).GetGeoTransform()
    trans = gdal.Open(args.source_files[0],gdal.GA_ReadOnly).GetGeoTransform()
    namelist = []
    # init=' -te {} {} {} {} -tr {} {} -init -1 '.format(
    #                       trans[0],
    #                       trans[3]+trans[5]*envi_sets[1].shape[0],
    #                       trans[0]+trans[1]*envi_sets[1].shape[-1],
    #                       trans[3],
    #                       trans[1],
    #                       trans[5])
    init=' -te {} {} {} {} -tr {} {} -init -1 '.format(
                          trans[0],
                          trans[3]+trans[5]*envi_sets[0].shape[0],
                          trans[0]+trans[1]*envi_sets[0].shape[-1],
                          trans[3],
                          trans[1],
                          trans[5])

    cover_raster_file = os.path.join(args.out_base,'cover_raster_test_site_2.tif')
    if (os.path.isfile(cover_raster_file)):
        print('cover raster file already exists at {}, using'.format(cover_raster_file))
        for index, shape_dir in enumerate(args.shape_dirs):
            shape_files = glob.glob(shape_dir + '/*.geojson')
            dirnames = [x for x in shape_dir.split('/') if x != '']
            namelist.append(dirnames[-1])
    else:
        for index, shape_dir in enumerate(args.shape_dirs):

            shape_files = glob.glob(shape_dir + '/*.geojson')
            dirnames = [x for x in shape_dir.split('/') if x != '']
            namelist.append(dirnames[-1])
            print("SHAPE FILES:",shape_files,"\n")
            print("SHAPE DIR:",shape_dir,"\n")
            
            for shpfile in shape_files:
                cmd_str = 'gdal_rasterize {} {} -burn {} {}'.format(
                          shpfile,
                          cover_raster_file,
                          index,
                          init,
                          )
                print(cmd_str)
                subprocess.call(cmd_str, shell=True)
                init=''

    # Open binary cover file
    print("COVER FILE:",cover_raster_file,"\n")
    cover_set = gdal.Open(cover_raster_file, gdal.GA_ReadOnly)
    cover_trans = cover_set.GetGeoTransform()
    assert cover_set is not None, 'Invalid input file'

    # Get cover coordinates
    covers = cover_set.ReadAsArray()
    un_covers = np.unique(covers[covers != -1]).astype(int)

    coord_lists = []
    num_outputs = 0
    np.random.seed(13)
    for _cover, cover in enumerate(un_covers):

        cover_coords = list(np.where(covers == cover))
        if len(cover_coords[0]) > args.max_samples_per_class and shuffle_coords:
            perm = np.random.permutation(len(cover_coords[0]))[:args.max_samples_per_class]
            cover_coords[0] = cover_coords[0][perm]
            cover_coords[1] = cover_coords[1][perm]
        print(f'Starting covertype: {namelist[_cover],cover}, len: {len(cover_coords[0])}')


        coord_lists.append(cover_coords)
        num_outputs += len(cover_coords[0])


    # Read through files and grab relevant data
    output_array = np.zeros((num_outputs, n_features + 3))
    output_names = []

    ray.init()
    print(ray.cluster_resources())
    start_index = 0
    for _cover, cover in enumerate(un_covers):

        cover_coords = coord_lists[_cover]
        print(f'Starting covertype: {_cover}/{len(un_covers)},{namelist[_cover],cover}, len: {len(cover_coords[0])}')
        for _line in range(len(cover_coords[0])):

            output_array[start_index + _line, 0] = covers[cover_coords[0][_line], cover_coords[1][_line]]
            output_array[start_index + _line, 1] = cover_coords[1][_line]*cover_trans[1]+cover_trans[0]
            output_array[start_index + _line, 2] = cover_coords[0][_line]*cover_trans[5]+cover_trans[3]

            output_names.append(namelist[_cover])

        feat_ind = 3
        for _f in range(len(envi_sets)):
            jobs = []
            for _line in range(len(cover_coords[0])):
                jobs.append(read_data_piece.remote(args.source_files[_f], cover_coords[0][_line], cover_coords[1][_line], _line)) 
                #if _line % 100 == 0:
                #    print('set: {}/{}, line: {}/{}'.format(_f,len(envi_sets),_line,len(cover_coords[0])))
                #point = envi_sets[_f][int(cover_coords[0][_line]), :, int(cover_coords[1][_line])]
                #output_array[start_index + _line, feat_ind:feat_ind+envi_sets[_f].shape[1]] = np.squeeze(point)
                rreturn = [ray.get(jid) for jid in jobs]
                for ind, point in rreturn:
                    output_array[start_index + ind, feat_ind:feat_ind+envi_sets[_f].shape[1]] = np.squeeze(point)
 
            feat_ind += envi_sets[_f].shape[1]

        start_index += len(cover_coords[0])

    output_names = np.array(output_names)

    # Export
    header = ['ID', 'X_UTM', 'Y_UTM',]
    for _f in range(len(envi_sets)):
        header.extend([os.path.splitext(os.path.basename(args.source_files[_f]))[0]
                       [-4:] + '_B_' + str(n+1) for n in range(envi_sets[_f].shape[1])])
    out_df = pd.DataFrame(data=output_array, columns=header)
    out_df['covertype'] = output_names
    out_df.to_csv(os.path.join(args.out_base,'cover_extraction_test_site_2.csv'),sep=',', index=False)


@ray.remote
def read_data_piece(filename, line, sample, idx):

    if idx % 100 == 0:
        print('set: {}, idx: {}'.format(filename,idx))

    envi_ds = envi.open(filename + '.hdr', image=filename)
    dat = envi_ds.open_memmap(interleave='bil',writeable=False)
    point = np.array(dat[int(line), :, int(sample)].copy())

    return idx, point
    

if __name__ == "__main__":
    main()
