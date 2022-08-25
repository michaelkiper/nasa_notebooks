from typing import Any, List, Dict, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd
from utils import (listify, tupleify)

class SpectrumData(object):
    '''
    Data holder for X_<nm>, F_<name>, y and related uncertainties 
    Xstd_<nm>, Fstd_<name>, ystd
    Used to hold subset of a FrameDataset corresponding to a split.
    Used to hold a reshaped HSI image as table.
    '''
    def __init__(self, df:pd.DataFrame,
                       details:Dict={}) -> None:
        '''
        `df`: DataFrame 
              Spectra columns are of the form `X_<nm>`      
              Spectra uncertainty columns are of the form `Xstd_<nm>`
              
              Feature columns are of the form `F_<descriptor>`
              Feature uncertainty columns are of the form `Fstd_<descriptor>`
              
              Target column is named 'y'.
              Target uncertainty is named 'ystd'.
              
        `details`: Dict
                   Example keys: 'from_image', 'image_shape'
        '''
        self.df = df.copy()
        self.details = {**details}
        if 'data_key' not in self.details:
            self.details['data_key'] = ''

        # measurements
        self.X_cols = [c for c in self.df.columns if ('X_' in c)]
        if len(self.X_cols) == 0: 
            self.X_cols = None
        self.Xstd_cols = [c for c in self.df.columns if ('Xstd_' in c)]
        if len(self.Xstd_cols) == 0: 
            self.Xstd_cols = None
            
        # features    
        self.F_cols = [c for c in self.df.columns if ('F_' in c)]
        if len(self.F_cols) == 0: 
            self.F_cols = None
        self.Fstd_cols = [c for c in self.df.columns if ('Fstd_' in c)]
        if len(self.Fstd_cols) == 0: 
            self.Fstd_cols = None
        
        # targets    
        self.y_col = 'y' if 'y' in self.df.columns else None
        self.ystd_col = 'ystd' if 'ystd' in self.df.columns else None 
        # others
        accounted_cols = []
        for cols in [self.X_cols, self.Xstd_cols,
                     self.F_cols, self.Fstd_cols,
                     self.y_col, self.ystd_col]:
            if cols is not None:
                accounted_cols += listify(cols)
        self.other_cols = [c for c in self.df.columns if c not in accounted_cols]
        
    def data_from_columns(self, columns:List[str]) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, len(columns)).
        '''
        return self.df[columns].values 
    
    def X(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, n_wavelengths).
        '''
        if not self.X_cols:
            return None
        return self.df[self.X_cols].values 
    
    def Xstd(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, n_wavelengths).
        '''
        if not self.Xstd_cols:
            return None
        return self.df[self.Xstd_cols].values
    
    def F(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, n_features).
        '''
        if not self.F_cols:
            return None
        return self.df[self.F_cols].values
    
    def Fstd(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, n_features).
        '''
        if not self.Fstd_cols:
            return None
        return self.df[self.Fstd_cols].values
    
    def X_F(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, (n_wavelengths + n_features)).
        '''
        xf = []
        if self.X_cols is not None:
            xf.append(self.df[self.X_cols].values)
        if self.F_cols is not None:
            xf.append(self.df[self.F_cols].values)
        
        if len(xf) == 0:
            return None
        return np.hstack(xf)
    
    def Xstd_Fstd(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, (n_wavelengths + n_features)).
        '''
        xf_std = []
        if self.Xstd_cols is not None:
            xf_std.append(self.df[self.Xstd_cols].values)
        if self.Fstd_cols is not None:
            xf_std.append(self.df[self.Fstd_cols].values)
        
        if len(xf_td) == 0:
            return None
        return np.hstack(xf_std)
    
    def y(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, 1).
        '''
        if not self.y_col:
            return None
        return self.df[self.y_col].values.reshape((self.df.shape[0], 1))
    
    def ystd(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (n_spectra, 1).
        '''
        if not self.ystd_col:
            return None
        return self.df[self.ystd_col].values.reshape((self.df.shape[0], 1))
    
    def wavelengths(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (1, n_wavelengths).
        '''
        if not self.X_cols:
            return None
        waves = [float(c.split('_')[-1]) for c in self.X_cols]
        return np.array(waves).reshape((1, len(waves)))
    
    def features(self) -> np.ndarray:
        '''
        Returns: np.ndarray with shape (1, n_features).
        '''
        if not self.F_cols:
            return None
        descs = [c.split('_')[-1].strip() for c in self.F_cols]
        return descs
    
    def sample_ids(self) -> List[str]:
        return list(self.df['sample_id'].values)
    
    def subsample_ids(self) -> List[str]:
        return list(self.df['subsample_id'].values)
    
    def add_columns(self, add_df:pd.DataFrame) -> None:
        if add_df.shape[0] == self.df.shape[0]:
            self.df = pd.concat([self.df, add_df], axis=1)
    
    def get_detail(self, key:str) -> Any:
        return self.details[key]
    
    def set_detail(self, key:str,
                         value:str) -> None:
        self.details[key] = value
