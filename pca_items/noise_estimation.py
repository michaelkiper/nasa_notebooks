from typing import Dict
import sys 
from pathlib import Path
import numpy as np
import json
# from spectrum_data import SpectrumData
    
    
class NoiseEstimator(object):
    def __init__(self) -> None:
        pass 
    
    def __call__(self, data) -> Dict:
        return {'noise_covariance': None,
                'method': ''}
    
    
class MultipleRegressionClassicNoiseEstimator(NoiseEstimator):
    '''
    Additive noise estimation for hyperspectral data.
    The method assumes that the signal in a band is well 
    approximated by a linear regression on the remaining 
    bands - a fast method is used to regress left out bands. 
    
    The algorithm is based on: 
    Hyperspectral Subspace Identification,
    Bioucas-Dias and Nascimento, 2008,
    doi: 10.1109/TGRS.2008.918089
    
    Based on MATLAB code by:
    1. Bioucas-Dias and Nascimento, 
    2. Kerry Cawse-Nicholson@JPL
    '''
    
    def __init__(self, epsilon:float=1e-6,
                       diagonalize:bool=False) -> None:
        '''
        Arguments:
        `epsilon`: float
                   Used for conditioning the inverse.
        '''
        self.epsilon = epsilon
        self.diagonalize = diagonalize
        
    # def __call__(self, data:SpectrumData=None) -> Dict:
    def __call__(self, data:np.array=None) -> Dict:
        '''
        Arguments: 
        `data`: SpectrumData 
                n_spectra each with n_bands
                
        Returns:
        A dict with the following keys: 
        `spectra_noise`: (n_spectra, n_bands) np.ndarray 
                         Noise for each spectra for each band.
        `noise_covariance`: (n_bands, n_bands) np.ndarray
                            Noise covariance matrix.
        '''
        # n_spectra, n_bands = data.X().shape
        n_spectra, n_bands = data.shape
    
        # observation correlation matrix
        # r_hat = np.dot(data.transpose(), data.X())
        r_hat = np.dot(data.transpose(), data)
        
        # the "first part" in the pseudo-inverse.
        # algorithm is fast as this is computed outside the loop.
        r_prime = np.linalg.inv(r_hat + self.epsilon*np.eye(n_bands))
     
        # estimate pixel noise at each band
        # "rank-one update" type operations that removes data for a band.
        # spectra_noise = np.zeros_like(data.X())
        spectra_noise = np.zeros_like(data)
        for band in np.arange(n_bands):
            r_prime_zeroed = r_prime - np.outer(r_prime[:, band], r_prime[band, :])/r_prime[band, band]
        
            r_hat_band_col = r_hat[:, band]
            r_hat_band_col[band] = 0.0 
        
            beta_hat = np.dot(r_prime_zeroed, r_hat_band_col)
            beta_hat[band] = 0.0
            
            # spectra_noise[:, band] = data.X()[:, band] - np.dot(data.X(), beta_hat)
            spectra_noise[:, band] = data[:, band] - np.dot(data, beta_hat)
        # noise covariance matrix
        noise_covariance = np.dot(spectra_noise.transpose(), spectra_noise)/n_spectra
        if self.diagonalize:
            noise_covariance = np.diag(np.diag(noise_covariance))
    
        return {'noise_covariance': noise_covariance,
                'spectra_noise': spectra_noise,
                'estimation_method': self.__class__.__name__}
