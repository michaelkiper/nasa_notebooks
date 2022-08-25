from typing import Dict
import sys 
from pathlib import Path
import numpy as np
import json
# from spectrum_data import SpectrumData
    
    
class IntrinsicDimensionRMT(object):
    '''
    Estimate of intrinsic dimension (ID).
    Uses random matrix theory to find ID.
    LeaveOneBandOutAdditiveNoiseEstimation for noise estimation.
    
    The algorithm is based on: 
    Determining the Intrinsic Dimension of a Hyperspectral 
    Image Using Random Matrix Theory,
    Cawse_Nicholson et al., 2012,
    doi: 10.1109/TIP.2012.2227765
    
    Based on MATLAB code by:
    1. Kerry Cawse-Nicholson@JPL
    '''
    def __init__(self, alpha:float=0.5,
                       a:float=0.5,
                       b:float=0.5) -> None:
        '''
        `alpha`: float 
                 RMT significance level (paper fixes this at 0.5%)
                 a parameter used to compute `s` (see Algorithm 1 of paper)
        `a`: float
             This is not a parameter in the paper. The MATLAB code defines 
             this variable, so making it a specifiable. In the paper, a = 0.5.
        `b`: float
             This is not a parameter in the paper. The MATLAB code defines 
             this variable, so making it a specifiable. In the paper, b = 0.5.
        '''
        self.alpha = alpha 
        self.a = a
        self.b = b
        
    def __call__(self, data:np.array, noise_data:Dict) -> Dict:
    # def __call__(self, data:SpectrumData, noise_data:Dict) -> Dict:
        '''
        Arguments: 
        `data`: SpectrumData 
                n_spectra each with n_bands
        `noise_data`: dict
                      keys `noise_covariance` and `estimation_method` are required.
                
        Returns: 
        The returned dict will have the keys:
        `intrinsic_dimension`, `eigen_values`, `eigen_vectors`
        '''
        # n_spectra, n_bands = data.X().shape
        # band_means = np.nanmean(data.X(), axis=0, keepdims=True)    
        # band_covariance = np.cov(data.X() - band_means, 
        #                          rowvar=False, 
        #                          bias=True)

        n_spectra, n_bands = data.shape
        band_means = np.nanmean(data, axis=0, keepdims=True)    
        band_covariance = np.cov(data - band_means,
                                 rowvar=False, 
                                 bias=True)
        band_covariance = ((data - band_means).T@(data - band_means))/(n_spectra)
        
        # The Tracy-Widom distribution is the solution of a second order Painlevé ordinary differential equation
        # Painlevé ordinary differential equation
        factor1 = (n_spectra - self.a)**(1/2) + (n_bands - self.b)**(1/2)
        factor2 = ((n_spectra - self.a)**(-1/2) + (n_bands - self.b)**(-1/2))**(1/3)
        
        r_mu = factor1*factor1/n_spectra
        r_sigma = factor1*factor2/n_spectra
        s = (-(3/2)*np.log(4*(np.pi**(1/2))*self.alpha/100))**(2/3)
        r = r_mu + s*r_sigma
    
        if noise_data['estimation_method'] == 'MultipleRegressionClassicNoiseEstimator':
            phi = noise_data['noise_covariance']
            
        [lambdas1, evecs1] = np.linalg.eig(band_covariance)
        descending1 = np.argsort(lambdas1)[::-1] 
        lambdas1, evecs1 = lambdas1[descending1], evecs1[:, descending1] 
        
        [lambdas2, evecs2] = np.linalg.eig(band_covariance - phi)
        descending2 = np.argsort(lambdas2)[::-1] 
        lambdas2, evecs2 = lambdas2[descending2], evecs2[:, descending2]
    
        rhos = np.zeros(n_bands, dtype=np.float64)
        for b in np.arange(n_bands):
            rhos[b] = np.dot(evecs1[:, b], np.dot(phi, evecs2[:, b]))/np.dot(evecs1[:, b], evecs2[:, b])
        
        k_indicies = np.where(lambdas1 - rhos*r > 0)
        k_est = np.size(k_indicies)
    
        return {'intrinsic_dimension': k_est,
                'id_indicies': k_indicies,
                'rhos': rhos,
                'r': r,
                'eigen_values': lambdas1,
                'eigen_vectors': evecs1}
