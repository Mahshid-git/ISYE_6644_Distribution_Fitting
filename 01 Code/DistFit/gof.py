import numpy as np
from scipy.stats import chi2, chisquare
import DistFit.datagen as dgn


class Gof():
    def __init__(self, dist_type, par) -> None:
        self.dist_type = dist_type
        if dist_type in ['normal',  'weibull', 'gamma']:
            self.s = 2 # number of estimated parameters
        else:
            self.s = 1
        self.par = par
    
    def frequency_calc(self, data, bin_edges):
        '''
        calculate frequency of data
        '''
        n0 = len(bin_edges)
        min_add = 0
        max_add = 0
        if data.min() < bin_edges[0]:
            bin_edges.insert(0, data.min())
            min_add = 1
        n = len(bin_edges)
        if data.max() > bin_edges[n-1]:
            bin_edges.insert(n, data.max())
            max_add = 1
       
        exp_freq = [0]*(len(bin_edges)-1)
        for i in np.arange(len(bin_edges)-1):
            if i == len(bin_edges)-2:
                exp_freq[i] = ((data >= bin_edges[i]) & (data <= bin_edges[i+1])).sum()
            else:
                exp_freq[i] = ((data >= bin_edges[i]) & (data < bin_edges[i+1])).sum()
    
        if n0 == len(bin_edges):
            return exp_freq
        else:
            if min_add == 1:
                result0 = exp_freq[0] + exp_freq[1]
                del exp_freq[0]
                exp_freq[0] = result0
    
            if max_add == 1:
                resultn = exp_freq[-1] + exp_freq[-2]
                del exp_freq[-1]
                exp_freq[-1] = resultn
        return exp_freq
    
    def bigger_bins(self, exp_freq, obs_freq):
        '''
        combines 2 bins expected frequency is less than 5
        However it keeps at least 3 bins even though frequency is less than 5
        '''
        n = len(obs_freq)
        for i in np.arange(n-1,-1,-1):
            if ((exp_freq[i] < 5) & (len(exp_freq) >= 3)): # having at least 3 intervals
                if i==(n-1):
                    freqn = exp_freq[i] + exp_freq[i-1]
                
                    if freqn > 5:
                        del exp_freq[i]
                        exp_freq[i-1] = freqn
                    
                        freqn = obs_freq[i] + obs_freq[i-1]
                        del obs_freq[i]
                        obs_freq[i-1] = freqn
                    else:
                        exp_freq[i-1] = freqn 
                        freqn = obs_freq[i] + obs_freq[i-1]
                        obs_freq[i-1] = freqn

                else:
                    freq0 = exp_freq[i] + exp_freq[i+1]
                    del exp_freq[i]
                    exp_freq[i] = freq0
                
                    freq0 = obs_freq[i] + obs_freq[i+1]
                    del obs_freq[i]
                    obs_freq[i] = freq0
      
        return exp_freq, obs_freq
    
    def gof(self, data, k=5, alfa=0.05, seed=43):
        if self.dist_type == 'bernoulli':
            raise ValueError("GoF is not valid for Bernoulli.")
        size = data.shape[0]
        # generate data from expected distribution
        np.random.seed(seed)
        data_dist = dgn.Datagen(dist_type=self.dist_type, row_count=size, par=self.par) 
        exp_data = data_dist.data_generation()
        # calculate frequency
        obs_freq, bin_edges = np.histogram(data, bins=k)
        exp_freq = Gof.frequency_calc(self, exp_data, list(bin_edges))
        # combine bins if frequency is small
        exp_freq, obs_freq = Gof.bigger_bins(self, list(exp_freq), list(obs_freq))
        k = len(exp_freq)
    
        test_stat = chisquare(obs_freq, exp_freq, ddof=k-1-self.s).statistic #one-way chi squared test
        critical_val = chi2.isf(alfa, k-1-self.s) #isf: inverse survival function
        print("Test Statistics:", test_stat, "; Critical Value:", critical_val)
        if test_stat < critical_val:
            print("Accept H0 that the distribution is a good fit at the given significance level.")
            return True
        else:
            print("Reject H0, the distribution is NOT a good fit at this significance level.")
            return False
