import numpy as np
from scipy.stats import chisquare
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

class Fitting():
    ''' 

    '''
    def __init__(self):
        pass    
    
    #################################################
    #########   Discrete Distributions   ############
    ################################################# 
    # def bernouli_fit(self, data):
    #     pass

    # def geometric_fit(self, data):
    #     pass
    


    #################################################
    ########   Continuous Distributions   ###########
    ################################################# 
    # def exponential_fit(self, data):
    #     pass

    def normal_fit(self, data):
        '''
        fits data to normal distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            mu: sample mean (1st parameter of Normal distribution) 
            sigma: MLE estimate of standard deviation (2nd parameter of Normal distribution)
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        mu = data.mean()
        sigma = data.std()

        return (mu, sigma)
    
    def normal_plot(self, data, params):
        x = (np.arange(-4, 4,.05))*params[1]+params[0]
        y = (1/(params[1]*np.sqrt(2*np.pi)))*np.exp(-((x-params[0])**2)/(2*params[1]**2))
        plt.hist(data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Normal")
        plt.legend()
        plt.show()
        pass

    # def gamma_fit(self, data):
    #     pass

    # def weibull_fit(self, data):
    #     pass

    # def plot_fit(self, data, parameters, distribution):
    #     pass