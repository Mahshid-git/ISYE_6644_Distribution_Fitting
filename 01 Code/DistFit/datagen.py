import numpy as np
from scipy.stats import chisquare
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

class Datagen():
    def __init__(self, dist_type, row_count, par) -> None:
        self.dist_type = dist_type
        self.row_count = row_count
        if self.dist_type== 'normal':
            self.mean, self.std = par
        
        elif self.dist_type== 'geometric':
            self.p = par
        
        elif self.dist_type== 'binomial':
            self.n, self.p= par
         
        elif self.dist_type == 'poisson':
            self.l= par
        
        elif self.dist_type== 'exponential':
            self.scale= par
    
        elif self.dist_type == 'gamma':
            self.shape , self.scale = par
       
        elif self.dist_type == 'weibull': 
            self.a, self.b = par # a= scale parameter b= shape 
       
        elif self.dist_type == 'uniform':
            self.a, self.b = par
             
        elif self.dist_type == 'bernoulli':
            self.a = par
        else:
            raise ValueError("Change distribution type or modify parameters")

    def data_generation(self):
        if self.dist_type== 'normal':
            data=np.random.normal(self.mean, self.std, self.row_count)
        
        elif self.dist_type== 'geometric':
            data=np.random.geometric(self.p, self.row_count)  
        
        elif self.dist_type== 'binomial':
            data=np.random.binomial(self.n, self.p, self.row_count)
        
        elif self.dist_type == 'poisson':
            data = np.random.poisson(self.l, self.row_count)    
        
        elif self.dist_type== 'exponential':
            data=np.random.exponential(self.scale, self.row_count)
    
        elif self.dist_type == 'gamma':
            data=np.random.gamma(self.shape, self.scale, self.row_count)
        
        elif self.dist_type == 'weibull': 
            data=b*np.random.weibull(a, self.row_count) #this is one-parameter Weibull; for 2-D => multiply by b
        
        elif self.dist_type == 'uniform':
            data = np.random.uniform(a, b, self.row_count)
            
        elif self.dist_type == 'bernoulli':
            data = np.random.bernoulli(a, self.row_count)
        
        else:
            raise ValueError("Change distribution type or modify parameters")
            
        return data