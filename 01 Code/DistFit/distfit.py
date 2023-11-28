import numpy as np
from scipy.stats import chisquare
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

class Fitting():
    ''' 
    Fits data to input distributions using maximum likelihood estimate (MLE). 
    Distributions: Bern(p), Bin(n,p), Geom(p), Poiss(λ), Unif(a,b), Exp(λ), Normal(μ,s), Gamma(α,β), Weibull(α,β)
    Args:
        data (pandas dataframe): data to be fitted to the specified distribution 
    Returns: 
        parameters of the specified distribution
    '''
    def __init__(self, data):
        if isinstance(data, pd.DataFrame):
            pass
        else:
             # convert data to the correct format
            data = pd.Series(data)

        self.discrete_dist = ['Binomial', 'Geometric', 'Poisson']
        self.continuous_dist = ['Uniform', 'Exponential', 'Normal',  'Weibull', 'Gamma']
        self.mu = data.mean()
        self.sigma = data.std()  
        self.size = data.shape[0]
        self.data_min = data.min()
        self.data_max = data.max()
        self.data = data

    def guess_distributions(self):
        data_np = np.array(self.data)    
        integer_array = np.mod(data_np, 1)
        if integer_array.all() == 0:  # discrete distribution
            if set(self.data.unique()) == {1, 0}:
                dist = ['Bernoulli']
            else:
                dist = self.discrete_dist
        else: # continuous distibution
            dist = self.continuous_dist
            if self.data_min < 0:
                dist.remove('Exponential')
                dist.remove('Weibull')
                dist.remove('Gamma')
            elif self.data_min>0:
                dist.remove('Normal')
        print('Note: only a limited number of distributions are considered in this library.\n')
        print('The possible distributions for the data are:', dist)
        if 'Binomial' in dist:
            print('If data are binomial, n is at least', self.data_max)
        return dist
    #################################################
    #########   Discrete Distributions   ############
    ################################################# 
    # 1) Bernoulli(p)
    def bernoulli_fit(self):
        '''
        fits data to ber(p) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            p: parameter of bernoulli distribution , success rate
        '''
        p = self.mu
        return p
    
    def bernoulli_plot(self, params):
        x = np.arange(0, 2)
        y = [1-params, params]
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Bernoulli")
        bins = np.arange(0, self.data_max + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.legend()
        plt.show()
        pass
    
    # 2) Binomial(n,p)
    def binomial_fit(self, n):
        '''
        fits data to bin(n,p) distribution assuming n is known

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 
            n: number of independent Bernoulli trials

        Returns: 
            p: parameter of binomial distribution  
        '''
        print('\nNote: When estimating p with very rare events and a small n, using MLE estimator leads to p=0 which sometimes is unrealistic and undesirable. In such cases, use alternative estimators.\n\n')

        p = self.mu/n
        return (n, p)
    
    def binomial_plot(self, params):
        import math
        n = params[0]
        x = np.arange(0, n+1) # number of success in n trials
        y = [math.comb(n, i)*(params[1]**i)*((1-params[1])**(n-i)) for i in x]
        bins = np.arange(0, self.data_max + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Binomial")
        plt.legend()
        plt.show()
        pass
    
    # 3) Geometric(p)
    def geometric_fit(self):
        '''
        fits data to geom(p) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            p: parameter of geometric distribution  
        '''
        p = 1/(self.mu+1)
        return p
    
    def geometric_plot(self, params):
        x = np.arange(1, self.data_max)
        y = (params)*(1-params)**x
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Geometric")
        bins = np.arange(0, self.data_max + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.legend()
        plt.show()
        pass

    # 4) Poisson(lambda)
    def poisson_fit(self):
        '''
        fits data to Poisson distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            _lambda: parameter of Poisson distribution  
        '''
              
        _lambda = self.mu
        return _lambda
    
    def poisson_plot(self, params):
        from scipy.special import factorial
        _lambda = params
        x = np.arange(0, 51)
        y = np.exp(-_lambda)*_lambda**x/factorial(x)
        bins = np.arange(0, self.data_max + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Poisson")
        plt.legend()
        plt.show()
        pass

    #################################################
    ########   Continuous Distributions   ###########
    ################################################# 
    # 1) Uniform(a,b)
    def uniform_fit(self):
        '''
        fits data to uniform(a,b) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            a: lower bound, parameter of uniform distribution 
            b: upper bound, parameter of uniform distribution 
        '''
        a = self.data_min
        b = self.data_max
        return (a,b)
    
    def uniform_plot(self, params):
        x = np.arange(params[0], params[1],.01)
        y = (1/(params[1]-params[0]))*np.ones(x.shape)
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Uniform")
        plt.legend()
        plt.show()
        pass

    # 2) Exponential(lambda)
    def exponential_fit(self):
        '''
        fits data to exponential distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            self.mu: 1/rate, parameter of exponential distribution 
        '''
        return self.mu
    
    def exponential_plot(self, params):
        x = np.arange(0, params*10,.01)
        y = (1/params)*np.exp(-x/params)
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Exponential")
        plt.legend()
        plt.show()
        pass
    
    # 3) Normal(mu, sigma)
    def normal_fit(self):
        '''
        fits data to normal distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            mu: sample mean (1st parameter of Normal distribution) 
            sigma: MLE estimate of standard deviation (2nd parameter of Normal distribution)
        '''
        mu = self.mu #data.mean()
        sigma = self.sigma #data.std()

        return (mu, sigma)
    
    def normal_plot(self, params):
        x = (np.arange(-4, 4,.05))*params[1]+params[0]
        y = (1/(params[1]*np.sqrt(2*np.pi)))*np.exp(-((x-params[0])**2)/(2*params[1]**2))
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Normal")
        plt.legend()
        plt.show()
        pass
    
    # 4) Weibul(alpha, beta)
    def weibull_fit(self, tol=0.001):
        '''
        fits data to Weibull(alfa, beta) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 
            tol: tolerance for fitting alpha

        Returns: 
            alfa: scale parameter of Weibull distribution; must be positive
            beta: shape parameter of Weibull distribution; must be positive 
        '''
   
        # starting point for alpha [Thoman, Bain, and Antle (1969)]
        sum_ln_x_square = (np.log(self.data)**2).sum()
        sum_ln_x = np.log(self.data).sum()
        alfa_0 = (6*(sum_ln_x_square-sum_ln_x**2/self.size)/((self.size-1)*np.pi**2))**(-0.5)
        A = sum_ln_x/self.size # average of ln(X)
        alfa_new = alfa_0
        alfa_old = 100000.00 
        while abs(alfa_new-alfa_old) < tol:
            alfa_old = alfa_new
            Bk = (self.data**alfa_old).sum()
            Ck = ((self.data**alfa_old)*np.log(self.data)).sum()
            Hk = ((self.data**alfa_old)*(np.log(self.data))**2).sum()
            alfa_new = alfa_old - (A+(1/alfa_old)-Ck/Bk)/(1/alfa_old + (Bk*Hk-Ck**2)/(BK**2))
        
        beta = ((self.data**alfa_new).sum()/self.size)**(1/alfa_new)
        return (alfa_new, beta)
    
    def weibull_plot(self, params):
        import math
        alfa = params[0]
        beta = params[1]
        # to change
        x1 = np.arange(0, 2, .01)
        x2 = np.arange(2,10,.2)
        x = np.concatenate((x1, x2), axis=0)
        y = alfa*(beta**(-alfa))*(x**(alfa-1))*(np.exp(-(x/beta)**alfa))
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Weibull")
        plt.legend()
        plt.show()
        pass

    # 5) gamma(alpha, beta)
    def lookup(self, x, x_list, y_list):
        '''
        interpolates linearly based on a look-up table
        reference: this function is used from: https://stackoverflow.com/questions/50508262/using-look-up-tables-in-python

        Args:
            x: point data whose y is unknown in (x,y) look-up table 
            x_list (list): look-up table x values
            y_list (list): look-up table y values

        Returns: 
            y: y value for x based on linear interpolation of a look-up table 
        '''
        from bisect import bisect_left
        if x <= x_list[0]:  return y_list[0]
        if x >= x_list[-1]: return y_list[-1]

        i = bisect_left(x_list, x)
        k = (x - x_list[i-1])/(x_list[i] - x_list[i-1])
        y = k*(y_list[i]-y_list[i-1]) + y_list[i-1]

        return y

    def gamma_fit(self):
        '''
        fits data to Gamma(alfa, beta) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            alfa: scale parameter of Weibull distribution; must be positive
            beta: shape parameter of Weibull distribution; must be positive 
        '''
        T = 1/(np.log(self.mu)-np.log(self.data).sum()/self.size)        
        
        alfa_list = [0.01, 0.019, 0.027, 0.036, 0.044, 0.052, 0.06, 0.068, 0.076, 0.083, 0.09, 0.098, 0.105, 
             0.112, 0.119, 0.126, 0.133, 0.14, 0.147, 0.153, 0.218, 0.279, 0.338, 0.396, 0.452, 0.507, 
             0.562, 0.616, 0.669, 0.722, 0.775, 0.827, 0.879, 0.931, 0.983, 1.035, 1.086, 1.138, 1.189, 
             1.24, 1.291, 1.342, 1.393, 1.444, 1.495, 1.546, 1.596, 1.647, 1.698, 1.748, 1.799, 1.849, 
             1.9, 1.95, 2.001, 2.051, 2.101, 2.152, 2.253, 2.353, 2.454, 2.554, 2.655, 2.755, 2.856, 
             2.956, 3.057, 3.157, 3.257, 3.357, 3.458, 3.558, 3.658, 3.759, 3.859, 3.959, 4.059, 4.159, 
             4.26, 4.36, 4.46, 4.56, 4.66, 4.76, 4.86, 4.961, 5.061, 5.161, 5.411, 5.661, 5.912, 6.162, 
             6.412, 6.662, 6.912, 7.163, 7.413, 7.663, 7.913, 8.163, 8.413, 8.663, 8.913, 9.163, 9.414, 
             9.664, 9.914, 10.164, 10.414, 10.664, 10.914, 11.164, 11.414, 11.664, 11.914, 12.164, 12.414,
             12.664, 15.165, 17.665, 20.165, 22.665, 25.166]
        T_array = np.concatenate((np.arange(.01,.2,.01), np.arange(.2, 4,.1), np.arange(4, 10,.2), np.arange(10, 25,.5), np.arange(25, 51, 5)), axis=0)
        T_list = list(T_array)
        
        alfa = Fitting.lookup(self, T, T_list, alfa_list)
        beta = self.mu/alfa
        return (alfa, beta)
    
    def gamma_plot(self, params):
        import math
        alfa = params[0]
        beta = params[1]
        # to change
        x = np.arange(0, 50, .1)
        y = (beta**(-alfa))*(x**(alfa-1))*(np.exp(-x/beta))/math.gamma(alfa)
        plt.hist(self.data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Gamma")
        plt.legend()
        plt.show()
        pass
