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
    # 1) Bernoulli(p)
    def bernoulli_fit(self, data):
        '''
        fits data to ber(p) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            p: parameter of bernoulli distribution , success rate
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        p = data.mean()
        return p
    
    def bernoulli_plot(self, data, params):
        x = np.arange(0, 2)
        y = [1-params, params]
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Bernoulli")
        bins = np.arange(0, data.max() + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.legend()
        plt.show()
        pass
    
    # 2) Binomial(n,p)
    def binomial_fit(self, data, n):
        '''
        fits data to bin(n,p) distribution assuming n is known

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 
            n: number of independent Bernoulli trials

        Returns: 
            p: parameter of binomial distribution  
        '''
        print('\nNote: When estimating p with very rare events and a small n, using MLE estimator leads to p=0 which sometimes is unrealistic and undesirable. In such cases, use alternative estimators.\n\n')
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        p = data.mean()/n
        return (n, p)
    
    def binomial_plot(self, data, params):
        import math
        n = params[0]
        x = np.arange(0, n+1) # number of success in n trials
        y = [math.comb(n, i)*(params[1]**i)*((1-params[1])**(n-i)) for i in x]
        bins = np.arange(0, data.max() + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Binomial")
        plt.legend()
        plt.show()
        pass
    
    # 3) Geometric(p)
    def geometric_fit(self, data):
        '''
        fits data to geom(p) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            p: parameter of geometric distribution  
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        p = 1/(data.mean()+1)
        return p
    
    def geometric_plot(self, data, params):
        x = np.arange(1, data.max())
        y = (params)*(1-params)**x
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Geometric")
        bins = np.arange(0, data.max() + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.legend()
        plt.show()
        pass

    # 4) Poisson(lmabda)
    def poisson_fit(self, data):
        '''
        fits data to Poisson distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            _lambda: parameter of Poisson distribution  
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
                
        _lambda = data.mean()
        return _lambda
    
    def poisson_plot(self, data, params):
        from scipy.special import factorial
        _lambda = params
        x = np.arange(0, 51)
        y = np.exp(-_lambda)*_lambda**x/factorial(x)
        bins = np.arange(0, data.max() + 1.5) - 0.5 # trick to have bins centered on integers
        plt.hist(data, alpha=.4, density = True, label="Data Histogram", bins=bins)
        plt.plot(x, y, marker='o', linestyle=' ', color ='r', label="Fitted Poisson")
        plt.legend()
        plt.show()
        pass

    #################################################
    ########   Continuous Distributions   ###########
    ################################################# 
    # 1) Uniform(a,b)
    def uniform_fit(self, data):
        '''
        fits data to uniform(a,b) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            a: lower bound, parameter of uniform distribution 
            b: upper bound, parameter of uniform distribution 
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        a = data.min()
        b = data.max()
        return (a,b)
    
    def uniform_plot(slef, data, params):
        x = np.arange(params[0], params[1],.01)
        y = (1/(params[1]-params[0]))*np.ones(x.shape)
        plt.hist(data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Uniform")
        plt.legend()
        plt.show()
        pass

    # 2) Exponential(lambda)
    def exponential_fit(self, data):
        '''
        fits data to exponential distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            mu: 1/rate, parameter of exponential distribution 
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        mu = data.mean()

        return mu
    
    def exponential_plot(self, data, params):
        x = np.arange(0, params*10,.01)
        y = (1/params)*np.exp(-x/params)
        plt.hist(data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Exponential")
        plt.legend()
        plt.show()
        pass
    
    # 3) Normal(mu, sigma)
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
    
    # 4) Weibul(alpha, beta)
    def weibull_fit(slef, data):
        '''
        fits data to Weibull(alfa, beta) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            alfa: scale parameter of Weibull distribution; must be positive
            beta: shape parameter of Weibull distribution; must be positive 
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        
        # starting point for alpha [Thoman, Bain, and Antle (1969)]
        n = data.shape[0] # data size
        sum_ln_x_square = (np.log(data)**2).sum()
        sum_ln_x = np.log(data).sum()
        alfa_0 = (6*(sum_ln_x_square-sum_ln_x**2/n)/((n-1)*np.pi**2))**(-0.5)
        A = sum_ln_x/n # average of ln(X)
        alfa_new = alfa_0
        alfa_old = 100000.00 
        while abs(alfa_new-alfa_old) < 0.001:
            alfa_old = alfa_new
            Bk = (data**alfa_old).sum()
            Ck = ((data**alfa_old)*np.log(data)).sum()
            Hk = ((data**alfa_old)*(np.log(data))**2).sum()
            alfa_new = alfa_old - (A+(1/alfa_old)-Ck/Bk)/(1/alfa_old + (Bk*Hk-Ck**2)/(BK**2))
        
        beta = ((data**alfa_new).sum()/n)**(1/alfa_new)
        return (alfa_new, beta)
    
    def weibull_plot(self, data, params):
        import math
        alfa = params[0]
        beta = params[1]
        # to change
        x1 = np.arange(0, 2, .01)
        x2 = np.arange(2,10,.2)
        x = np.concatenate((x1, x2), axis=0)
        y = alfa*(beta**(-alfa))*(x**(alfa-1))*(np.exp(-(x/beta)**alfa))
        plt.hist(data, alpha=.4, density = True, label="Data Histogram")
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
            x: data to be fitted to the specified distribution 
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

    def gamma_fit(self, data):
        '''
        fits data to Gamma(alfa, beta) distribution

        Args:
            data (pandas dataframe): data to be fitted to the specified distribution 

        Returns: 
            alfa: scale parameter of Weibull distribution; must be positive
            beta: shape parameter of Weibull distribution; must be positive 
        '''
        if isinstance(data, pd.DataFrame):
                pass
        else:
                data = pd.Series(data)
        
        n = data.shape[0] # data size
        T = 1/(np.log(data.mean())-np.log(data).sum()/n)        
        
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
        
        alfa = lookup(self, T, T_list, alfa_list)
        beta = data.mean()/alfa
        return (alfa, beta)
    
    def gamma_plot(self, data, params):
        import math
        alfa = params[0]
        beta = params[1]
        # to change
        x = np.arange(0, 50, .1)
        y = (beta**(-alfa))*(x**(alfa-1))*(np.exp(-x/beta))/math.gamma(alfa)
        plt.hist(data, alpha=.4, density = True, label="Data Histogram")
        plt.plot(x, y, color ='r', label="Fitted Gamma")
        plt.legend()
        plt.show()
        pass
