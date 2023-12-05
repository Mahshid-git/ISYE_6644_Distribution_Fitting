# DistFit Application 
## Fitting Data to Probability Distributions Using Maximum Likelihood Estimate (MLE)
Authors: Mahshid Jafar Pour, Mehdi Sadeghi

Date: December 2023

# Abstract
The DistFit package is a Python package designed for distribution fitting and goodness-of-fit testing. It estimates distribution parameters using Maximum Likelihood Estimation (MLE)  for different probability distributions, as well as Chi-Square goodness of fit for assessment of how well a given distribution fits a set of observed data, The package is adept at handling both discrete and continuous distributions, offering users a flexible and versatile tool for conducting in-depth statistical analyses. The implemented distributions are: Bernoulli, Binomial, Geometric, Poisson, Uniform, Exponential, Normal, Weibull, and Gamma. This paper serves as a comprehensive guide to DistFit, including some details about each distribution type. The MLE approach is employed to estimate the parameters of these distributions.

Key features of DistFit include easy use, flexibility in handling diverse distribution types, and efficient algorithms for parameter estimation. The package serves as a valuable asset for distribution analysis and hypothesis testing in Python.


# Methodology
The DistFit library is a Python based application that has three distinct modules: datagen, distfit, and gof. These modules perform data generation, distribution fitting, data visualization, and goodness-of-fit analysis. Additionally, an examples file demonstrates real-world applications of the datagen, distfit, and gof modules. It is a practical guide, describing how to use DistFit to fit distributions to datasets and analyze them. The examples show the versatility  of DistFit in fitting distributions to data and assessing the goodness of fit. The subsequent sections describe the specific details of each module, their functionalities and how to use them.

## datagen
The datagen module within DistFit is used for random data generation based on specified probability distributions. This module provides a convenient way for users to create artificial datasets of various probability distributions. The primary purpose of the datagen module is to facilitate testing and experimentation with DistFit's distribution fitting capabilities.

The Datagen takes distribution type, the desired number of data, and distribution-specific parameters. It supports a variety of distribution types, including Normal, Geometric, Binomial, Poisson, Exponential, Gamma, Weibull, Uniform, and Bernoulli.

Initialization Parameters are:

* dist_type: Type of probability distribution (e.g., 'Normal', 'Geometric', 'Binomial').

* row_count: Number of rows (data points) to be generated.

* par: Distribution-specific parameters. The format of parameters varies based on the distribution type.

The data_generation method in this module generates synthetic data based on the specified distribution type and parameters. It utilizes NumPy's random number generation functions for each distribution type.

## distfit
The methodology implemented in the Fitting class of the DistFit Python package revolves around fitting data to various probability distributions using MLE method. The class supports both discrete and continuous distributions. The key components of the methodology involve Initialization, Guessing Distributions, and parameter estimation of various distribution types. Additionally, the plot method can be used to visually compare the fitted distributions versus the data histogram

### Initialization
The class is initialized with a given distribution and dataset (data), which can be provided as a pandas DataFrame or a pandas Series. If the input is not a DataFrame, it is converted to the correct format.

Essential statistics about the dataset are computed during initialization, such as mean (mu), standard deviation (sigma), size (size), minimum value (data_min), maximum value (data_max), and the original data itself.

### Guessing Distributions
The ‘guess_distributions’ method determines the possible distributions based on the data type (discrete or continuous) and characteristics of the dataset.

For discrete data, it checks for specific conditions to identify Bernoulli distribution or selects from a predefined list of discrete distributions.

For continuous distributions, it considers all the specified continuous distributions, adjusting the list based on the properties of the dataset. For example, if the data histogram is skewed, it cannot be normally distributed.

This function should be used as a high-level guess of the distribution type. It is always the subject matter expert who knows which distribution is best suited for the data.

### Fit (Parameter Estimation)
MLE provides a systematic and versatile approach to fitting data to a range of distributions. It alsoempowering users with tools for exploratory data analysis.The visualizations provide an understanding of how well the fitted distributions align with the observed data.

### Gof (Goodness of Fit Test)
The gof (Goodness of Fit) module in DistFit evaluates how well a chosen probability distribution fits the observed data. It employs chi-squared goodness of fit test to assess the statistical significance of the difference between observed and expected frequencies. .. 

The following are the key components of this module:

* Initialization: Initializes the Gof class with the distribution type (dist_type) and its parameters (par). The class automatically determines the number of estimated parameters (s) based on the distribution type.
* Frequency Calculation: Calculates the frequency of the expected data within the specified bin edges. This step is required  to calculate the test statistics for goodness-of-fit test. This calculation is a numerical approximation of the expected data frequency as it is performed numerically.
* Binning Optimization: Optimizes bins by combining adjacent bins if the expected frequency is less than 5. However, it ensures a minimum of 3 bins even when frequencies are low.
* Goodness of Fit Test: Conducts a goodness-of-fit test using the chi-squared statistic. It generates expected data based on the specified distribution, calculates observed and expected frequencies, optimizes bins, and performs the chi-squared test. As a result of approximate frequency calculation for the expected data, the goodness of fit test is approximate and the results should be treated cautiously when the test statistics and critical values are close in values.
The  goodness-of-fit analysis should be used in conjunction with the "distfit" class to ensure the selected distribution and its estimated parameters are a good fit. The “distfit” class, as previously discussed, efficiently fits data to a variety of probability distributions using the MLE method.

## Examples 
The DistFit package includes an Examples file showcasing the integration of these modules. Users can visualize the fitted distributions, explore possible alternatives, and conduct hypothesis tests to assess the fit of the chosen distribution.
