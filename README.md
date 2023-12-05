# DistFit Application 
## Fitting Data to Probability Distributions Using Maximum Likelihood Estimate (MLE)
Authors: Mahshid Jafar Pour, Mehdi Sadeghi

Date: December 2023

# Abstract
The DistFit package is a Python package designed for distribution fitting and goodness-of-fit testing. It estimates distribution parameters using Maximum Likelihood Estimation (MLE)  for different probability distributions, as well as Chi-Square goodness of fit for assessment of how well a given distribution fits a set of observed data, The package is adept at handling both discrete and continuous distributions, offering users a flexible and versatile tool for conducting in-depth statistical analyses. The implemented distributions are: Bernoulli, Binomial, Geometric, Poisson, Uniform, Exponential, Normal, Weibull, and Gamma. This paper serves as a comprehensive guide to DistFit, including some details about each distribution type. The MLE approach is employed to estimate the parameters of these distributions.

Furthermore, the DistFit package includes an Examples file showcasing the integration of these modules. Users can visualize the fitted distributions, explore possible alternatives, and conduct hypothesis tests to assess the fit of the chosen distribution.

Key features of DistFit include easy use, flexibility in handling diverse distribution types, and efficient algorithms for parameter estimation. The package serves as a valuable asset for distribution analysis and hypothesis testing in Python.


# Methodology
The DistFit library is a Python based application that has three distinct modules: datagen, distfit, and gof. These modules perform data generation, distribution fitting, data visualization, and goodness-of-fit analysis. Additionally, an examples file demonstrates real-world applications of the datagen, distfit, and gof modules. It is a practical guide, describing how to use DistFit to fit distributions to datasets and analyze them. The examples show the versatility  of DistFit in fitting distributions to data and assessing the goodness of fit. The subsequent sections describe the specific details of each module, their functionalities and how to use them.

## datagen
The datagen module within DistFit is used for random data generation based on specified probability distributions. This module provides a convenient way for users to create artificial datasets of various probability distributions. The primary purpose of the datagen module is to facilitate testing and experimentation with DistFit's distribution fitting capabilities.

The Datagen takes distribution type, the desired number of data, and distribution-specific parameters. It supports a variety of distribution types, including Normal, Geometric, Binomial, Poisson, Exponential, Gamma, Weibull, Uniform, and Bernoulli.

Initialization Parameters are:

* dist_type: Type of probability distribution (e.g., 'Normal', 'Geometric', 'Binomial').

  •  row_count: Number of rows (data points) to be generated.

  •  par: Distribution-specific parameters. The format of parameters varies based on the distribution type.

The data_generation method in this module generates synthetic data based on the specified distribution type and parameters. It utilizes NumPy's random number generation functions for each distribution type.

## distfit


### Initialization

### Guessing Distributions

### Fit (Parameter Estimation)

### Gof (Goodness of Fit Test)

## Examples 
