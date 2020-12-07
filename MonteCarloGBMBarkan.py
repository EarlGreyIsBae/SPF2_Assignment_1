#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 00:47:15 2020

@author: connorstevens
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:56:30 2020

@author: connorstevens

Question 1a)
"""
print("Question 1a)")
print(" ")

import numpy as np
import pandas as pd
import pandas_datareader as pdr
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy import stats

#Set seed for testing.
np.random.seed(101)


#Download daily close data from Yahoo! and convert to monthly log return data.
pd.set_option('max_columns', None)
tckr = '^GSPC' 
data = pdr.get_data_yahoo(tckr, "2008-08-31", "2020-08-31")
daily_returns = data['Adj Close'].pct_change()
monthly_returns = data['Adj Close'].resample('M').ffill().pct_change()
monthly_returns = np.log(monthly_returns + 1)

#Calculate standard deviation of monthly returns and convert to annual.
sigma = np.std(monthly_returns) * np.sqrt(12)

#Set Black-Scholes price from previous assignment.
BSPrice = 238.5602321011006

#Print out Black-Scholes price from previous assignment.
print("Black-Scholes price calculated in assignment one: " + str(BSPrice))

#Input results from Q2e of assignment 1 for comparison.
SPFA1Q2eResult = 237.7574

"""
Name: MonteCarloSim

Description: Function simulates geometric brownian motion using Euler's method for various various call options based on annual inputs.
Inputs: r = annual interest rate, X - strike price, T - time to maturity,
S0 - initial stock price, M - number of paths, Dt - Delta-t, size of timesteps. Function coded for either montly or daily.

Returns: Call value
"""

def MonteCarloSim(S0, r, M, X, sigma, Dt):

    #If statements to change the value of Dt for changing delta-t.
    if Dt == "month":
        Dt = 1/12
        T = 3/12
    
    if Dt == "day":
        Dt = 1/252
        T = 63/252

    #Create empty vector array to hold path values.
    Euler = np.zeros((int(T/Dt) + 1, M))

    #Put S0 in first value of all columns for iteration purposes.
    Euler[0, :] = S0 

    #Create M paths of the three month Monte Carlo simulation with 
    for j in range(0, M):
        for i in range(1,len(Euler)):
            Euler[i, j] = Euler[i-1, j] + (Euler[i-1, j] * (r * Dt + sigma * np.sqrt(Dt) * np.random.normal(0, 1)))

    #Create empty vector to hold payoffs.
    Payoff = np.zeros(M)

    #Loop through paths and calculate payoff for each path.
    for prices in range(0, M):
        Payoff[prices] = max(Euler[-1, prices] - X, 0)

    #Calculate option value.
    Value = np.exp(-0.004997 * 3/12) * np.average(Payoff)
    
    return Value

#Create vector of number of paths for iterations to loop through.
Q1aPaths = [100, 500, 1000, 5000, 10000]

#Create empty vector to store results for Q1a.
Q1aResults = np.zeros(5)

#Run Monte Carlo simulations with delta-t = one month. Print results out and store in array for plotting.
print("One month steps")
print("---------------")
for paths1 in range(len(Q1aPaths)):
    Q1aResults[paths1] = MonteCarloSim(S0 = 3500.31005859375, r = 0.005, M = Q1aPaths[paths1], X = 3300, sigma = sigma, Dt = "month")
    print(str(Q1aResults[paths1]) + " - " + str(Q1aPaths[paths1]) + " paths.")

#Plot results.
fig1 = plt.plot(Q1aPaths, Q1aResults, label = "Euler's Method Monthly Steps")
plt.title('Convergence of Euler Method Stock Price to Black-Scholes Price for Monthly Steps')
plt.xlabel('Paths')
plt.ylabel('Option Price')
plt.xlim(100, 10000)
plt.ylim(220, 270)
plt.hlines(y = BSPrice, colors = 'r', xmin = 100, xmax=10000, label = 'Black-Scholes Price')
plt.hlines(y = SPFA1Q2eResult, colors = 'g', xmin = 100, xmax=10000, label = 'Assignment 1: 2e Result')
plt.xticks([100, 500, 1000, 5000, 10000], rotation = 'vertical')
plt.legend()
plt.show(fig1)

"""
b)
"""
#Create vector to store results for Q1b.
Q1bResults = np.zeros(5)

print(" ")
print("Question 1b)")

#Run Monte Carlo simulation with delta-t = one day. Print results out and store in array for plotting.
print(" ")
print("One day steps")
print("-------------")

#Run Monte Carlo simulations with delta-t = one day. Print results out and store in array for plotting.    
for paths2 in range(len(Q1aPaths)):
    Q1bResults[paths2] = MonteCarloSim(S0 = 3500.31005859375, r = 0.005, M = Q1aPaths[paths2], X = 3300, sigma = sigma, Dt = "day")
    print(str(Q1bResults[paths2]) + " - " + str(Q1aPaths[paths2]) + " paths.")

#Plot results.
fig2 = plt.plot(Q1aPaths, Q1bResults, label = "Euler's Method Daily Steps")
plt.title('Convergence of Euler Method Stock Price to Black-Scholes Price for Daily Steps')
plt.xlabel('Paths')
plt.ylabel('Option Price')
plt.xlim(100, 10000)
plt.ylim(220, 270)
plt.hlines(y = BSPrice, colors = 'r', xmin = 100, xmax=10000, label = 'Black-Scholes Price')  
plt.xticks([100, 500, 1000, 5000, 10000], rotation = 'vertical')
plt.legend()
plt.show(fig2)


#Plot results of Q1a and Q1b on same graph to compare results.
fig3 = plt.plot(Q1aPaths, Q1aResults, label = "Euler's Method Monthly Steps")
plt.plot(Q1aPaths, Q1bResults, label = "Euler's Method Daily Steps")
plt.title('Comparison of Convergence of Results from Q1a and Q1b to Black-Scholes Option Price')
plt.xlim(100, 10000)
plt.ylim(220, 270)
plt.xlabel('Paths')
plt.ylabel('Option Price')
plt.hlines(y = BSPrice, colors = 'r', xmin = 100, xmax=10000, label = 'Black-Scholes Price')  
plt.xticks([100, 500, 1000, 5000, 10000], rotation = 'vertical')
plt.legend()
plt.show(fig3)
