#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 10:59:01 2020

@author: connorstevens
"""
import numpy as np
import matplotlib.pyplot as plt

### Part 1: Option prices between the three methods with the same inputs

"""
Name: MonteCarloSim

Description: Function simulates stock prices movements using geometric brownian motion and then calculates the average payoff of the paths and discounting it for option price. Selection of parameters are set for purpose of this assignment.
Inputs: M - number of paths to be simulated, T - Time to maturity, Option - variable which can be set to either 'call' or 'put' for corresponding option value.

Returns: Option value
"""

def MonteCarloSim(M, T, Option):
    
    strike = 3300

    S0 = np.exp(8.160606831839509)
    
    r = 0.005
    
    sigma = 0.15732612210613792

    #If statements to change the value of Dt for changing delta-t.
    Dt = 1/252

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

    if Option == 'call':
        #Loop through paths and calculate payoff for each path.
        for prices in range(0, M):
            Payoff[prices] = max(Euler[-1, prices] - strike, 0)
        
    if Option == 'put':
        #Loop through paths and calculate payoff for each path.
        for prices in range(0, M):
            Payoff[prices] = max(strike - Euler[-1, prices], 0)

    #Calculate option value.
    Value = np.exp(-0.004997 * 3/12) * np.average(Payoff)
    
    return Value

MonteCarloSim(M = 1000, T = 63/252, Option = 'call')

"""
Name: MRCensoredBinTree

Description: Function simulates stock prices movements using the censored mean-reverting binomial tree for log of stock price and then calculates the option price by discounting the option value backwards from time to maturity. Selection of parameters are set for purpose of this assignment.
Inputs: T - Time to maturity, eta - Mean reversion speed, Option - variable which can be set to either 'call' or 'put' for corresponding option value.

Returns: Option value
"""

def MRCensoredBinTree(T, eta, Option):
    #Define time to maturity and time step size.
    Dt = 1/252
    
    #Define standard deviation.
    sigma = 0.13
    
    #Set initial value in binomial tree (already in log form).
    x0 = 8.160606831839509
    
    #Define mu.
    mu = 8.160606831839509
    
    #Define interest rate for discounting.
    r = 0.005
    
    #Define log of strike price.
    strike = 3300
    
    """Binomial Tree"""
    
    #Define time to maturity and time step size.
    Dt = 1/252
    
    
    #Initialize binomial tree.
    BinTree = np.zeros((int(T/Dt) + 1, int(T/Dt) + 1))
    
    #Set initial value in binomial tree (already in log form).
    BinTree[0, 0] = x0
    
    #Loop through diagonal and populate with down values.
    for diagonal in range(1, len(BinTree)):
        BinTree[diagonal, diagonal] = BinTree[diagonal-1, diagonal-1 ]  -np.sqrt(Dt) * sigma
        
    #Initialise counting variable for for loop.
    count = 0
    
    #Loop across periods and populate with up values.
    for nodes in range(0, len(BinTree)):
        count = count + 1
        for periods in range(count, len(BinTree)):
            BinTree[nodes, periods] =  BinTree[nodes, periods - 1] + np.sqrt(Dt) * sigma
            
    """Probability Tree"""
    
    #Initialize probability tree.
    ProbTree = np.zeros((int(T/Dt) + 1, int(T/Dt) + 1))
    
    #Initialise counting variable for for loop.
    count = - 1
    
    #Loop across periods and populate with up probabilities.
    for nodes in range(0, len(ProbTree)):
        count = count + 1
        for periods in range(count, len(ProbTree)):
            ProbTree[nodes, periods] = max(0, min(1, 0.5 + 0.5*(eta*(mu - BinTree[nodes, periods])/sigma) *np.sqrt(Dt)))
    
    """Calculating Option Value"""
    
    #Initialize value tree.
    ValTree = np.zeros((int(T/Dt) + 1, int(T/Dt) + 1))
    
    if Option == 'call':
        #Set option payoff at each node at maturity as last node value of value tree.
        ValTree[:, -1] = np.maximum(np.exp(BinTree[:, -1]) - strike, 0)
    
    if Option == 'put':
        #Set option payoff at each node at maturity as last node value of value tree.
        ValTree[:, -1] = np.maximum(strike - np.exp(BinTree[:, -1]), 0)
    
    #Initialise counting variable for for loop.
    count = T * 252
    
    #Loop across periods and populate with up values.
    for periods in range(len(ValTree) - 2, -1, -1):
        count = count - 1
        for nodes in range(0, int(count) + 1):
            ValTree[nodes, periods] = np.exp(-r * 1/252) * (ValTree[nodes, periods + 1] * ProbTree[nodes, periods] + ValTree[nodes + 1, periods + 1] * (1 - ProbTree[nodes, periods]))
        
    return ValTree[0,0]

MRCensoredBinTree(T=63/252,eta = 6, Option = 'call')

"""
Name: MonteCarloSimMR

Description: Function simulates stock prices movements using the Ornstein-Uhlenbeck mean reverting process for log of stock price and then calculates the average payoff of the paths and discounting it for option price. Selection of parameters are set for purpose of this assignment.
Inputs: M - number of paths to be simulated, T - Time to maturity, eta - Mean reversion speed, Option - variable which can be set to either 'call' or 'put' for corresponding option value.

Returns: Option value
"""

def MonteCarloSimMR(M, T, eta, Option,):
    
    #Define standard deviation.
    sigma = 0.13
    
    #Set initial value in binomial tree (already in log form).
    S0 = 8.160606831839509
    
    #Define mu.
    mu = 8.160606831839509
    
    #Define log of strike price.
    strike = 3300
    
    #Dt assumed to be daily.
    Dt = 1/252
    T = 63/252

    #Create empty vector array to hold path values.
    OU_MR = np.zeros((int(T/Dt) + 1, M))

    #Put S0 in first value of all columns for iteration purposes.
    OU_MR[0, :] = S0

    #Create M paths of the three month Monte Carlo simulation with 
    for j in range(0, M):
        for i in range(1,len(OU_MR)):
            OU_MR[i, j] = S0 * np.exp(-eta*Dt) + mu*(1-np.exp(-eta * Dt)) + sigma*np.sqrt((1-np.exp(-2*eta*Dt))/(2*eta))*np.random.normal()
    
    #And on the 8th day, God made Gary...
    Gary = np.zeros(OU_MR.shape)
    
    for rows in range(0, len(OU_MR)):
        for columns in range(0, M):
            Gary[rows, columns] = np.exp(OU_MR[rows, columns] - (0.5*(1-np.exp(-2*eta*T))*(sigma**2)/(2*eta)))

    #Create empty vector to hold payoffs.
    Payoff = np.zeros(M)
    
    if Option == "call":
        #Loop through paths and calculate payoff for call each path.
        for prices in range(0, M):
            Payoff[prices] = max(Gary[-1, prices] - strike, 0)
    
    if Option == 'put':
        #Loop through paths and calculate payoff for put each path.
        for prices in range(0, M):
            Payoff[prices] = max(strike - Gary[-1, prices], 0)
    

    #Calculate option value.
    Value = np.exp(-0.005 * T) * np.average(Payoff)
    
    return Value

MonteCarloSimMR(M = 1000,T = 63/.252 , eta = 6, Option = 'call')

Paths = [100, 500, 1000, 5000, 10000]

MonteCarloGBMPricesCall = np.zeros(len(Paths))
MonteCarloMRPricesCall = np.zeros(len(Paths))

for i in range(0, len(Paths)):
    MonteCarloGBMPricesCall[i] = MonteCarloSim(M = Paths[i], T = 63/252, Option = 'call')
    MonteCarloMRPricesCall[i] = MonteCarloSimMR(M = Paths[i], T = 63/252, eta = 6, Option = 'call')
    
BinTreeCall = MRCensoredBinTree(T=63/252,eta = 6, Option = 'call')
    
fig1 = plt.plot(Paths,MonteCarloGBMPricesCall, label = "GBM MC")
plt.plot(Paths, MonteCarloMRPricesCall, label = 'MR MC')
plt.hlines(BinTreeCall, colors = 'r', xmin = 0, xmax=10000, label = 'MR BinTree')
plt.title('Monte Carlo call price for idential option using Binomial Tree, GBM and MR processes')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.legend()
plt.show(fig1)

MonteCarloGBMPricesPut = np.zeros(len(Paths))
MonteCarloMRPricesPut = np.zeros(len(Paths))

BinTreePut = MRCensoredBinTree(T=63/252,eta = 6, Option = 'put')

for i in range(0, len(Paths)):
    MonteCarloGBMPricesPut[i] = MonteCarloSim(M = Paths[i], T = 63/252, Option = 'put')
    MonteCarloMRPricesPut[i] = MonteCarloSimMR(M = Paths[i], T = 63/252, eta = 6, Option = 'put')

fig2 = plt.plot(Paths, MonteCarloGBMPricesPut, label = "GBM MC")
plt.plot(Paths, MonteCarloMRPricesPut, label = 'MR MC')
plt.hlines(BinTreePut, colors = 'r', xmin = 0, xmax=10000, label = 'MR BinTree')
plt.title('Monte Carlo Put price for Idential Option using Binomial Tree, GBM and MR processes')
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.legend()
plt.show(fig2)

### Part 2: Option prices between the two mean-reverting methods with changing mean-reversion speed.

MRSpeed = np.arange(0, 20, 0.05)
#[0.5, 5, 10, 15, 20, 50, 75, 100]

MRBinTreePricesCall = np.zeros(len(MRSpeed))
MonteCarloMRPricesCall = np.zeros(len(MRSpeed))

for j in range(0, len(MRSpeed)):
    MRBinTreePricesCall[j] = MRCensoredBinTree(T = 63/252, eta = MRSpeed[j], Option = 'call')
    MonteCarloMRPricesCall[j] = MonteCarloSimMR(M = 5000, T = 63/252, eta = MRSpeed[j], Option = 'call')

fig3 = plt.plot(MRSpeed, MRBinTreePricesCall, label = "MR BinTree")
#plt.plot(MRSpeed, MonteCarloMRPricesCall, label = 'MR MC')
plt.title('Mean-Reverting Binomial Tree Call Price with Changing Mean-Reversion Speed')
plt.xlabel('Mean Reversion Speed')
plt.ylabel('Option Price')
#plt.xticks([0.5, 5, 10, 15, 20, 50, 75, 100], rotation = 'vertical')
plt.legend()
plt.show(fig3)

fig9 = plt.plot(MRSpeed, MonteCarloMRPricesCall, label = 'MR MC')
#plt.plot(MRSpeed, MRBinTreePricesCall, label = "MR BinTree")
plt.title('Mean-Reverting Monte Carlo Call Price with Changing Mean-Reversion Speed')
plt.xlabel('Mean Reversion Speed')
plt.ylabel('Option Price')
#plt.xticks([0.5, 5, 10, 15, 20, 50, 75, 100], rotation = 'vertical')
plt.legend()
plt.show(fig9)

MRBinTreePricesPut = np.zeros(len(MRSpeed))
MonteCarloMRPricesPut = np.zeros(len(MRSpeed))

for j in range(0, len(MRSpeed)):
    MRBinTreePricesPut[j] = MRCensoredBinTree(T = 63/252, eta = MRSpeed[j], Option = 'put')
    MonteCarloMRPricesPut[j] = MonteCarloSimMR(M = 5000, T = 63/252, eta = MRSpeed[j], Option = 'put')

fig4 = plt.plot(MRSpeed, MRBinTreePricesPut, label = "MR BinTree")
plt.plot(MRSpeed, MonteCarloMRPricesPut, label = 'MR MC')
plt.title('Mean-Reverting Binomial Tree Put Price with Changing Mean-Reversion Speed')
plt.xlabel('Mean Reversion Speed')
plt.ylabel('Option Price')
plt.xticks([0.5, 5, 10, 15, 20, 50, 75, 100], rotation = 'vertical')
plt.legend()
plt.show(fig4)

### Part 3: Option prices between the three methods with changing time to maturity.

TTM = [1/252, 10/252, 30/252, 50/252, 100/252, 200/252, 252/252]

MRBinTreePricesCall = np.zeros(len(TTM))
MonteCarloMRPricesCall = np.zeros(len(TTM))
MonteCarloGBMPricesCall = np.zeros(len(TTM))

for l in range(0, len(TTM)):
    MRBinTreePricesCall[l] = MRCensoredBinTree(T = TTM[l], eta = 6, Option = 'call')
    MonteCarloMRPricesCall[l] = MonteCarloSimMR(M = 5000, T = TTM[l], eta = 6, Option = 'call')
    MonteCarloGBMPricesCall[l] = MonteCarloSim(M = 5000, T = TTM[l], Option = 'call')

fig5 = plt.plot(TTM, MRBinTreePricesCall, label = "MR BinTree")
plt.plot(TTM, MonteCarloMRPricesCall, label = 'MR MC')
plt.plot(TTM, MonteCarloGBMPricesCall, label = 'GBM MC')
plt.title('Mean-Reverting Binomial Tree Call Price with Changing Time to Maturity')
plt.xlabel('Time to Maturity (Days)')
plt.ylabel('Option Price')
plt.xticks([1/252, 10/252, 30/252, 50/252, 100/252, 200/252, 252/252], labels = ['1/252', '10/252', '30/252', '50/252', '100/252', '200/252', '252/252'], rotation = 'vertical')
plt.legend()
plt.show(fig5)

#Plot again without GBM MC to see detailed behaviour of MR methods.

fig6 = plt.plot(TTM, MRBinTreePricesCall, label = "MR BinTree")
plt.plot(TTM, MonteCarloMRPricesCall, label = 'MR MC')
plt.title('Mean-Reverting Binomial Tree Call Price with Changing Time to Maturity')
plt.xlabel('Time to Maturity (Days)')
plt.ylabel('Option Price')
plt.xticks([1/252, 10/252, 30/252, 50/252, 100/252, 200/252, 252/252], labels = ['1/252', '10/252', '30/252', '50/252', '100/252', '200/252', '252/252'], rotation = 'vertical')
plt.legend()
plt.show(fig6)

MRBinTreePricesPut = np.zeros(len(TTM))
MonteCarloMRPricesPut = np.zeros(len(TTM))
MonteCarloGBMPricesPut = np.zeros(len(TTM))

for l in range(0, len(TTM)):
    MRBinTreePricesPut[l] = MRCensoredBinTree(T = TTM[l], eta = 6, Option = 'put')
    MonteCarloMRPricesPut[l] = MonteCarloSimMR(M = 5000, T = TTM[l], eta = 6, Option = 'put')
    MonteCarloGBMPricesPut[l] = MonteCarloSim(M = 5000, T = TTM[l], Option = 'put')

fig7 = plt.plot(TTM, MRBinTreePricesPut, label = "MR BinTree")
plt.plot(TTM, MonteCarloMRPricesPut, label = 'MR MC')
plt.plot(TTM, MonteCarloGBMPricesPut, label = 'GBM MC')
plt.title('Mean-Reverting Binomial Tree Put Price with Changing Time to Maturity')
plt.xlabel('Time to Maturity (Days)')
plt.ylabel('Option Price')
plt.xticks([1/252, 10/252, 30/252, 50/252, 100/252, 200/252, 252/252], labels = ['1/252', '10/252', '30/252', '50/252', '100/252', '200/252', '252/252'], rotation = 'vertical')
plt.legend()
plt.show(fig7)

#Plot again without GBM MC to see detailed behaviour of MR methods.

fig8 = plt.plot(TTM, MRBinTreePricesPut, label = "MR BinTree")
plt.plot(TTM, MonteCarloMRPricesPut, label = 'MR MC')
plt.title('Mean-Reverting Binomial Tree Put Price with Changing Time to Maturity')
plt.xlabel('Time to Maturity (Days)')
plt.ylabel('Option Price')
plt.xticks([1/252, 10/252, 30/252, 50/252, 100/252, 200/252, 252/252], labels = ['1/252', '10/252', '30/252', '50/252', '100/252', '200/252', '252/252'], rotation = 'vertical')
plt.legend()
plt.show(fig8)