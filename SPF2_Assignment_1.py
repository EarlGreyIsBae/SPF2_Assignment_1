#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 12:34:59 2020

@author: connorstevens
"""
import pandas as pd
import numpy as np 

index = range(3, 363, 3)

#Manually input EURIBOR rates into dictionary.
EURIBOR = {'3': -0.00527,'6': -0.00512,'9': 0, '12':  -0.00483}

#Linearly interpolate for 9 month EURIBOR.
EURIBOR['9'] = np.interp(x = 9, xp = [6, 12], fp = [-0.00512, -0.00483])

#Make EURIBOR dataframe for easy handling later.
EURIBOR = pd.DataFrame.from_dict(EURIBOR, orient = 'index', dtype = float)

#Manually input swap rates up to 30 years (in months) into dictionary.
MktSwaps = {'24':  -0.0052, '36': -0.0052, '48': -0.0049, '60': -0.0046, '72': -0.0043, '84': -0.0039, '96': -0.0034, '108': -0.0030, '120': -0.0025, '144': -0.0016, '180': -0.0006, '240': -0.0003, '360': 0}

#Make Swaps dataframe for easy handling later.
MktSwaps = pd.DataFrame.from_dict(MktSwaps, orient = 'index', dtype = float)

#Empty array for to store interpolated values for swap rates.
empty = np.zeros(120)

#Dataframe for holding interpolated quarterly swap rates
Rates = pd.DataFrame(index = index)

Rates[3:12] = EURIBOR



TwoYR_swap_rate = np.log((100-2.5*np.exp(-.04*.5)-2.5*np.exp(-.045)-2.5*np.exp(-.048*1.5))/102.5)/-2

