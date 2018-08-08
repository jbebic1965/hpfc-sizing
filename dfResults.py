#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 11:17:22 2018

@author: Jovan Z. Bebic

v1.0 JZB 20180803
Data structures for holding results in SizeFACTS.py and ReachableSets.py
Data frames are index by scenario name

"""
import pandas as pd # Python data analysis library (pandas.pydata.org)

#%% Define datastructure to hold the results
dfS = pd.DataFrame(columns=['Ss', 'S0', 'S1', 'Sm', "Sm'", 'S2', 'S3', 'S4', 'Sr', 'Note'])
dfU = pd.DataFrame(columns=['Us', 'U1', 'U2', 'U3', 'Ur'])
dfUPFC = pd.DataFrame(columns=['Ssh', 'Sser', 'Ush', 'User'])
dfHPFC = pd.DataFrame(columns=['SX', 'SY', 'UM', 'UX', 'UY', 'QM'])

dfI = pd.DataFrame(columns=['Is', 'I1', 'IM', 'I2', 'I3', 'Ir', 'Im'])

