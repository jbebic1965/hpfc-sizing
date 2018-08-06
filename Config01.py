#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 08:27:36 2018

@author: Jovan Z. Bebic

v1.0 JZB 20180803
Configuration file shared by SizeFACTS and ReachableSets

"""

import numpy as np

#%% Circuit parameters
fs = 50 # system frequency China
ws = 2*np.pi*fs # 

XL = ws * 0.015836 # equivalent reactance of each single line, per spec
ZL = 1.j * XL # Setting line impedance as purely inductive, per spec

Xm = ws * 0.17807 # equivalent reactance of parallel connection
Zm = 1.j * Xm 

Zs = 0.423125 + 1.j * ws * 0.011464 # Thevenin's impedance of the sending end, per spec
Zr = 2.5159  + 1.j * ws * 0.05059  # Thevenin's impedance of the receiving end, per spec

#%% Define Voltages
UbLL = 220.
Ub = UbLL/np.sqrt(3.) # L-N RMS [kV], backed out from specified flows
Us = 186.5/np.sqrt(2.)*np.exp(18.8*np.pi/180.*1.j)
Ur = 172.2/np.sqrt(2.)
