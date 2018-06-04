#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 08:46:34 2018

@author: Jovan Z. Bebic

v1.0 JZB 20180603
Sizing calculations for HPFC. The '_01' in file name designates the application 
scenario. The sizing methodology is based on considering HPFC as a two terminal 
device and setting up load flow equations to solve for voltages and current at 
the equipment terminals. The solution is then resolved into converter ratings 
based on the type of FACTS device used. The load flow solver is from PyPSA 
package.

"""

#%% Import packages
import numpy as np # Package for scientific compyting (www.numpy.org)
import pypsa # Python for Power System Analysis (www.pypsa.org/)

import matplotlib.backends.backend_pdf as dpdf
import matplotlib.pyplot as plt

from datetime import datetime # time stamps
import os # operating system interface

#%% Document outputs
codeVersion = '1.0'
codeCopyright = 'GNU General Public License v3.0'
codeAuthors = 'Jovan Z Bebic\n'
codeName = 'SizeHPFC_01.py' # _01 designates an application scnario
dirout = './'
fnameLog = 'SizeHPFC_01.log'
fnamePlt = 'SizeHPFC_01.pdf'

OutputPlots = True # set to False if you don't want the pdf file output
OutputPSA = False # set to False to ommit outputting PyPSA information into log file

#%% Plotting functions
def OutputVectorsPage(pltPdf, Vs, V1, V2, V3, Vr, Is, I1, Il, Im, Ir, Iscale=1.0, VM_hpfc=np.nan, IM_hpfc=np.nan, pageTitle=''):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,10)) # , sharex=True
    
    # plt.rc('text', usetex=True)
    fig.suptitle(pageTitle, fontsize=14) # This titles the figure
    
    # Setting up voltage vectors
    if np.isnan(VM_hpfc):
        vTips = np.array([Vs, V1, V2, V3, Vr]) # coordinates of vector tips (complex numbers)
        vTails = np.zeros_like(vTips) # coordinates of vector tails
        vColors = ['k', 'r', 'b', 'k', 'k'] # colors of vectors
    else:
        vTips = np.array([Vs, VM_hpfc, V1-VM_hpfc, V2-VM_hpfc, V3, Vr]) # coordinates of vector tips (complex numbers)
        vTails = np.array([0, 0, VM_hpfc, VM_hpfc, 0, 0]) # coordinates of vector tails
        vColors = ['k', 'g', 'r', 'b', 'k', 'k'] # colors of vectors
        
    ax[0,0].set_title('Voltages [pu]')
    q00 = ax[0,0].quiver(np.real(vTails), np.imag(vTails), 
                    np.real(vTips), np.imag(vTips), 
                    angles='xy', scale_units='xy', scale=1.,
                    color=vColors)
    if np.isnan(VM_hpfc):
        ax[0,0].quiverkey(q00, 0.2, 0.9, 0.1, 'V1', labelpos='E', color='r', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.8, 0.1, 'V2', labelpos='E', color='b', coordinates='axes')
    else:
        ax[0,0].quiverkey(q00, 0.2, 0.9, 0.1, 'VX', labelpos='E', color='r', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.8, 0.1, 'VY', labelpos='E', color='b', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.7, 0.1, 'VM', labelpos='E', color='g', coordinates='axes')
    
    ax[0,0].set_xlim([-0.1, 1.1])
    ax[0,0].set_ylim([-0.1, 1.1])
    ax[0,0].set_aspect('equal')
    ax[0,0].grid(True, which='both')

    ax[0,1].set_title('Voltages Zoomed-in')
    q01 = ax[0,1].quiver(np.real(vTails), np.imag(vTails), 
                    np.real(vTips), np.imag(vTips), 
                    angles='xy', scale_units='xy', scale=1.,
                    color=vColors)
    if np.isnan(VM_hpfc):
        ax[0,1].quiverkey(q01, 0.2, 0.9, 0.1*0.4/1.2, 'V1', labelpos='E', color='r', coordinates='axes')
        ax[0,1].quiverkey(q01, 0.2, 0.8, 0.1*0.4/1.2, 'V2', labelpos='E', color='b', coordinates='axes')
    else:
        ax[0,1].quiverkey(q01, 0.2, 0.9, 0.1*0.4/1.2, 'VX', labelpos='E', color='r', coordinates='axes')
        ax[0,1].quiverkey(q01, 0.2, 0.8, 0.1*0.4/1.2, 'VY', labelpos='E', color='b', coordinates='axes')
        ax[0,1].quiverkey(q01, 0.2, 0.7, 0.1*0.4/1.2, 'VM', labelpos='E', color='g', coordinates='axes')
    
    ax[0,1].set_xlim([0.7, 1.1])
    ax[0,1].set_ylim([0.1, 0.5])
    ax[0,1].set_aspect('equal')
    ax[0,1].grid(True, which='both')

    # Setting up current vectors
    if np.isnan(IM_hpfc):
        iTips = np.array([Is, I1, 2.*Il, Im, Ir]) # coordinates of vector tips (complex numbers)
        iTails = np.zeros_like(iTips) # coordinates of vector tails
        iColors = ['k', 'r', 'b', 'k', 'k'] # colors of vectors
    else:
        iTips = np.array([Is, I1, 2.*Il, Im, IM_hpfc, Ir]) # coordinates of vector tips (complex numbers)
        iTails = np.array([0, 0, 0, 0, 2.*Il/Iscale, 0]) # coordinates of vector tails
        iColors = ['k', 'r', 'b', 'k', 'g', 'k'] # colors of vectors
    
    ax[1,0].set_title('Currents [pu] Ib=' + str(Iscale) + 'kA')
    q10 = ax[1,0].quiver(np.real(iTails), np.imag(iTails), 
                    np.real(iTips), np.imag(iTips), 
                    angles='xy', scale_units='xy', scale=Iscale,
                    color=iColors)
    if np.isnan(IM_hpfc):
        ax[1,0].quiverkey(q10, 0.2, 0.9, 0.1*Iscale, '2*IL', labelpos='E', color='b', coordinates='axes')
        ax[1,0].quiverkey(q10, 0.2, 0.8, 0.1*Iscale, 'I1', labelpos='E', color='r', coordinates='axes')
    else:
        ax[1,0].quiverkey(q10, 0.2, 0.9, 0.1*Iscale, '2*IL', labelpos='E', color='b', coordinates='axes')
        ax[1,0].quiverkey(q10, 0.2, 0.8, 0.1*Iscale, 'I1', labelpos='E', color='r', coordinates='axes')
        ax[1,0].quiverkey(q10, 0.2, 0.7, 0.1*Iscale, 'IM', labelpos='E', color='g', coordinates='axes')
    
    ax[1,0].set_xlim([-0.1, 1.1])
    ax[1,0].set_ylim([-0.2, 0.7])
    ax[1,0].set_aspect('equal')
    ax[1,0].grid(True, which='both')

    ax[1,1].set_title('Currents Zoomed-in')
    q11 = ax[1,1].quiver(np.real(iTails), np.imag(iTails), 
                    np.real(iTips), np.imag(iTips), 
                    angles='xy', scale_units='xy', scale=Iscale,
                    color=iColors)
    if np.isnan(IM_hpfc):
        ax[1,1].quiverkey(q11, 0.2, 0.9, 0.1*Iscale*0.7/1.2, 'I1', labelpos='E', color='r', coordinates='axes')
        ax[1,1].quiverkey(q11, 0.2, 0.8, 0.1*Iscale*0.7/1.2, '2*IL', labelpos='E', color='b', coordinates='axes')
    else:
        ax[1,1].quiverkey(q11, 0.2, 0.9, 0.1*Iscale*0.7/1.2, 'I1', labelpos='E', color='r', coordinates='axes')
        ax[1,1].quiverkey(q11, 0.2, 0.8, 0.1*Iscale*0.7/1.2, '2*IL', labelpos='E', color='b', coordinates='axes')
        ax[1,1].quiverkey(q11, 0.2, 0.7, 0.1*Iscale*0.7/1.2, 'IM', labelpos='E', color='g', coordinates='axes')
    
    ax[1,1].set_xlim([0.4, 1.1])
    ax[1,1].set_ylim([-0.1, 0.5])
    ax[1,1].set_aspect('equal')
    ax[1,1].grid(True, which='both')
    
    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    
    return

#%% Capture start time of code execution and open log file
codeTstart = datetime.now()
foutLog = open(os.path.join(dirout, fnameLog), 'w')

#%% Output log file header information
print('This is %s v%s' %(codeName, codeVersion))
foutLog.write('This is %s v%s\n' %(codeName, codeVersion))
foutLog.write('%s\n' %(codeCopyright))
foutLog.write('%s\n' %(codeAuthors))
foutLog.write('Run started on: %s\n\n' %(str(codeTstart)))

#%% Input data
fs = 50 # system frequency
ws = 2*np.pi*fs

Ub = 220./np.sqrt(3.) # L-N RMS [kV], backed out from specified flows

Xl = ws * 0.015836 # equivalent reactance of each single line, specified
Zl = 1.j * Xl # Setting line impedance as purely inductive, per spec

Xm = ws * 0.17807 # equivalent reactance of parallel connection
Zm = Xm/7.36 + 1.j * Xm # Using a typical X/R value for 220kV system to set Rm. It helps match the specified Q of 79MVAr

Zs = 0.423125 + 1.j * ws * 0.011464 # Thevenin's impedance of the sending end, per spec
Zr = 2.5159  + 1.j * ws * 0.05059  # Thevenin's impedance of the receiving end, per spec

#%% Voltages at sending and receiving end, per spec
angle_s = 18.8 * np.pi / 180. 
Us = 186.5/np.sqrt(2.)*np.exp(1.j*angle_s)
Ur = 172.2/np.sqrt(2.)

#%% Cross checking versus specifed P and Q flows
Z12 = 1./(1./Zm + 1./Zl + 1./Zl)
Is = (Us-Ur)/(Zs + Z12 + Zr)
Ir = Is

U1 = Us - Zs*Is # voltage at Bus1
U2 = U1 # No FACTS compensator
U3 = Ur + Zr*Ir

Il = (U2-U3)/Zl # Current flows
Im = (U1-U3)/Zm

Ss = 3.*Us*np.conj(Is) # Apparent powers
Sl2 = 3.*U2*np.conj(Il)
Sl3 = 3.*U3*np.conj(Il)
Srin = 3.*U3*np.conj(Is)
Srout = 3.*Ur*np.conj(Is)
Sm = 3.*U1*np.conj(Im)

print('Calculating baseline flows')
foutLog.write('Before compensation\n')
foutLog.write('Power flows\n')
foutLog.write('  Ps = %.2f\n' %(np.real(Ss))) # Formatted results
foutLog.write('  Qs = %.2f\n\n' %(np.imag(Ss)))

foutLog.write('  Pl2 = %.2f\n' %(np.real(Sl2)))
foutLog.write('  Ql2 = %.2f\n\n' %(np.imag(Sl2)))

foutLog.write('  Pl3 = %.2f\n' %(np.real(Sl3)))
foutLog.write('  Ql3 = %.2f\n\n' %(np.imag(Sl3)))

foutLog.write('  Pm = %.2f\n' %(np.real(Sm)))
foutLog.write('  Qm = %.2f\n\n' %(np.imag(Sm)))

foutLog.write('  Prin = %.2f\n' %(np.real(Srin)))
foutLog.write('  Qrin = %.2f\n' %(np.imag(Srin)))

foutLog.write('  Prout = %.2f\n' %(np.real(Srout)))
foutLog.write('  Qrout = %.2f\n\n' %(np.imag(Srout)))

if False: # additional info
    foutLog.write('Reactive consumption of each Xl circuit\n')
    foutLog.write('  Ql1-Ql3 = %.2f\n' %(3. * np.imag((U1-U3)*np.conj(Il))))
    foutLog.write('  Ql1-Ql3 = %.2f\n\n' %(np.imag(Sl2-Sl3)))
    
    foutLog.write('Losses\n')
    foutLog.write('  Prin-Prout = %.2f\n' %(np.real(Srin-Srout)))
    foutLog.write('  Rr*Ir^2 = %.2f\n\n' %(3.*np.real(Zr)*np.abs(Is)**2.))

foutLog.write('Voltage values\n')
foutLog.write('  |Us| = %.4f pu, ang(Us) = %.2f deg\n' %(np.abs(Us)/Ub, np.angle(Us, deg=True)))
foutLog.write('  |U1| = %.4f pu, ang(U1) = %.2f deg\n' %(np.abs(U1)/Ub, np.angle(U1, deg=True)))
foutLog.write('  |U3| = %.4f pu, ang(U3) = %.2f deg\n' %(np.abs(U3)/Ub, np.angle(U3, deg=True)))
foutLog.write('  |Ur| = %.4f pu, ang(Ur) = %.2f deg\n\n' %(np.abs(Ur)/Ub, np.angle(Ur, deg=True)))

foutLog.write('Current values\n')
foutLog.write('  |Is| = %.4f kA, ang(Is) = %.2f deg\n' %(np.abs(Is), np.angle(Is, deg=True)))
foutLog.write('  |Il| = %.4f kA, ang(Il) = %.2f deg\n' %(np.abs(Il), np.angle(Il, deg=True)))
foutLog.write('  |Im| = %.4f kA, ang(Im) = %.2f deg\n\n' %(np.abs(Im), np.angle(Im, deg=True)))

#%% Specifying target flows for after compensation, 't' added to subscript of variables to denote 'target'
Plt = 450. # Target P flow through XL in MW
Qlt = 100. # Target Q flow through XL in MVAr

# Target PQQ values at HPFC terminals
Phpfc = 2*Plt # double the target of single line
Q2hpfc = 2*Qlt # double the target of single line
Q1hpfc = 30. # specified value

#%% Load flow calculations using PyPSA
print('Calculating target flows using PyPSA')
network = pypsa.Network()

#add the buses
network.add("Bus","Us", v_nom=220., v_mag_pu_set=1.0382) # PV bus, voltage set here, P set in the corresponding generator
network.add("Bus","U1", v_nom=220.)
network.add("Bus","U2", v_nom=220.)
network.add("Bus","U3", v_nom=220.)
network.add("Bus","Ur", v_nom=220., v_mag_pu_set=0.9586) # PV bus

network.add("Line", "Zs", bus0="Us", bus1="U1", x=np.imag(Zs), r=np.real(Zs))
network.add("Line", "Zm", bus0="U1", bus1="U3", x=np.imag(Zm), r=np.real(Zm))
network.add("Line", "Zl1", bus0="U2", bus1="U3", x=np.imag(Zl), r=np.real(Zl))
network.add("Line", "Zl2", bus0="U2", bus1="U3", x=np.imag(Zl), r=np.real(Zl))
network.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))

#%% set generators
if False: # Set to True to validate the solution with baseline flows
    print('Note: PyPSA used for validation of baseline flows')
    network.add("Generator", "Us", bus="Us", control="PV", p_set=np.real(Ss))
    network.add("Generator", "U1", bus="U1", control="PQ", p_set=-2.*np.real(Sl2), q_set=-2.*np.imag(Sl2))
    network.add("Generator", "U2", bus="U2", control="PQ", p_set=2.*np.real(Sl2), q_set=2.*np.imag(Sl2))
    network.add("Generator", "Ur", bus="Ur", control="Slack", p_set=-np.real(Srout))
else: # the target flows after compensation
    network.add("Generator", "Us", bus="Us", control="PV", p_set=Phpfc+np.real(Sm))
    network.add("Generator", "U1", bus="U1", control="PQ", p_set=-Phpfc, q_set=Q1hpfc)
    network.add("Generator", "U2", bus="U2", control="PQ", p_set=Phpfc, q_set=Q2hpfc)
    network.add("Generator", "Ur", bus="Ur", control="Slack", p_set=-Phpfc-np.real(Sm))

#%% Record what was entered into PyPSA
if OutputPSA:
    foutLog.write(network.generators)
    foutLog.write(network.buses)
    foutLog.write(network.lines)
    foutLog.write(network.loads)

#%% Solve load flow
network.pf()

#%% Output results
if OutputPSA: # set to True to see output from PyPSA
    foutLog.write(network.lines_t.p0)
    foutLog.write(network.lines_t.q0)
    
    foutLog.write(network.buses_t.v_mag_pu)
    foutLog.write(network.buses_t.v_ang*180/np.pi)
    
    foutLog.write(network.generators_t.p)
    foutLog.write(network.generators_t.q)

# retrieve solved voltages from load flow
print('Calculating currents and flows using voltage solutions from PyPSA')
Ust = Ub*network.buses_t['v_mag_pu']['Us']['now']*np.exp(1.j*network.buses_t['v_ang']['Us']['now'])
U1t = Ub*network.buses_t['v_mag_pu']['U1']['now']*np.exp(1.j*network.buses_t['v_ang']['U1']['now'])
U2t = Ub*network.buses_t['v_mag_pu']['U2']['now']*np.exp(1.j*network.buses_t['v_ang']['U2']['now'])
U3t = Ub*network.buses_t['v_mag_pu']['U3']['now']*np.exp(1.j*network.buses_t['v_ang']['U3']['now'])
Urt = Ub*network.buses_t['v_mag_pu']['Ur']['now']*np.exp(1.j*network.buses_t['v_ang']['Ur']['now'])

# Recalculate power flows to cross-check
Ist = (Ust-U1t)/Zs # Current flows
Ilt = (U2t-U3t)/Zl
Imt = (U1t-U3t)/Zm
Irt = (U3t-Urt)/Zr
I1t = Ist-Imt # Current entering V1

Sst = 3.*Ust*np.conj(Ist) # Apparent powers
Sl1t = 3.*U1t*np.conj(I1t)
Sl2t = 3.*U2t*np.conj(Ilt)
Sl3t = 3.*U3t*np.conj(Ilt)
Srtin = 3.*U3t*np.conj(Irt)
Srtout = 3.*Urt*np.conj(Irt)
Smt = 3.*U1t*np.conj(Imt)

foutLog.write('After compensation\n')
foutLog.write('Power flows\n')
foutLog.write('  Pst = %.2f\n' %(np.real(Sst))) # Formatted results
foutLog.write('  Qst = %.2f\n\n' %(np.imag(Sst)))

foutLog.write('  Pl1t = %.2f\n' %(np.real(Sl1t)))
foutLog.write('  Ql1t = %.2f\n\n' %(np.imag(Sl1t)))

foutLog.write('  Pl2t = %.2f\n' %(np.real(Sl2t)))
foutLog.write('  Ql2t = %.2f\n\n' %(np.imag(Sl2t)))

foutLog.write('  Pl3t = %.2f\n' %(np.real(Sl3t)))
foutLog.write('  Ql3t = %.2f\n\n' %(np.imag(Sl3t)))

foutLog.write('  Pmt = %.2f\n' %(np.real(Smt)))
foutLog.write('  Qmt = %.2f\n\n' %(np.imag(Smt)))

foutLog.write('  Prtin = %.2f\n' %(np.real(Srtin)))
foutLog.write('  Qrtin = %.2f\n' %(np.imag(Srtin)))

foutLog.write('  Prtout = %.2f\n' %(np.real(Srtout)))
foutLog.write('  Qrtout = %.2f\n\n' %(np.imag(Srtout)))

foutLog.write('Voltage values\n')
foutLog.write('  |Ust| = %.4f pu, ang(Ust) = %.2f deg\n' %(np.abs(Ust)/Ub, np.angle(Ust, deg=True)))
foutLog.write('  |U1t| = %.4f pu, ang(U1t) = %.2f deg\n' %(np.abs(U1t)/Ub, np.angle(U1t, deg=True)))
foutLog.write('  |U2t| = %.4f pu, ang(U2t) = %.2f deg\n' %(np.abs(U2t)/Ub, np.angle(U2t, deg=True)))
foutLog.write('  |U3t| = %.4f pu, ang(U3t) = %.2f deg\n' %(np.abs(U3t)/Ub, np.angle(U3t, deg=True)))
foutLog.write('  |Urt| = %.4f pu, ang(Urt) = %.2f deg\n\n' %(np.abs(Urt)/Ub, np.angle(Urt, deg=True)))

foutLog.write('Current values\n')
foutLog.write('  |Ist| = %.4f kA, ang(Ist) = %.2f deg\n' %(np.abs(Ist), np.angle(Ist, deg=True)))
foutLog.write('  |I1t| = %.4f kA, ang(I1t) = %.2f deg\n' %(np.abs(I1t), np.angle(I1t, deg=True)))
foutLog.write('  |Ilt| = %.4f kA, ang(Ilt) = %.2f deg\n' %(np.abs(Ilt), np.angle(Ilt, deg=True)))
foutLog.write('  |Imt| = %.4f kA, ang(Imt) = %.2f deg\n' %(np.abs(Imt), np.angle(Imt, deg=True)))
foutLog.write('  |Irt| = %.4f kA, ang(Irt) = %.2f deg\n\n' %(np.abs(Irt), np.angle(Irt, deg=True)))

#%% Sizing calculations for HPFC
print('Running sizing calculations for HPFC')
Qt_hpfc = np.imag(Sl1t)-2.*np.imag(Sl2t) # total Q consumed by HPFC
IM_hpfc = I1t - 2.*Ilt

BM_hpfc = np.abs(IM_hpfc)/((np.abs(U1t)+np.abs(U1t))/2.) # Selecting sufficient BM to hit (|U1t|+|U2t|)/2 with available IM_hpfc

UM_hpfc = IM_hpfc/(1.j*BM_hpfc)
QM_hpfc = -3.*BM_hpfc*np.abs(UM_hpfc)**2
# QM2_hpfc = np.imag(3.*UM_hpfc*np.conj(IM_hpfc))
    
foutLog.write('HPFC Ratings\n')
foutLog.write('  Qt_hpfc = %.2f\n' %(Qt_hpfc))
foutLog.write('  QM_hpfc = %.2f\n' %(QM_hpfc))
# foutLog.write('  QM2_hpfc = %.2f\n' %(QM2_hpfc))
foutLog.write('  |IM_hpfc| = %.4f kA, ang(IM_hpfc) = %.2f deg\n' %(np.abs(IM_hpfc), np.angle(IM_hpfc, deg=True)))
foutLog.write('  |UM_hpfc| = %.4f pu, ang(UM_hpfc) = %.2f deg\n' %(np.abs(UM_hpfc)/Ub, np.angle(UM_hpfc, deg=True)))

UX_hpfc = U1t-UM_hpfc
UY_hpfc = U2t-UM_hpfc

foutLog.write('  |UX_hpfc| = %.4f pu, ang(UX_hpfc) = %.2f deg\n' %(np.abs(UX_hpfc)/Ub, np.angle(UX_hpfc, deg=True)))
foutLog.write('  |UY_hpfc| = %.4f pu, ang(UY_hpfc) = %.2f deg\n\n' %(np.abs(UY_hpfc)/Ub, np.angle(UY_hpfc, deg=True)))

SX_hpfc = 3.*UX_hpfc*np.conj(I1t)
SY_hpfc = 3.*UY_hpfc*np.conj(-2.*Ilt)

foutLog.write('  PX_hpfc = %.2f\n' %(np.real(SX_hpfc)))
foutLog.write('  QX_hpfc = %.2f\n' %(np.imag(SX_hpfc)))
foutLog.write('  SX_hpfc = %.2f\n\n' %(np.abs(SX_hpfc)))

foutLog.write('  PY_hpfc = %.2f\n' %(np.real(SY_hpfc)))
foutLog.write('  QY_hpfc = %.2f\n' %(np.imag(SY_hpfc)))
foutLog.write('  SY_hpfc = %.2f\n\n' %(np.abs(SY_hpfc)))

foutLog.write('  QM+QX+QY = %.2f\n\n' %(QM_hpfc+np.imag(SX_hpfc)+np.imag(SY_hpfc)))

#%% Sizing calculations for UPFC
print('Running sizing calculations for UPFC')
Ssh_upfc = 3.*U1t*np.conj(I1t-2.*Ilt)
Sser_upfc = 3.*(U2t-U1t)*np.conj(-2.*Ilt)

foutLog.write('UPFC Ratings\n')
foutLog.write('  Psh_upfc = %.2f\n' %(np.real(Ssh_upfc)))
foutLog.write('  Qsh_upfc = %.2f\n' %(np.imag(Ssh_upfc)))
foutLog.write('  Ssh_upfc = %.2f\n\n' %(np.abs(Ssh_upfc)))
foutLog.write('  Pser_upfc = %.2f\n' %(np.real(Sser_upfc)))
foutLog.write('  Qser_upfc = %.2f\n' %(np.imag(Sser_upfc)))
foutLog.write('  Sser_upfc = %.2f\n\n' %(np.abs(Sser_upfc)))
foutLog.write('  Qsh+Qser = %.2f\n\n' %(np.imag(Ssh_upfc)+np.imag(Sser_upfc)))

#%% Preparing pdf file for plotting
if OutputPlots:
    foutLog.write('Starting to plot at: %s\n' %(str(datetime.now())))
    print('Opening plot files')     
    pltPdf1 = dpdf.PdfPages(os.path.join(dirout,fnamePlt))

if OutputPlots:
    # OutputVectorsPage(pltPdf1, Us/Ub, U1/Ub, U2/Ub, U3/Ub, Ur/Ub, Is, Is-Im, Il, Im, Ir, Iscale=2.*1.3, 
    #                  pageTitle='Calculated by '+codeName+' v'+codeVersion+'\n\n' + r'$\bf{System\ Before\ Compensation}$')
    OutputVectorsPage(pltPdf1, Ust/Ub, U1t/Ub, U2t/Ub, U3t/Ub, Urt/Ub, Ist, I1t, Ilt, Imt, Irt, Iscale=2.*1.3, VM_hpfc=UM_hpfc/Ub, IM_hpfc=IM_hpfc, 
                      pageTitle='Calculated by '+codeName+' v'+codeVersion+'\n\n'+r'$\bf{System\ Compensated\ by\ an\ HPFC}$')

#%% Closing plot files
if OutputPlots:
    print("Closing plot files")
    pltPdf1.close()

#%% time stamp and close log file
codeTfinish = datetime.now()
foutLog.write('\nRun finished at: %s\n' %(str(codeTfinish)))
codeTdelta = codeTfinish - codeTstart
foutLog.write('Run Lasted: %.3f seconds\n' %(codeTdelta.total_seconds()))
foutLog.close()
