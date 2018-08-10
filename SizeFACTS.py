#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 18:53:53 2018

@author: Jovan Z. Bebic

v1.51 JZB 201808008
Added logging of the system operating point, enabled use of snapshots in 
AddGeneratorsUsUr and AddGenerators4PQQComp and tested it using alpha for sensitivity, 
finalized the case (moved State11 calculations to later in the code, added an SVC case,
deleted hand-adjusted HPFC cases used before)

v1.5 JZB 20180807-08
Added algorithmic adjustment of the HPFC operating point, and algorithmic choice 
of a system operating point compatible to HPFC. Minor cleanup of other code

v1.4 JZB 20180806
Turned it into a module:
    - moved input definitions into the imported Config01
    - moved dataframes into dfResults

v1.31 JZB 20180806
Added a function to define voltages

v1.3 JZB 20180806
Added output of HPFC and UPFC operating points to log file

v1.2 JZB 20180805
Added target flows functionality and finalized the cases

v1.1 JZB 20180805
Added PVV and SVC specification for the compensator, ability to use snapshots,
and filtering of cases saved to Excel.

v1.0 JZB 20180803
Consolidated all earlier sizing code to facilitiate scenario analysis
- Scenario specifications and results are held in dataframes
- Results are saved to an excel spreadsheet to facilitate cross-checking and comparison
- Plots adjusted to Vsh and Vser in UPFC cases

"""
#%% Import packages
import numpy as np # Package for scientific compyting (www.numpy.org)
import pypsa # Python for Power System Analysis (www.pypsa.org)
import pandas as pd # Python data analysis library (pandas.pydata.org)

import matplotlib.backends.backend_pdf as dpdf
import matplotlib.pyplot as plt

from datetime import datetime # time stamps
import os # operating system interface

# Circuit parameters, specified voltages and apparent powers
# Apparent powers are given as dictionaries containing specifed values at various locations in the circuit
from Config01 import fs, ZL, Zm, Zs, Zr, \
                     Ub, Us, Ur, \
                     State01s, State02s

# Data structures for storing the results of different scenarios
from dfResults import dfS, dfU, dfUPFC, dfHPFC

def OutputVectorsPage(pltPdf, caseIx, Iscale=2*1300, pageTitle = ''):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,10)) # , sharex=True
    
    if pageTitle == '': pageTitle = caseIx
    fig.suptitle(pageTitle, fontsize=14) # This titles the figure
    
    Vs = dfU.Us[caseIx]
    V1 = dfU.U1[caseIx]
    V2 = dfU.U2[caseIx]
    V3 = dfU.U3[caseIx]
    Vr = dfU.Ur[caseIx]
    # Setting up voltage vectors
    if caseIx in dfHPFC.index:
        VM = dfHPFC.UM[caseIx]
        vTips = np.array([Vs, VM, V1-VM, V2-VM, V3, Vr]) # coordinates of vector tips (complex numbers)
        vTails = np.array([0, 0, VM, VM, 0, 0]) # coordinates of vector tails
        vColors = ['k', 'g', 'r', 'b', 'k', 'k'] # colors of vectors
    elif caseIx in dfUPFC.index:
        vTips = np.array([Vs, V1, V2-V1, V3, Vr]) # coordinates of vector tips (complex numbers)
        vTails = np.array([0, 0, V1, 0, 0]) # coordinates of vector tails
        vColors = ['k', 'r', 'b', 'k', 'k'] # colors of vectors
    else:
        vTips = np.array([Vs, V1, V2, V3, Vr]) # coordinates of vector tips (complex numbers)
        vTails = np.zeros_like(vTips) # coordinates of vector tails
        vColors = ['k', 'r', 'b', 'k', 'k'] # colors of vectors
        
    ax[0,0].set_title('Voltages [pu]')
    q00 = ax[0,0].quiver(np.real(vTails), np.imag(vTails), 
                    np.real(vTips), np.imag(vTips), 
                    angles='xy', scale_units='xy', scale=1.,
                    color=vColors)
    if caseIx in dfHPFC.index:
        ax[0,0].quiverkey(q00, 0.2, 0.9, 0.1, 'VX', labelpos='E', color='r', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.8, 0.1, 'VY', labelpos='E', color='b', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.7, 0.1, 'VM', labelpos='E', color='g', coordinates='axes')
    elif caseIx in dfUPFC.index:
        ax[0,0].quiverkey(q00, 0.2, 0.9, 0.1, 'Vsh', labelpos='E', color='r', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.8, 0.1, 'Vser', labelpos='E', color='b', coordinates='axes')
    else:
        ax[0,0].quiverkey(q00, 0.2, 0.9, 0.1, 'V1', labelpos='E', color='r', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.8, 0.1, 'V2', labelpos='E', color='b', coordinates='axes')
    
    ax[0,0].set_xlim([-0.1, 1.1])
    ax[0,0].set_ylim([-0.1, 1.1])
    ax[0,0].set_aspect('equal')
    ax[0,0].grid(True, which='both')

    ax[0,1].set_title('Voltages Zoomed-in')
    q01 = ax[0,1].quiver(np.real(vTails), np.imag(vTails), 
                    np.real(vTips), np.imag(vTips), 
                    angles='xy', scale_units='xy', scale=1.,
                    color=vColors)
    if caseIx in dfHPFC.index:
        ax[0,1].quiverkey(q01, 0.2, 0.9, 0.1*0.4/1.2, 'VX', labelpos='E', color='r', coordinates='axes')
        ax[0,1].quiverkey(q01, 0.2, 0.8, 0.1*0.4/1.2, 'VY', labelpos='E', color='b', coordinates='axes')
        ax[0,1].quiverkey(q01, 0.2, 0.7, 0.1*0.4/1.2, 'VM', labelpos='E', color='g', coordinates='axes')
    elif caseIx in dfUPFC.index:
        ax[0,0].quiverkey(q00, 0.2, 0.9, 0.1, 'Vsh', labelpos='E', color='r', coordinates='axes')
        ax[0,0].quiverkey(q00, 0.2, 0.8, 0.1, 'Vser', labelpos='E', color='b', coordinates='axes')
    else:
        ax[0,1].quiverkey(q01, 0.2, 0.9, 0.1*0.4/1.2, 'V1', labelpos='E', color='r', coordinates='axes')
        ax[0,1].quiverkey(q01, 0.2, 0.8, 0.1*0.4/1.2, 'V2', labelpos='E', color='b', coordinates='axes')
    
    ax[0,1].set_xlim([0.7, 1.1])
    ax[0,1].set_ylim([0.1, 0.5])
    ax[0,1].set_aspect('equal')
    ax[0,1].grid(True, which='both')

    # Setting up current vectors
    Ss = dfS.Ss[caseIx] # apparent powers in MW
    S1 = dfS.S1[caseIx]
    S2 = dfS.S2[caseIx]
    Sr = dfS.Sr[caseIx]
    Us = dfU.Us[caseIx]*Ub # voltages in kV
    U1 = dfU.U1[caseIx]*Ub
    U2 = dfU.U2[caseIx]*Ub
    Ur = dfU.Ur[caseIx]*Ub

    Is = np.conj(Ss/(3.*Us))*1000. # currents in amperes
    I1 = np.conj(-S1/(3.*U1))*1000.
    I2 = np.conj(S2/(3.*U2))*1000.
    Im = Is - I1
    Ir = np.conj(Sr/(3.*Ur))*1000.
    IM = I1-I2
    
    if caseIx in dfHPFC.index:
        iTips = np.array([Is, I1, I2, Im, IM, Ir]) # coordinates of vector tips (complex numbers)
        iTails = np.array([0, 0, 0, 0, I2/Iscale, 0]) # coordinates of vector tails
        iColors = ['k', 'r', 'b', 'k', 'g', 'k'] # colors of vectors
    else:
        iTips = np.array([Is, I1, I2, Im, Ir]) # coordinates of vector tips (complex numbers)
        iTails = np.zeros_like(iTips) # coordinates of vector tails
        iColors = ['k', 'r', 'b', 'k', 'k'] # colors of vectors
    
    ax[1,0].set_title('Currents [pu] Ib=' + str(Iscale) + 'kA')
    q10 = ax[1,0].quiver(np.real(iTails), np.imag(iTails), 
                    np.real(iTips), np.imag(iTips), 
                    angles='xy', scale_units='xy', scale=Iscale,
                    color=iColors)
    if caseIx in dfHPFC.index:
        ax[1,0].quiverkey(q10, 0.2, 0.9, 0.1*Iscale, 'I2', labelpos='E', color='b', coordinates='axes')
        ax[1,0].quiverkey(q10, 0.2, 0.8, 0.1*Iscale, 'I1', labelpos='E', color='r', coordinates='axes')
        ax[1,0].quiverkey(q10, 0.2, 0.7, 0.1*Iscale, 'IM', labelpos='E', color='g', coordinates='axes')
    else:
        ax[1,0].quiverkey(q10, 0.2, 0.9, 0.1*Iscale, 'I2', labelpos='E', color='b', coordinates='axes')
        ax[1,0].quiverkey(q10, 0.2, 0.8, 0.1*Iscale, 'I1', labelpos='E', color='r', coordinates='axes')
    
    ax[1,0].set_xlim([-0.1, 1.1])
    ax[1,0].set_ylim([-0.2, 0.8])
    ax[1,0].set_aspect('equal')
    ax[1,0].grid(True, which='both')

    ax[1,1].set_title('Currents Zoomed-in')
    q11 = ax[1,1].quiver(np.real(iTails), np.imag(iTails), 
                    np.real(iTips), np.imag(iTips), 
                    angles='xy', scale_units='xy', scale=Iscale,
                    color=iColors)
    if caseIx in dfHPFC.index:
        ax[1,1].quiverkey(q11, 0.2, 0.9, 0.1*Iscale*0.7/1.2, 'I1', labelpos='E', color='r', coordinates='axes')
        ax[1,1].quiverkey(q11, 0.2, 0.8, 0.1*Iscale*0.7/1.2, 'I2', labelpos='E', color='b', coordinates='axes')
        ax[1,1].quiverkey(q11, 0.2, 0.7, 0.1*Iscale*0.7/1.2, 'IM', labelpos='E', color='g', coordinates='axes')
    else:
        ax[1,1].quiverkey(q11, 0.2, 0.9, 0.1*Iscale*0.7/1.2, 'I1', labelpos='E', color='r', coordinates='axes')
        ax[1,1].quiverkey(q11, 0.2, 0.8, 0.1*Iscale*0.7/1.2, 'I2', labelpos='E', color='b', coordinates='axes')
    
    ax[1,1].set_xlim([0.6, 1.0]) # [0.4, 1.1]
    ax[1,1].set_ylim([0.0, 0.3]) # [-0.1, 0.5]
    ax[1,1].set_aspect('equal')
    ax[1,1].grid(True, which='both')
    
    pltPdf.savefig() # Saves fig to pdf
    plt.close() # Closes fig to clean up memory
    
    return

#%% Save the results to Excel
def SaveToExcel(caselist, dirout='./', fnameXlsx='Results.xlsx', SortCases=False):

    # opening an excel file 
    writer = pd.ExcelWriter(os.path.join(dirout, fnameXlsx), engine='xlsxwriter')

    # Recording code version info
    workbook = writer.book
    worksheet1 = workbook.add_worksheet('Intro')
    worksheet1.write('A1', codeName+' v'+codeVersion)
    worksheet1.write('A2', codeCopyright)
    worksheet1.write('A3', codeAuthors)
    worksheet1.write('A4', 'Ran on %s' %str(codeTstart))
    row = 5
    dfSNf = dfS.loc[dfS.index.intersection(caselist), ['Note']]
    if SortCases: dfSNf.sort_index(inplace=True)
    for case in dfSNf.index.tolist():
        worksheet1.write_string(row, 0, case)
        worksheet1.write_string(row, 1, dfS.Note[case])
        row +=1

    # Preparing real and reactive power flows
    dfSf = dfS.loc[dfS.index.intersection(caselist), ['Ss', 'S0', 'S1', 'Sm', "Sm'", 'S2', 'S3', 'S4', 'Sr']]
    if SortCases: dfSf.sort_index(inplace=True)
    dfP = dfSf.applymap(lambda x: np.real(x)).round(1)
    dfQ = dfSf.applymap(lambda x: np.imag(x)).round(1)

    dfP.columns = ['Re('+x+')' for x in dfSf.columns.tolist()]
    dfQ.columns = ['Im('+x+')' for x in dfSf.columns.tolist()]

    row = 0
    dfx1 = pd.concat([dfP, dfQ], axis=1, join_axes=[dfP.index])
    dfx1.to_excel(writer, 'Results', startrow=row)
    row += dfx1.index.size + 2

    # Preparing voltages
    dfUf = dfU.loc[dfU.index.intersection(caselist)]
    if SortCases: dfUf.sort_index(inplace=True)
    dfUmagpu = dfUf.applymap(lambda x: np.abs(x)).round(4)
    dfUang = dfUf.applymap(lambda x: np.angle(x, deg=True)).round(2)
    dfUmag = dfUf.applymap(lambda x: np.abs(x)*Ub*np.sqrt(2.)).round(1)

    dfUmagpu.columns = ['|'+x+'pu|' for x in dfUf.columns.tolist()]
    dfUang.columns = ['ang('+x+')' for x in dfUf.columns.tolist()]
    dfUmag.columns = ['|'+x+'|' for x in dfUf.columns.tolist()]

    dfx2 = pd.concat([dfUmagpu, dfUang, dfUmag], axis=1, join_axes=[dfUmagpu.index])
    dfx2.to_excel(writer, 'Results', startrow=row)
    row += dfx2.index.size + 2

    # Preparing UPFC sizing info
    UPFCcases = list(set(caselist) & set(dfUPFC.index.tolist())) # intersect cases from dfUPFC index with specified cases
    dfUPFCf = dfUPFC.loc[dfUPFC.index.intersection(UPFCcases)]
    if SortCases: dfUPFCf.sort_index(inplace=True)
    df1 = dfUPFCf.loc[:, ['Ssh', 'Sser']]
    df1re = df1.applymap(lambda x: np.real(x)).round(2)
    df1im = df1.applymap(lambda x: np.imag(x)).round(1)

    df1re.columns = ['Re('+x+')' for x in df1.columns.tolist()]
    df1im.columns = ['Im('+x+')' for x in df1.columns.tolist()]

    df2 = dfUPFCf.loc[:, ['Ush', 'User']]
    df2magpu = df2.applymap(lambda x: np.abs(x)).round(4)
    df2ang = df2.applymap(lambda x: np.angle(x, deg=True)).round(2)
    df2mag = df2.applymap(lambda x: np.abs(x)*Ub*np.sqrt(2.)).round(1)

    df2magpu.columns = ['|'+x+'pu|' for x in df2.columns.tolist()]
    df2ang.columns = ['ang('+x+')' for x in df2.columns.tolist()]
    df2mag.columns = ['|'+x+'|' for x in df2.columns.tolist()]

    dfx3 = pd.concat([df1re, df1im, df2magpu, df2ang, df2mag], axis=1, join_axes=[df1re.index])
    dfx3.to_excel(writer,'Results', startrow=row)
    row += dfx3.index.size + 2

    # Preparing HPFC sizing info
    HPFCcases = list(set(caselist) & set(dfHPFC.index.tolist())) # intersect cases from dfHPFC index with specified cases
    dfHPFCf = dfHPFC.loc[dfHPFC.index.intersection(HPFCcases)]
    if SortCases: dfHPFCf.sort_index(inplace=True)
    df0 = dfHPFCf.loc[:, ['QM']].applymap(lambda x: x).round(1)
    df1 = dfHPFCf.loc[:, ['SX', 'SY']]
    df1re = df1.applymap(lambda x: np.real(x)).round(2)
    df1im = df1.applymap(lambda x: np.imag(x)).round(1)

    df1re.columns = ['Re('+x+')' for x in df1.columns.tolist()]
    df1im.columns = ['Im('+x+')' for x in df1.columns.tolist()]

    df2 = dfHPFCf.loc[:, ['UM', 'UX', 'UY']]
    df2magpu = df2.applymap(lambda x: np.abs(x)).round(4)
    df2ang = df2.applymap(lambda x: np.angle(x, deg=True)).round(2)
    df2mag = df2.applymap(lambda x: np.abs(x)*Ub*np.sqrt(2.)).round(1)

    df2magpu.columns = ['|'+x+'pu|' for x in df2.columns.tolist()]
    df2ang.columns = ['ang('+x+')' for x in df2.columns.tolist()]
    df2mag.columns = ['|'+x+'|' for x in df2.columns.tolist()]

    dfx4 = pd.concat([df0, df1re, df1im, df2magpu, df2ang, df2mag], axis=1, join_axes=[df0.index])
    dfx4.to_excel(writer,'Results', startrow=row)
    row += dfx4.index.size + 2

    # writing parameters
    worksheet2 = workbook.add_worksheet('Parameters')
    worksheet2.write('A1', fs); worksheet2.write('B1', 'fs [Hz]')
    worksheet2.write_formula('A2', '=2.*PI()*A1'); worksheet2.write('B2', 'ws [rad/s]')
    
    worksheet2.write('A3', np.real(Zs)); worksheet2.write('B3', 'Rs [ohm]')
    worksheet2.write('A4', np.imag(Zs)); worksheet2.write('B4', 'Xs [ohm]')
    worksheet2.write_formula('A5', '=A4/A$2'); worksheet2.write('B5', 'Ls [H]')

    worksheet2.write('A6', np.imag(Zm)); worksheet2.write('B6', 'Xm [ohm]')
    worksheet2.write_formula('A7', '=A6/A$2'); worksheet2.write('B7', 'Lm [H]')
    worksheet2.write('A8', np.imag(ZL)); worksheet2.write('B8', 'XL [ohm]')
    worksheet2.write_formula('A9', '=A8/A$2'); worksheet2.write('B9', 'LL [H]')

    worksheet2.write('A10', np.real(Zr)); worksheet2.write('B10', 'Rr [ohm]')
    worksheet2.write('A11', np.imag(Zr)); worksheet2.write('B11', 'Xr [ohm]')
    worksheet2.write_formula('A12', '=A11/A$2'); worksheet2.write('B12', 'Lr [H]')
    
    # Save excel file to disk
    writer.save()

    return

#%% Overcoming incomplete definitions of power flows
def SolveSs(Uspu, S0, UbLL=220.):

    nw = pypsa.Network() # holds network data

    nw.add("Bus","Us", v_nom=UbLL, v_mag_pu_set=Uspu) # PV bus, voltage set here, P set in the corresponding generator
    nw.add("Bus","U1", v_nom=UbLL)

    nw.add("Line", "Zs", bus0="Us", bus1="U1", x=np.imag(Zs), r=np.real(Zs))
    nw.add("Generator", "U1", bus="U1", control="PQ", p_set=np.real(S0), q_set=np.imag(S0))
    nw.add("Generator", "Us", bus="Us", control="Slack")    
    
    nw.pf() # Solve load flow
    return np.complex(nw.lines_t.p0.Zs['now'], nw.lines_t.q0.Zs['now'])

def SolveSr(Urpu, S4, UbLL=220.):

    nw = pypsa.Network() # holds network data

    nw.add("Bus","U3", v_nom=UbLL) 
    nw.add("Bus","Ur", v_nom=UbLL, v_mag_pu_set=Urpu) # PV bus, voltage set here, P set in the corresponding generator

    nw.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))
    nw.add("Generator", "U3", bus="U3", control="PQ", p_set=np.real(S4), q_set=np.imag(S4))
    nw.add("Generator", "Ur", bus="Ur", control="Slack")

    nw.pf() # Solve load flow
    
    return np.complex(-nw.lines_t.p1.Zr['now'], -nw.lines_t.q1.Zr['now'])

#%% Solve baseline
def SolveBaselineFlows(UbLL=220.):
    # all the parameters are defined via Config01 module
    Z12 = 1./(1./Zm + 1./ZL + 1./ZL)
    Is = (Us-Ur)/(Zs + Z12 + Zr)
    Ir = Is
    
    U1 = Us - Zs*Is # voltage at Bus1
    U2 = U1 # No FACTS compensator
    U3 = Ur + Zr*Ir
    
    IL = (U2-U3)/ZL # Current flows
    Im = (U1-U3)/Zm
    
    Ss = 3.*Us*np.conj(Is) # Apparent powers
    S2 = 3.*U2*np.conj(2.*IL)
    S3 = 3.*U3*np.conj(2.*IL)
    Sr = 3.*Ur*np.conj(Ir)
    Sm = 3.*U1*np.conj(Im)
    Smm = 3.*U3*np.conj(Im)
    S1 = -S2
    S0 = S1-Sm
    S4 = S3+Smm
    
    Ub = UbLL/np.sqrt(3.)
    return [pd.Series({'Us':Us/Ub, 'U1':U1/Ub, 'U2':U2/Ub, 'U3':U3/Ub, 'Ur':Ur/Ub}),
            pd.Series({'Ss':Ss, 'S0':S0, 'S1':S1, 'Sm':Sm, "Sm'":Smm, 'S2':S2, 'S3':S3, 'S4':S4, 'Sr':Sr})]
    
#%% Set system dispatch compatible to HPFC compensation
def SetSystemDispatch4HPFC(Us, Ur, Ss, Sr, alpha=1.0):
    # Us, Ur: Voltage vectors at the sending and receiving end [pu]
    # Ss, Sr: Apparent powers sending and receiving ends of the system [MVA]
    # alpha: Multiplier for QM; when left at 1, QM is equal to the Imag(S2+S1)
    Is = np.conj(Ss/(3.*Us*Ub))*1000.
    U1 = Us - Zs*Is/(Ub*1000.)
    Ir = np.conj(Sr/(3.*Ur*Ub))*1000.
    U3 = Ur + Zr*Ir/(Ub*1000.)
    Im = Ub*(U1 - U3)/Zm*1000.
    I1 = Is - Im
    I2 = Ir - Im
    U2 = U3 + (ZL/2.*I2)/(Ub*1000.)
    S1 = 3.*Ub*U1*np.conj(-I1)/1000.
    S2 = 3.*Ub*U2*np.conj(I2)/1000.

    # Adjusting terminal voltages currents of the compensator
    U12avg = (U1 + U2)/2.
    I12avg = (I1 + I2)/2.
    IM = I1 - I2 # shunt current at the compensator 
    IMr = np.abs(IM)*np.exp(1.j*(np.pi/2+np.angle(U12avg))) # rotated shunt current

    aI2 = I12avg - alpha*IMr/2.
    aI1 = I12avg + alpha*IMr/2.
    aU1 = S1/(3.*Ub*np.conj(-aI1)/1000.)
    aU2 = S2/(3.*Ub*np.conj(aI2)/1000.)

    # Recalculating adjusted voltages and currents in the system
    aU3 = aU2 - (ZL/2.*aI2)/(Ub*1000.)
    aIm = Ub*(aU1 - aU3)/Zm*1000.
    aIs = aI1 + aIm
    aI3 = aI2
    aIr = aI3 + aIm
    aUr = aU3 - Zr*aIr/(Ub*1000.)
    aUs = aU1 + Zs*aIs/(Ub*1000.)

    # Recalculating adjusted apparent powers
    aSs = 3.*Ub*aUs*np.conj(aIs)/1000.
    aS1 = 3.*Ub*aU1*np.conj(-aI1)/1000.
    aS2 = 3.*Ub*aU2*np.conj(aI2)/1000.
    aS3 = 3.*Ub*aU3*np.conj(aI2)/1000.
    aSm = 3.*Ub*aU1*np.conj(aIm)/1000.
    aSmm = 3.*Ub*aU3*np.conj(aIm)/1000.
    aS0 = 3.*Ub*aU1*np.conj(-aIs)/1000.
    aS4 = 3.*Ub*aU3*np.conj(aIr)/1000.
    aSr = 3.*Ub*aUr*np.conj(aIr)/1000.
    return [pd.Series({'Us':aUs, 'U1':aU1, 'U2':aU2, 'U3':aU3, 'Ur':aUr}),
            pd.Series({'Ss':aSs, 'S0':aS0, 'S1':aS1, 'Sm':aSm, "Sm'":aSmm, 'S2':aS2, 'S3':aS3, 'S4':aS4, 'Sr':aSr})]
    
#%% Adjust compensator operating points to make it compatible with HPFC technology
def AdjustCompensation4HPFC(U1, U2, S1, S2, alpha=1.0):
    # U1, U2: Voltage vectors at compensator terminals [pu]
    # S1, S2: Apparent powers at compensator terminals [MVA]
    # alpha: Multiplier for QM; when left at 1, QM is equal to the Imag(S2+S1)
    I1 = np.conj(-S1/(3.*U1*Ub))*1000. # Currents are in Amperes
    I2 = np.conj(S2/(3.*U2*Ub))*1000.

    # Adjusting terminal voltages currents of the compensator
    U12avg = (U1 + U2)/2.
    I12avg = (I1 + I2)/2.
    IM = I1 - I2 # shunt current at the compensator 
    IMr = np.abs(IM)*np.exp(1.j*(np.pi/2+np.angle(U12avg))) # rotated shunt current

    # Adjusting terminal variables
    aI2 = I12avg - alpha*IMr/2.
    aI1 = I12avg + alpha*IMr/2.
    aU1 = S1/(3.*Ub*np.conj(-aI1)/1000.)
    aU2 = S2/(3.*Ub*np.conj(aI2)/1000.)

    # Recalculating adjusted voltages and currents in the system
    aU3 = aU2 - (ZL/2.*aI2)/(Ub*1000.)
    aIm = Ub*(aU1 - aU3)/Zm*1000.
    aIs = aI1 + aIm
    aI3 = aI2
    aIr = aI3 + aIm
    aUr = aU3 - Zr*aIr/(Ub*1000.)
    aUs = aU1 + Zs*aIs/(Ub*1000.)

    # Recalculating adjusted apparent powers
    aSs = 3.*Ub*aUs*np.conj(aIs)/1000.
    aS1 = 3.*Ub*aU1*np.conj(-aI1)/1000.
    aS2 = 3.*Ub*aU2*np.conj(aI2)/1000.
    aS3 = 3.*Ub*aU3*np.conj(aI2)/1000.
    aSm = 3.*Ub*aU1*np.conj(aIm)/1000.
    aSmm = 3.*Ub*aU3*np.conj(aIm)/1000.
    aS0 = 3.*Ub*aU1*np.conj(-aIs)/1000.
    aS4 = 3.*Ub*aU3*np.conj(aIr)/1000.
    aSr = 3.*Ub*aUr*np.conj(aIr)/1000.
    return [pd.Series({'Us':aUs, 'U1':aU1, 'U2':aU2, 'U3':aU3, 'Ur':aUr}),
            pd.Series({'Ss':aSs, 'S0':aS0, 'S1':aS1, 'Sm':aSm, "Sm'":aSmm, 'S2':aS2, 'S3':aS3, 'S4':aS4, 'Sr':aSr})]
    
#%% Create a baseline network: Bus1 is the same as Bus2
def CreateBaselineNetwork(Uspu, Urpu, snapshots=['now'], UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data
    nw.set_snapshots(snapshots) # provision to add snapshots

    #add the buses
    nw.add("Bus","Us", v_nom=UbLL, v_mag_pu_set=Uspu) # PV bus, voltage set here, P set in the corresponding generator
    nw.add("Bus","U2", v_nom=UbLL)
    nw.add("Bus","U3", v_nom=UbLL)
    nw.add("Bus","Ur", v_nom=UbLL, v_mag_pu_set=Urpu) # PV bus
    
    # add the lines
    nw.add("Line", "Zs", bus0="Us", bus1="U2", x=np.imag(Zs), r=np.real(Zs))
    nw.add("Line", "Zm", bus0="U2", bus1="U3", x=np.imag(Zm), r=np.real(Zm))
    nw.add("Line", "ZL1", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "ZL2", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))
    
    return nw

#%% Creates a two-part network
def CreateNetwork4TargetFlows(Uspu, Urpu, snapshots=['now'], UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data
    nw.set_snapshots(snapshots) # provision to add snapshots

    #add the buses
    nw.add("Bus","Us", v_nom=UbLL, v_mag_pu_set=Uspu) # PV bus, voltage set here, P set in the corresponding generator
    nw.add("Bus","U1", v_nom=UbLL)
    nw.add("Bus","U2", v_nom=UbLL)
    nw.add("Bus","U3", v_nom=UbLL)
    nw.add("Bus","Ur", v_nom=UbLL, v_mag_pu_set=Urpu) # PV bus
    
    # add the lines
    nw.add("Line", "Zs", bus0="Us", bus1="U1", x=np.imag(Zs), r=np.real(Zs))
    nw.add("Line", "ZL1", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "ZL2", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))
    
    return nw

#%% Creates a baseline network with a provision to add an SVC at Bus2 (Bus1 = Bus2)
def CreateNetwork4SVCComp(Uspu, Urpu, snapshots=['now'], U2pu=1., UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data
    nw.set_snapshots(snapshots) # provision to add snapshots

    #add the buses
    nw.add("Bus","Us", v_nom=UbLL, v_mag_pu_set=Uspu) # PV bus, voltage set here, P set in the corresponding generator
    nw.add("Bus","U2", v_nom=UbLL, v_mag_pu_set=U2pu)
    nw.add("Bus","U3", v_nom=UbLL)
    nw.add("Bus","Ur", v_nom=UbLL, v_mag_pu_set=Urpu) # PV bus
    
    # add the lines
    nw.add("Line", "Zs", bus0="Us", bus1="U2", x=np.imag(Zs), r=np.real(Zs))
    nw.add("Line", "Zm", bus0="U2", bus1="U3", x=np.imag(Zm), r=np.real(Zm))
    nw.add("Line", "ZL1", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "ZL2", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))
    
    return nw

#%% Creates a network with separate buses 1 and 2 that are coupled via PQ specs at the terminals
def CreateNetwork4PQQComp(Uspu, Urpu, snapshots=['now'], UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data
    nw.set_snapshots(snapshots) # provision to add snapshots

    #add the buses
    nw.add("Bus","Us", v_nom=UbLL, v_mag_pu_set=Uspu) # PV bus, voltage set here, P set in the corresponding generator
    nw.add("Bus","U1", v_nom=UbLL)
    nw.add("Bus","U2", v_nom=UbLL)
    nw.add("Bus","U3", v_nom=UbLL)
    nw.add("Bus","Ur", v_nom=UbLL, v_mag_pu_set=Urpu) # PV bus
    
    # add the lines
    nw.add("Line", "Zs", bus0="Us", bus1="U1", x=np.imag(Zs), r=np.real(Zs))
    nw.add("Line", "Zm", bus0="U1", bus1="U3", x=np.imag(Zm), r=np.real(Zm))
    nw.add("Line", "ZL1", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "ZL2", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))
    
    return nw

#%% Creates a network with separate buses 1 and 2 that are coupled via PVV specs at the terminals
def CreateNetwork4PVVComp(Uspu, Urpu, snapshots=['now'], U1pu=1., U2pu=1., UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data
    nw.set_snapshots(snapshots) # provision to add snapshots

    #add the buses
    nw.add("Bus","Us", v_nom=UbLL, v_mag_pu_set=Uspu) # PV bus, voltage set here, P set in the corresponding generator
    nw.add("Bus","U1", v_nom=UbLL, v_mag_pu_set=U1pu)
    nw.add("Bus","U2", v_nom=UbLL, v_mag_pu_set=U2pu)
    nw.add("Bus","U3", v_nom=UbLL)
    nw.add("Bus","Ur", v_nom=UbLL, v_mag_pu_set=Urpu) # PV bus
    
    # add the lines
    nw.add("Line", "Zs", bus0="Us", bus1="U1", x=np.imag(Zs), r=np.real(Zs))
    nw.add("Line", "Zm", bus0="U1", bus1="U3", x=np.imag(Zm), r=np.real(Zm))
    nw.add("Line", "ZL1", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "ZL2", bus0="U2", bus1="U3", x=np.imag(ZL), r=np.real(ZL))
    nw.add("Line", "Zr", bus0="U3", bus1="Ur", x=np.imag(Zr), r=np.real(Zr))
    
    return nw

#%% Add Generators
def AddGeneratorsUsUr(nw, P):
    if nw.snapshots.size > 1:
        Ps = pd.Series(P, index = nw.snapshots)
    else:
        Ps=P
    # sets up system generators
    nw.add("Generator", "Us", bus="Us", control="PV", p_set=Ps)
    nw.add("Generator", "Ur", bus="Ur", control="Slack")
    return nw

def AddGenerators4TargetFlows(nw, P, Q1, Q2, Sm1, Sm3):
    #%% set up equivlent generators for the compensator
    nw.add("Generator", "Us", bus="Us", control="Slack")
    nw.add("Generator", "U1", bus="U1", control="PQ", p_set=-P, q_set=Q1)
    nw.add("Generator", "U2", bus="U2", control="PQ", p_set= P, q_set=Q2)
    nw.add("Generator", "Ur", bus="Ur", control="Slack")
    nw.add("Generator", "Um1", bus="U1", control="PQ", p_set=-np.real(Sm1), q_set=-np.imag(Sm1))
    nw.add("Generator", "Um3", bus="U3", control="PQ", p_set= np.real(Sm3), q_set= np.imag(Sm3))
    return nw

def AddGenerators4PQQComp(nw, P, Q1, Q2):
    if nw.snapshots.size > 1:
        Ps1 = pd.Series(P, index = nw.snapshots)
        Q1s = pd.Series(Q1, index = nw.snapshots)
        Q2s = pd.Series(Q2, index = nw.snapshots)
    else:
        Ps1=P
        Q1s=Q1
        Q2s=Q2
    #%% set up equivlent generators for the compensator
    nw.add("Generator", "U1", bus="U1", control="PQ", p_set=Ps1*(-1.), q_set=Q1s)
    nw.add("Generator", "U2", bus="U2", control="PQ", p_set=Ps1, q_set=Q2s)
    return nw

def AddGenerators4PVVComp(nw, P):
    #%% set up equivlent generators for the compensator
    nw.add("Generator", "U1", bus="U1", control="PV", p_set=-P)
    nw.add("Generator", "U2", bus="U2", control="PV", p_set=P)
    return nw

def AddGenerators4SVCComp(nw):
    #%% set up equivlent generators for the compensator
    nw.add("Generator", "U2", bus="U2", control="PV", p_set=0.)
    return nw

#%% Store load flow solutions into dataframes
def StoreSolutions(nw, caseIx, snapshot='now'):
    # Store solved voltages all at once
    dfU.at[caseIx]=nw.buses_t.v_mag_pu.loc[snapshot] * np.exp(nw.buses_t.v_ang.loc[snapshot]*1.j)
    # Store solved aparent powers, pull them out of active and reactive flows of circuit elements
    dfS.at[caseIx, 'Ss'] = np.complex(nw.lines_t.p0.Zs[snapshot], nw.lines_t.q0.Zs[snapshot])
    dfS.at[caseIx, 'S0'] = np.complex(nw.lines_t.p1.Zs[snapshot], nw.lines_t.q1.Zs[snapshot])
    dfS.at[caseIx, 'Sm'] = np.complex(nw.lines_t.p0.Zm[snapshot], nw.lines_t.q0.Zm[snapshot])
    dfS.at[caseIx, "Sm'"] = np.complex(-nw.lines_t.p1.Zm[snapshot], -nw.lines_t.q1.Zm[snapshot])
    dfS.at[caseIx, 'S2'] = np.complex(nw.lines_t.p0.ZL1[snapshot] + nw.lines_t.p0.ZL2[snapshot], nw.lines_t.q0.ZL1[snapshot] + nw.lines_t.q0.ZL2[snapshot])
    dfS.at[caseIx, 'S3'] = np.complex(-nw.lines_t.p1.ZL1[snapshot] - nw.lines_t.p1.ZL2[snapshot], -nw.lines_t.q1.ZL1[snapshot] - nw.lines_t.q1.ZL2[snapshot])
    dfS.at[caseIx, 'S4'] = np.complex(nw.lines_t.p0.Zr[snapshot], nw.lines_t.q0.Zr[snapshot])
    dfS.at[caseIx, 'Sr'] = np.complex(-nw.lines_t.p1.Zr[snapshot], -nw.lines_t.q1.Zr[snapshot])
    dfS.at[caseIx, 'S1'] = dfS.S0[caseIx]+dfS.Sm[caseIx]
    return

def StoreTargetFlows(nw, caseIx, snapshot='now'):
    # Store solved voltages all at once
    dfU.at[caseIx]=nw.buses_t.v_mag_pu.loc[snapshot] * np.exp(nw.buses_t.v_ang.loc[snapshot]*1.j)
    # Store solved aparent powers, pull them out of active and reactive flows of circuit elements
    dfS.at[caseIx, 'Ss'] = np.complex(nw.lines_t.p0.Zs[snapshot], nw.lines_t.q0.Zs[snapshot])
    dfS.at[caseIx, 'S0'] = np.complex(nw.lines_t.p1.Zs[snapshot], nw.lines_t.q1.Zs[snapshot])
    dfS.at[caseIx, 'Sm'] = np.complex(-nw.generators_t.p.Um1[snapshot], -nw.generators_t.q.Um1[snapshot])
    dfS.at[caseIx, "Sm'"] = np.complex(nw.generators_t.p.Um3[snapshot], nw.generators_t.q.Um3[snapshot])
    dfS.at[caseIx, 'S2'] = np.complex(nw.lines_t.p0.ZL1[snapshot] + nw.lines_t.p0.ZL2[snapshot], nw.lines_t.q0.ZL1[snapshot] + nw.lines_t.q0.ZL2[snapshot])
    dfS.at[caseIx, 'S3'] = np.complex(-nw.lines_t.p1.ZL1[snapshot] - nw.lines_t.p1.ZL2[snapshot], -nw.lines_t.q1.ZL1[snapshot] - nw.lines_t.q1.ZL2[snapshot])
    dfS.at[caseIx, 'S4'] = np.complex(nw.lines_t.p0.Zr[snapshot], nw.lines_t.q0.Zr[snapshot])
    dfS.at[caseIx, 'Sr'] = np.complex(-nw.lines_t.p1.Zr[snapshot], -nw.lines_t.q1.Zr[snapshot])
    dfS.at[caseIx, 'S1'] = dfS.S0[caseIx]+dfS.Sm[caseIx]
    return

#%% Calculate UPFC operating point
def CalculateUPFCop(caseIx):
    S1 = dfS.S1[caseIx]
    U1 = dfU.U1[caseIx]*Ub
    S2 = dfS.S2[caseIx]
    U2 = dfU.U2[caseIx]*Ub
    I1 = np.conj(-S1/(3.*U1))*1000. # currents in amperes
    I2 = np.conj(S2/(3.*U2))*1000.

    Ish = I1 - I2
    dfUPFC.at[caseIx, 'Ssh'] = 3.*U1*np.conj(-Ish)/1000.
    dfUPFC.at[caseIx, 'Sser'] = 3.*(U2-U1)*np.conj(I2)/1000.
    dfUPFC.at[caseIx, 'User'] = (U2-U1)/Ub
    dfUPFC.at[caseIx, 'Ush'] = U1/Ub

    return

#%% Calculate HPFC operating point
def CalculateHPFCop(caseIx):
    S1 = dfS.S1[caseIx] # apparent powers in MW
    S2 = dfS.S2[caseIx]
    U1 = dfU.U1[caseIx]*Ub # voltages in kV
    U2 = dfU.U2[caseIx]*Ub
    I1 = np.conj(-S1/(3.*U1))*1000. # currents in amperes
    I2 = np.conj(S2/(3.*U2))*1000. 
    IM = I1-I2
    BM = np.abs(IM)/((np.abs(U1)+np.abs(U2))/2.) # Selecting sufficient BM to hit (|U1|+|U2|)/2 with available IM_hpfc
    
    UM = IM/(1.j*BM)
    dfHPFC.at[caseIx, 'UM'] = UM/Ub
    dfHPFC.at[caseIx, 'QM'] = np.imag(-3.*UM*np.conj(IM)/1000.)

    UX = U1-UM
    UY = U2-UM

    dfHPFC.at[caseIx, 'SX'] = 3.*UX*np.conj(I1)/1000.
    dfHPFC.at[caseIx, 'SY'] = 3.*UY*np.conj(I2)/1000.
    dfHPFC.at[caseIx, 'UX'] = UX/Ub
    dfHPFC.at[caseIx, 'UY'] = UY/Ub

    return

#%% Log HPFC operating point (save all relevant info to the log file as formatted text)
def LogHPFCop(foutLog, caseIx):
    foutLog.write('\n%s: %s\n' %(caseIx, dfS.Note[caseIx]))
    QM = dfHPFC.QM[caseIx]
    SX = dfHPFC.SX[caseIx]
    SY = dfHPFC.SY[caseIx]
    
    UM = dfHPFC.UM[caseIx]
    UX = dfHPFC.UX[caseIx]
    UY = dfHPFC.UY[caseIx]
    
    S1 = dfS.S1[caseIx] # apparent powers in MW
    S2 = dfS.S2[caseIx]
    U1 = (UM+UX)*Ub
    U2 = (UM+UY)*Ub

    I1 = np.conj(-S1/(3.*U1)) # current in kA
    I2 = np.conj(S2/(3.*U2))
    IM = I1-I2

    # foutLog.write('HPFC Ratings -- shunt reactive branch variant\n')
    foutLog.write('  QM = %.2f\n' %(QM))
    
    foutLog.write('  |IM| = %.4f kA, ang(IM) = %.2f deg\n' %(np.abs(IM), np.angle(IM, deg=True)))
    foutLog.write('  |UM| = %.4f pu, ang(UM) = %.2f deg\n' %(np.abs(UM), np.angle(UM, deg=True)))
    
    foutLog.write('  |UX| = %.4f pu, ang(UX) = %.2f deg\n' %(np.abs(UX), np.angle(UX, deg=True)))
    foutLog.write('  |UY| = %.4f pu, ang(UY) = %.2f deg\n\n' %(np.abs(UY), np.angle(UY, deg=True)))
    
    foutLog.write('  PX_hpfc = %.2f\n' %(np.real(SX)))
    foutLog.write('  QX_hpfc = %.2f\n' %(np.imag(SX)))
    foutLog.write('  SX_hpfc = %.2f\n\n' %(np.abs(SX)))
    
    foutLog.write('  PY_hpfc = %.2f\n' %(np.real(SY)))
    foutLog.write('  QY_hpfc = %.2f\n' %(np.imag(SY)))
    foutLog.write('  SY_hpfc = %.2f\n\n' %(np.abs(SY)))
    
    foutLog.write('  QM-QX+QY = %.2f\n' %(QM-np.imag(SX)+np.imag(SY)))

    return

#%% Log UPFC operating point (save all relevant info to the log file as formatted text)
def LogUPFCop(foutLog, caseIx):
    foutLog.write('\n%s: %s\n' %(caseIx, dfS.Note[caseIx]))
    S1 = dfS.S1[caseIx] # apparent powers in MW
    S2 = dfS.S2[caseIx]
    Ush = dfUPFC.Ush[caseIx]*Ub
    User = dfUPFC.User[caseIx]*Ub

    I1 = np.conj(-S1/(3.*Ush)) # current in kA
    I2 = np.conj(S2/(3.*(Ush+User)))
    Ish = I1 - I2

    Ssh_upfc = 3.*Ush*np.conj(-Ish)
    Sser_upfc = 3.*(User)*np.conj(I2)
    
    # foutLog.write('\nUPFC Ratings\n')
    foutLog.write('  Psh = %.2f\n' %(np.real(Ssh_upfc)))
    foutLog.write('  Qsh = %.2f\n' %(np.imag(Ssh_upfc)))
    foutLog.write('  Ssh = %.2f\n\n' %(np.abs(Ssh_upfc)))
    foutLog.write('  Pser = %.2f\n' %(np.real(Sser_upfc)))
    foutLog.write('  Qser = %.2f\n' %(np.imag(Sser_upfc)))
    foutLog.write('  Sser = %.2f\n\n' %(np.abs(Sser_upfc)))
    foutLog.write('  Psh+Pser = %.2f\n' %(np.real(Ssh_upfc)+np.real(Sser_upfc)))
    foutLog.write('  Qsh+Qser = %.2f\n' %(np.imag(Ssh_upfc)+np.imag(Sser_upfc)))
    return

#%% Log system operating point (save all relevant info to the log file as formatted text)
def LogSystem(foutLog, caseIx):
    foutLog.write('\n%s: %s\n' %(caseIx, dfS.Note[caseIx]))
    Us = dfU.Us[caseIx] # voltages in pu
    U1 = dfU.U1[caseIx]
    U2 = dfU.U2[caseIx]
    U3 = dfU.U3[caseIx]
    Ur = dfU.Ur[caseIx]

    Ss = dfS.Ss[caseIx] # apparent powers in MW
    S0 = dfS.S0[caseIx]
    S1 = dfS.S1[caseIx]
    S2 = dfS.S2[caseIx]
    S3 = dfS.S3[caseIx]
    Sm = dfS.Sm[caseIx]
    # Smm = dfS["Sm'"][caseIx]
    S4 = dfS.S4[caseIx]
    Sr = dfS.Sr[caseIx]

    Is = np.conj(Ss/(3.*Us*Ub)) # current in kA
    I1 = np.conj(-S1/(3.*U1*Ub))
    Im = np.conj(Sm/(3.*U1*Ub))
    I2 = np.conj(S2/(3.*U2*Ub))
    # I3 = np.conj(S3/(3.*(U3)))
    IL = I2/2.
    Ir = np.conj(Sr/(3.*U1*Ub))

    foutLog.write('Terminal voltages:\n')
    foutLog.write('  |Us| = %.4f pu, ang(Us) = %.2f deg\n' %(np.abs(Us), np.angle(Us, deg=True)))
    foutLog.write('  |U1| = %.4f pu, ang(U1) = %.2f deg\n' %(np.abs(U1), np.angle(U1, deg=True)))
    foutLog.write('  |U2| = %.4f pu, ang(U2) = %.2f deg\n' %(np.abs(U2), np.angle(U2, deg=True)))
    foutLog.write('  |U3| = %.4f pu, ang(U3) = %.2f deg\n' %(np.abs(U3), np.angle(U3, deg=True)))
    foutLog.write('  |Ur| = %.4f pu, ang(Ur) = %.2f deg\n' %(np.abs(Ur), np.angle(Ur, deg=True)))
    
    foutLog.write('Current values\n')
    foutLog.write('  |Is| = %.3f kA, ang(Is) = %.2f deg\n' %(np.abs(Is), np.angle(Is, deg=True)))
    foutLog.write('  |I1| = %.3f kA, ang(I1) = %.2f deg\n' %(np.abs(I1), np.angle(I1, deg=True)))
    foutLog.write('  |I2| = %.3f kA, ang(I2) = %.2f deg\n' %(np.abs(I2), np.angle(I2, deg=True)))
    foutLog.write('  |IL| = %.3f kA, ang(IL) = %.2f deg\n' %(np.abs(IL), np.angle(IL, deg=True)))
    foutLog.write('  |Im| = %.3f kA, ang(Im) = %.2f deg\n' %(np.abs(Im), np.angle(Im, deg=True)))
    foutLog.write('  |Ir| = %.3f kA, ang(Ir) = %.2f deg\n' %(np.abs(Ir), np.angle(Ir, deg=True)))
    
    foutLog.write('Power flows\n')
    foutLog.write('  Ps = %.1f MW, ' %(np.real(Ss))) # Formatted results
    foutLog.write('Qs = %.1f MVAr\n\n' %(np.imag(Ss)))
    
    foutLog.write('  P0 = %.1f MW, ' %(np.real(S0)))
    foutLog.write('Q0 = %.1f MVAr\n' %(np.imag(S0)))
    foutLog.write('  P1 = %.1f MW, ' %(np.real(S1)))
    foutLog.write('Q1 = %.1f MVAr\n' %(np.imag(S1)))
    foutLog.write('  P2 = %.1f MW, ' %(np.real(S2)))
    foutLog.write('Q2 = %.1f MVAr\n\n' %(np.imag(S2)))
    
    foutLog.write('  P3 = %.1f MW, ' %(np.real(S3)))
    foutLog.write('Q3 = %.1f MVAr\n\n' %(np.imag(S3)))
    
    foutLog.write('  Pm = %.1f MW, ' %(np.real(Sm)))
    foutLog.write('Qm = %.1f MVAr\n\n' %(np.imag(Sm)))
    
    foutLog.write('  P4 = %.1f MW, ' %(np.real(S4)))
    foutLog.write('Q4 = %.1f MVAr\n' %(np.imag(S4)))
    foutLog.write('  Pr = %.1f MW, ' %(np.real(Sr)))
    foutLog.write('Qr = %.1f MVAr\n\n' %(np.imag(Sr)))
    return

#%% Main script begins here
if __name__ == "__main__":
    #%% Code info and file names
    codeVersion = '1.51'
    codeCopyright = 'GNU General Public License v3.0'
    codeAuthors = 'Jovan Z. Bebic\n'
    codeName = 'SizeFACTS.py'
    dirout = 'Results/'
    fnameLog = 'SizeFACTS.log'
    fnamePlt = 'SizeFACTS.pdf'
    OutputPlots = True
    
    #%% Capture start time of code execution and open log file
    codeTstart = datetime.now()
    foutLog = open(os.path.join(dirout, fnameLog), 'w')
    
    #%% Output log file header information
    print('This is %s v%s' %(codeName, codeVersion))
    foutLog.write('This is %s v%s\n' %(codeName, codeVersion))
    foutLog.write('%s\n' %(codeCopyright))
    foutLog.write('%s\n' %(codeAuthors))
    foutLog.write('Run started on: %s\n' %(str(codeTstart)))

    #%% Store specified values into dataframes
    dfS.at['State01s'] = pd.Series(State01s)
    dfS.at['State01s', 'Note'] = 'Apparent powers before compensation, as specified'
    dfS.at['State02s'] = pd.Series(State02s)
    dfS.at['State02s', 'Note'] = 'Apparent powers setpoints for compensation by a UPFC, specification adjusted by JZB to achieve ang(Us)=18.8deg'
    dfS.at['State03s'] = pd.Series(State02s) # the same spec for HPFC as for the UPFC
    dfS.at['State03s', 'Note'] = 'Apparent powers setpoints for compensation by an HPFC, specifications adjusted by JZB to balance the HPFC converters ratings.'
    dfU.at['State01s'] = pd.Series({'Us': Us/Ub, 'Ur': Ur/Ub})
    dfU.at['State02s'] = pd.Series({'Us': Us/Ub, 'Ur': Ur/Ub})
    dfU.at['State03s'] = pd.Series({'Us': Us/Ub, 'Ur': Ur/Ub})
    
    #%% Solve for accurate State01 flows: 'State01a'
    print('Solving State01a')
    [dfU.at['State01a'], dfS.at['State01a']] = SolveBaselineFlows()
    dfS.at['State01a', 'Note'] = "Solved baseline circuit ('a' = accurate)"
    LogSystem(foutLog, 'State01a')
    
    #%% Define per unit voltages used to set up load flows
    Uspu = np.abs(dfU.Us.State01s)
    Urpu = np.abs(dfU.Ur.State01s)
    
    #%% Set up and solve baseline load flow 'State01'
    print('Solving State01')
    n4c = CreateNetwork4PQQComp(Uspu, Urpu)
    n4c = AddGeneratorsUsUr(n4c, np.real(dfS.Ss.State01a))
    n4c = AddGenerators4PQQComp(n4c, np.real(dfS.S2.State01a), np.imag(dfS.S1.State01a), np.imag(dfS.S2.State01a))
    n4c.pf()
    StoreSolutions(n4c, 'State01')
    dfS.at['State01', 'Note'] = 'Solved baseline circuit with apparent power set points from State01a'
    LogSystem(foutLog, 'State01')
    
    #%% Set up and solve the UPFC-compensated load flow: State02
    print('Solving State02')
    dfS.Ss.State02s = SolveSs(Uspu, dfS.S0.State02s)
    dfS.Sr.State02s = SolveSr(Urpu, dfS.S4.State02s)
    dfS.at['State02s', 'Ss'] = dfS.Ss.State02s - 3.431  # hand corrected to restore ang(Us) = 18.8deg
    nu = CreateNetwork4PQQComp(Uspu, Urpu)
    nu = AddGeneratorsUsUr(nu, np.real(dfS.Ss.State02s))
    nu = AddGenerators4PQQComp(nu, np.real(dfS.S2.State02s), np.imag(dfS.S1.State02s), np.imag(dfS.S2.State02s))
    nu.pf()
    StoreSolutions(nu, 'State02')
    CalculateUPFCop('State02')
    dfS.at['State02', 'Note'] = 'Solved circuit compensated by a UPFC'
    LogSystem(foutLog, 'State02')
    LogUPFCop(foutLog, 'State02')
    
    #%% Set up and solve the HPFC-compensated system: State03 blindly follows the UPFC operating point
    print('Solving State03')
    dfS.Ss.State03s = SolveSs(Uspu, dfS.S0.State03s)
    dfS.Sr.State03s = SolveSr(Urpu, dfS.S4.State03s)
    Q1ref = np.imag(dfS.S1.State01) + (np.imag(dfS.S2.State02) - np.imag(dfS.S2.State01))
    Q2ref = np.imag(dfS.S2.State03s)
    dfS.at['State03s', 'S1'] = np.complex(-np.real(dfS.S2.State03s), Q1ref) # reactive power hand adjusted to balance the converter ratings
    dfS.at['State03s', 'S2'] = np.complex( np.real(dfS.S2.State03s), Q2ref) # reactive power hand adjusted to balance the converter ratings
    dfS.at['State03s', 'Ss'] = dfS.Ss.State03s - 3.407  # hand corrected to restore ang(Us) = 18.8deg
    nh = CreateNetwork4PQQComp(Uspu, Urpu)
    nh = AddGeneratorsUsUr(nh, np.real(dfS.Ss.State03s))
    nh = AddGenerators4PQQComp(nh, np.real(dfS.S2.State03s), np.imag(dfS.S1.State03s), np.imag(dfS.S2.State03s))
    nh.pf()
    StoreSolutions(nh, 'State03')
    CalculateHPFCop('State03')
    dfS.at['State03', 'Note'] = 'Solved circuit compensated by an HPFC with Q1cmd and Q2cmd as were used in the UPFC case'
    LogSystem(foutLog, 'State03')
    LogHPFCop(foutLog, 'State03')
    
    #%% Set up and solve the HPFC-compensated system. State05 algorithmically adjusted system dispatch that balances HPFC converter ratings
    print('Solving State05')
    [dfU.at['State05s'], dfS.at['State05s']] = AdjustCompensation4HPFC(dfU.U1.State02, dfU.U2.State02, dfS.S1.State02, dfS.S2.State02)
    dfS.at['State05s', 'Note'] = 'Algorithmically adjusted compensator operating point for HPFC technology'
    nh5 = CreateNetwork4PQQComp(np.abs(dfU.Us.State05s), np.abs(dfU.Ur.State05s))
    nh5 = AddGeneratorsUsUr(nh5, np.real(dfS.Ss.State05s))
    nh5 = AddGenerators4PQQComp(nh5, np.real(dfS.S2.State05s), np.imag(dfS.S1.State05s), np.imag(dfS.S2.State05s))
    nh5.pf()
    StoreSolutions(nh5, 'State05')
    CalculateHPFCop('State05')
    dfS.at['State05', 'Note'] = 'Solved circuit compensated by an HPFC: Algorithmically adjusted Q1cmd and Q2cmd to balance HPFC converters ratings'
    LogSystem(foutLog, 'State05')
    LogHPFCop(foutLog, 'State05')
    
    #%% Increase flow by taking 200MW more from Us without any compensation
    print('Solving State11')
    n11 = CreateBaselineNetwork(Uspu, Urpu)
    n11 = AddGeneratorsUsUr(n11, np.real(dfS.Ss.State01)+200)
    n11.pf()
    StoreSolutions(n11, 'State11')
    dfU.at['State11', 'U1'] = dfU.U2.State11
    dfS.at['State11', 'Note'] = 'Solved baseline circuit with increased dispatch of Us units by 200MW'
    LogSystem(foutLog, 'State11')
    
    #%% Set up and solve the target flows achieving the desired S2 while preserving the flow through Sm
    print('Solving State12')
    dfS.at['State12s'] = dfS.loc['State02'] # Transfer State02 setpoints
    dfS.at['State12s', 'Sm'] = dfS.Sm.State01a # Set the flow through Zm branch to the pre-compensation value
    dfS.at['State12s', 'Sm'] = dfS.Sm.State01a # Set the flow through Zm branch to the pre-compensation value
    dfS.at['State12s', "Sm'"] = dfS["Sm'"].State01a # Set the flow through Zm branch to the pre-compensation value
    nh12 = CreateNetwork4TargetFlows(Uspu, Urpu)
    nh12 = AddGenerators4TargetFlows(nh12, np.real(dfS.S2.State12s), np.imag(dfS.S1.State12s), np.imag(dfS.S2.State12s), dfS['Sm'].State12s, dfS["Sm'"].State12s)
    nh12.pf()
    StoreTargetFlows(nh12, 'State12')
    dfS.at['State12', 'Note'] = 'Solved circuit with target flows, Us units dispatched to get 200MW more and maintain the baseline flow via Zm'
    LogSystem(foutLog, 'State12')
    
    #%% Set up and solve the SVC-compensated system to match the operating point from State11
    print('Solving State13')
    dfS.at['State13s'] = dfS.loc['State12']
    ns13 = CreateNetwork4SVCComp(Uspu, Urpu, U2pu=(np.abs(dfU.U1.State12) + np.abs(dfU.U2.State12))/2.)
    ns13 = AddGeneratorsUsUr(ns13, np.real(dfS.Ss.State13s))
    ns13 = AddGenerators4SVCComp(ns13)
    ns13.pf()
    StoreSolutions(ns13, 'State13')
    dfU.at['State13', 'U1'] = dfU.U2.State13 # The circuit for SVC compensation does not include bus U1 (because U1=U2), define it to enable solving HPFC operating point
    CalculateHPFCop('State13')
    dfS.at['State13', 'Note'] = 'Solved circuit compensated by an SVC to approximate the operating point of an optimized HPFC'
    LogSystem(foutLog, 'State13')
    LogHPFCop(foutLog, 'State13')
    
    #%% Set up and solve the UPFC-compensated system to match the operating point from State25 (optimized HPFC)
    print('Solving State22')
    dfS.at['State22s'] = dfS.loc['State12']
    nu22 = CreateNetwork4PQQComp(Uspu, Urpu)
    nu22 = AddGeneratorsUsUr(nu22, np.real(dfS.Ss.State22s))
    nu22 = AddGenerators4PQQComp(nu22, np.real(dfS.S2.State22s), np.imag(dfS.S1.State22s), np.imag(dfS.S2.State22s))
    nu22.pf()
    StoreSolutions(nu22, 'State22')
    CalculateUPFCop('State22')
    dfS.at['State22', 'Note'] = 'Solved circuit compensated by a UPFC to match the operating point of an optimized HPFC'
    LogSystem(foutLog, 'State22')
    LogUPFCop(foutLog, 'State22')
    
    #%% Set up and solve the HPFC-compensated system. State25 algorithmically adjusted system dispatch that balances HPFC converter ratings
    print('Solving State25')
    [dfU.at['State25s'], dfS.at['State25s']] = AdjustCompensation4HPFC(dfU.U1.State22, dfU.U2.State22, dfS.S1.State22, dfS.S2.State22)
    dfS.at['State25s', 'Note'] = 'Algorithmically adjusted compensator operating point for HPFC technology'
    nh25 = CreateNetwork4PQQComp(np.abs(dfU.Us.State25s), np.abs(dfU.Ur.State25s))
    nh25 = AddGeneratorsUsUr(nh25, np.real(dfS.Ss.State25s))
    nh25 = AddGenerators4PQQComp(nh25, np.real(dfS.S2.State25s), np.imag(dfS.S1.State25s), np.imag(dfS.S2.State25s))
    nh25.pf()
    StoreSolutions(nh25, 'State25')
    CalculateHPFCop('State25')
    dfS.at['State25', 'Note'] = 'Solved circuit compensated by an HPFC: Algorithmically adjusted Q1cmd and Q2cmd to balance HPFC converters ratings'
    LogSystem(foutLog, 'State25')
    LogHPFCop(foutLog, 'State25')
    
    #%% Use HPFC to increase the flow by 200MW, without additional reactive burden on Us and Ur
    if False:
        print('Solving State26')
        [dfU.at['State26s'], dfS.at['State26s']] = SetSystemDispatch4HPFC(dfU.Us.State25, dfU.Ur.State25, 
                                                                            np.real(dfS.Ss.State25) + 1.j*np.imag(dfS.Ss.State01), 
                                                                            np.real(dfS.Sr.State25) + 1.j*np.imag(dfS.Sr.State01))
        dfS.at['State26s', 'Note'] = 'HPFC increases the flow by 200MW, without placing additional reactive demand on Us and Ur'
        nh26 = CreateNetwork4PQQComp(np.abs(dfU.Us.State26s), np.abs(dfU.Ur.State26s))
        nh26 = AddGeneratorsUsUr(nh26, np.real(dfS.Ss.State25s))
        nh26 = AddGenerators4PQQComp(nh26, np.real(dfS.S2.State26s), np.imag(dfS.S1.State26s), np.imag(dfS.S2.State26s))
        nh26.pf()
        StoreSolutions(nh26, 'State26')
        CalculateHPFCop('State26')
        dfS.at['State26', 'Note'] = 'Solved circuit compensated by an HPFC: Flow increase by 200MW without additional reactive demand on Us and Ur'
        LogSystem(foutLog, 'State26')
        LogHPFCop(foutLog, 'State26')
    
    #%% Set up and solve the HPFC-compensated system. State25 algorithmically adjusted system dispatch that balances HPFC converter ratings
    if False: # demonstration of multiple snapshots to study sensitivity to alpha
        print('Solving State25')
        [dfU.at['State25as'], dfS.at['State25as']] = AdjustCompensation4HPFC(dfU.U1.State22, dfU.U2.State22, dfS.S1.State22, dfS.S2.State22)
        [dfU.at['State25bs'], dfS.at['State25bs']] = AdjustCompensation4HPFC(dfU.U1.State22, dfU.U2.State22, dfS.S1.State22, dfS.S2.State22, alpha=1.1)
        [dfU.at['State25cs'], dfS.at['State25cs']] = AdjustCompensation4HPFC(dfU.U1.State22, dfU.U2.State22, dfS.S1.State22, dfS.S2.State22, alpha=1.2)
        dfS.at['State25as', 'Note'] = 'Algorithmically adjusted compensator operating point for HPFC technology, alpha=1.0'
        dfS.at['State25bs', 'Note'] = 'Algorithmically adjusted compensator operating point for HPFC technology, alpha=1.1'
        dfS.at['State25cs', 'Note'] = 'Algorithmically adjusted compensator operating point for HPFC technology, alpha=1.2'
        nh25 = CreateNetwork4PQQComp([np.abs(dfU.Us.State25as), np.abs(dfU.Us.State25bs), np.abs(dfU.Us.State25cs)],
                                     [np.abs(dfU.Ur.State25as), np.abs(dfU.Ur.State25bs), np.abs(dfU.Ur.State25cs)],
                                     snapshots=['a', 'b', 'c'])
        nh25 = AddGeneratorsUsUr(nh25, [np.real(dfS.Ss.State25as), np.real(dfS.Ss.State25bs), np.real(dfS.Ss.State25bs)])
        nh25 = AddGenerators4PQQComp(nh25, [np.real(dfS.S2.State25as), np.real(dfS.S2.State25bs), np.real(dfS.S2.State25cs)],
                                           [np.imag(dfS.S1.State25as), np.imag(dfS.S1.State25bs), np.imag(dfS.S1.State25cs)],
                                           [np.imag(dfS.S2.State25as), np.imag(dfS.S2.State25bs), np.imag(dfS.S2.State25cs)])
        nh25.pf('a')
        StoreSolutions(nh25, 'State25a', snapshot='a')
        CalculateHPFCop('State25a')
        dfS.at['State25a', 'Note'] = 'HPFC compensated circuit, alpha=1.0'
        LogSystem(foutLog, 'State25a')
        LogHPFCop(foutLog, 'State25a')
        nh25.pf('b')
        StoreSolutions(nh25, 'State25b', snapshot='b')
        CalculateHPFCop('State25b')
        dfS.at['State25b', 'Note'] = 'HPFC compensated circuit, alpha=1.1'
        LogSystem(foutLog, 'State25b')
        LogHPFCop(foutLog, 'State25b')
        nh25.pf('c')
        StoreSolutions(nh25, 'State25c', snapshot='c')
        CalculateHPFCop('State25c')
        dfS.at['State25c', 'Note'] = 'HPFC compensated circuit, alpha=1.2'
        LogSystem(foutLog, 'State25c')
        LogHPFCop(foutLog, 'State25c')
    
    #%% Example case illustrating use of multiple snapshots to explore the system sensitivity to a setpoint
    if False:
        print('Solving State06')
        dfS.at['State06s'] = dfS.loc['State01a']
        dfS.at['State06s', 'S1'] = np.complex(np.nan, np.nan) # 
        dfS.at['State06s', 'S2'] = np.complex(np.real(dfS.S2.State01a)+200, np.nan) # reactive power hand adjusted to balance the converter ratings
        U2pu11=np.abs(dfU.U2.State01a)
        nh6 = CreateNetwork4SVCComp(Uspu, Urpu, snapshots=['a', 'b'], U2pu=[U2pu11+0.001, U2pu11+0.002])
        nh6 = AddGeneratorsUsUr(nh6, np.real(dfS.Ss.State06s)+200)
        nh6 = AddGenerators4SVCComp(nh6)
        nh6.pf('a')
        StoreSolutions(nh6, 'State06a', snapshot='a')
        dfU.at['State06a', 'U1'] = dfU.U2.State06a # The circuit for SVC compensation does not include bus U1 (because U1=U2), define it to enable solving HPFC operating point
        CalculateHPFCop('State06a')
        dfS.at['State06a', 'Note'] = 'Solved circuit compensated by SVC using PV commands, P=0 V2pu+0.1% of uncompensated system'
        LogSystem(foutLog, 'State06a')
        LogHPFCop(foutLog, 'State06a')
        nh6.pf('b')
        StoreSolutions(nh6, 'State06b', snapshot='b')
        dfU.at['State06b', 'U1'] = dfU.U2.State06b # The circuit for SVC compensation does not include bus U1 (because U1=U2), define it to enable solving HPFC operating point
        CalculateHPFCop('State06b')
        dfS.at['State06b', 'Note'] = 'Solved circuit compensated by SVC using PV commands, P=0 V2pu+0.2% of uncompensated system'
        LogSystem(foutLog, 'State06b')
        LogHPFCop(foutLog, 'State06b')
        
    #%% Save results to Excel
    print('Saving to Excel')
    # caselist = dfS.index.tolist() 
    caselist = ['State01', 'State01s', 'State02', 'State02s', 'State03', 'State05',
                'State11', 'State12', 'State13', 
                'State22', 'State25'] 
    SaveToExcel(caselist, dirout='Results/', SortCases=True)
    
    if OutputPlots:
        # Opening plot files
        foutLog.write('\nStarting to plot at: %s\n' %(str(datetime.now())))
        print('Opening plot files')     
        pltPdf1 = dpdf.PdfPages(os.path.join(dirout,fnamePlt))
    
        for case in ['State01', 'State02', 'State03', 'State05',
                     'State11', 'State13',
                     'State22', 'State25']:
            OutputVectorsPage(pltPdf1, case, 
                              pageTitle='Created by '+ codeName + ' v' + codeVersion + '\n' + 
                                          r'$\bf{' + case + '}$')
    
        # Closing plot files
        print("Closing plot files")
        pltPdf1.close()
    
    #%% time stamp and close log file
    codeTfinish = datetime.now()
    foutLog.write('\nRun finished at: %s\n' %(str(codeTfinish)))
    codeTdelta = codeTfinish - codeTstart
    foutLog.write('Run Lasted: %.3f seconds\n' %(codeTdelta.total_seconds()))
    foutLog.close()
