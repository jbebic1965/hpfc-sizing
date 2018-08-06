#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 18:53:53 2018

@author: Jovan Z. Bebic

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

#%% Code info and file names
codeVersion = '1.2'
codeCopyright = 'GNU General Public License v3.0'
codeAuthors = 'Jovan Z. Bebic\n'
codeName = 'SizeFACTS.py'
dirout = 'Results/'
fnameLog = 'SizeFACTS.log'
fnamePlt = 'SizeFACTS.pdf'
OutputPlots = True

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
    df0 = dfHPFCf.loc[:, 'QM'].round(1)
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

#%% Define circuit parameters to specified values
def DefineCircuitParameters():
    
    global fs, ws, ZL, Zm, Zs, Zr

    fs = 50 # system frequency China
    ws = 2*np.pi*fs # 
    
    XL = ws * 0.015836 # equivalent reactance of each single line, per spec
    ZL = 1.j * XL # Setting line impedance as purely inductive, per spec
    
    Xm = ws * 0.17807 # equivalent reactance of parallel connection
    Zm = 1.j * Xm 
    
    Zs = 0.423125 + 1.j * ws * 0.011464 # Thevenin's impedance of the sending end, per spec
    Zr = 2.5159  + 1.j * ws * 0.05059  # Thevenin's impedance of the receiving end, per spec

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
def SolveBaselineFlows(Us, Ur, UbLL=220.):
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
    
#%% Create a baseline network: Bus1 is the same as Bus2
def CreateNetwork(Uspu, Urpu, snapshots=['now'], UbLL=220.):

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
def AddGeneratorsUsUr(nw, Ps):
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
    #%% set up equivlent generators for the compensator
    nw.add("Generator", "U1", bus="U1", control="PQ", p_set=-P, q_set=Q1)
    nw.add("Generator", "U2", bus="U2", control="PQ", p_set=P, q_set=Q2)
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
    I1 = np.conj(-S1/(3.*U1))
    I2 = np.conj(S2/(3.*U2))

    Ish = I1 - I2
    dfUPFC.at[caseIx, 'Ssh'] = 3.*U1*np.conj(-Ish)
    dfUPFC.at[caseIx, 'Sser'] = 3.*(U2-U1)*np.conj(I2)
    dfUPFC.at[caseIx, 'User'] = (U2-U1)/Ub
    dfUPFC.at[caseIx, 'Ush'] = U1/Ub

    return

#%% Calculate HPFC operating point
def CalculateHPFCop(caseIx):
    S1 = dfS.S1[caseIx] # apparent powers in MW
    S2 = dfS.S2[caseIx]
    U1 = dfU.U1[caseIx]*Ub # voltages in kV
    U2 = dfU.U2[caseIx]*Ub
    I1 = np.conj(-S1/(3.*U1))*1000. # current in amperes
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

#%% Capture start time of code execution and open log file
codeTstart = datetime.now()
foutLog = open(os.path.join(dirout, fnameLog), 'w')

#%% Output log file header information
print('This is %s v%s' %(codeName, codeVersion))
foutLog.write('This is %s v%s\n' %(codeName, codeVersion))
foutLog.write('%s\n' %(codeCopyright))
foutLog.write('%s\n' %(codeAuthors))
foutLog.write('Run started on: %s\n\n' %(str(codeTstart)))

#%% Define datastructure to hold the results
dfS = pd.DataFrame(columns=['Ss', 'S0', 'S1', 'Sm', "Sm'", 'S2', 'S3', 'S4', 'Sr', 'Note'])
dfU = pd.DataFrame(columns=['Us', 'U1', 'U2', 'U3', 'Ur'])

dfUPFC = pd.DataFrame(columns=['Ssh', 'Sser', 'Ush', 'User'])
dfHPFC = pd.DataFrame(columns=['SM', 'SX', 'SY', 'UM', 'UX', 'UY'])

# Define all constants
DefineCircuitParameters()

# Define voltages, per spec
UbLL = 220.
Ub = UbLL/np.sqrt(3.) # L-N RMS [kV], backed out from specified flows
Us = 186.5/np.sqrt(2.)*np.exp(18.8*np.pi/180.*1.j)
Ur = 172.2/np.sqrt(2.)

#%% Solve for accurate State01 flows: 'State01a'
print('Solving State01a')
[dfU.at['State01a'], dfS.at['State01a']] = SolveBaselineFlows(Us, Ur)
dfS.at['State01a', 'Note'] = "Solved baseline circuit ('a' = accurate)"

#%% Specified apparent powers
State01s = {'S0': -733   - 161.3j, 
           'S1': -701.8 - 154.4j,
           'S2':  701.8 + 154.4j,
           'S3':  700.6 + 128.5j,
           'S4':  731.7 + 134.2j}

State02s = {'S0': -881.6 - 111.5j, 
           'S1': -903.4 - 107.1j,
           'S2':  900.0 + 200.0j,
           'S3':  899.8 + 158.4j,
           'S4':  878.0 + 162.3j}

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

#%% Increase flow by taking 200MW more from Us without any compensation
print('Solving State11')
n11 = CreateNetwork(Uspu, Urpu)
n11 = AddGeneratorsUsUr(n11, np.real(dfS.Ss.State01)+200)
n11.pf()
StoreSolutions(n11, 'State11')
dfU.at['State11', 'U1'] = dfU.U2.State11
dfS.at['State11', 'Note'] = 'Solved baseline circuit with increased dispatch of Us units by 200MW'

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

#%% Set up and solve the HPFC-compensated system: State04 adjusts the UPFC operating point to balance the converter ratings
print('Solving State04')
dQ = 30
dfS.at['State04s'] = dfS.loc['State03s']
dfS.at['State04s', 'S1'] = np.complex(-np.real(dfS.S2.State03s), Q1ref) # reactive power hand adjusted to balance the converter ratings
dfS.at['State04s', 'S2'] = np.complex( np.real(dfS.S2.State03s), Q2ref+dQ) # reactive power hand adjusted to balance the converter ratings
dfS.at['State04s', 'Ss'] = dfS.Ss.State04s + 2.227  # hand corrected to restore ang(Us) = 18.8deg
nh4 = CreateNetwork4PQQComp(Uspu, Urpu)
nh4 = AddGeneratorsUsUr(nh4, np.real(dfS.Ss.State04s))
nh4 = AddGenerators4PQQComp(nh4, np.real(dfS.S2.State04s), np.imag(dfS.S1.State04s), np.imag(dfS.S2.State04s))
nh4.pf()
StoreSolutions(nh4, 'State04')
CalculateHPFCop('State04')
dfS.at['State04', 'Note'] = 'Solved circuit compensated by an HPFC with adjusted Q1cmd and Q2cmd to balance the HPFC converters ratings'

#%% Set up and solve the target flows achieving the desired S2 while preserving flows through Sm
print('Solving State12')
dfS.at['State12s'] = dfS.loc['State02s'] # Transfer State02 setpoints
dfS.at['State12s', 'Sm'] = dfS.Sm.State01a # Set the flow through Zm branch to the pre-compensation value
dfS.at['State12s', "Sm'"] = dfS["Sm'"].State01a # Set the flow through Zm branch to the pre-compensation value
nh12 = CreateNetwork4TargetFlows(Uspu, Urpu)
nh12 = AddGenerators4TargetFlows(nh12, np.real(dfS.S2.State12s), np.imag(dfS.S1.State12s), np.imag(dfS.S2.State12s), dfS['Sm'].State12s, dfS["Sm'"].State12s)
nh12.pf()
StoreTargetFlows(nh12, 'State12')
dfS.at['State12', 'Note'] = 'Solved circuit with target flows, Us units dispatched to get 200MW more and maintain the baseline flow via Zm'

#%% Set up and solve the HPFC-compensated system: State23 is the first cut of the operating point based on the spec
print('Solving State23')
dfS.at['State23s'] = dfS.loc['State12']
nh23 = CreateNetwork4PQQComp(Uspu, Urpu)
nh23 = AddGeneratorsUsUr(nh23, np.real(dfS.Ss.State23s))
nh23 = AddGenerators4PQQComp(nh23, np.real(dfS.S2.State23s), np.imag(dfS.S1.State23s), np.imag(dfS.S2.State23s))
nh23.pf()
StoreSolutions(nh23, 'State23')
CalculateHPFCop('State23')
dfS.at['State23', 'Note'] = 'Solved circuit compensated by an HPFC with the first-cut values for Q1cmd, Q2cmd'

#%% Set up and solve the HPFC-compensated system: State24 adjusts the UPFC operating point to balance the converter ratings
print('Solving State24')
Q1ref = np.imag(dfS.S1.State01a) + (np.imag(dfS.S2.State02s) - np.imag(dfS.S2.State01a))
dQ1 = 20
dfS.at['State24s'] = dfS.loc['State12']
dfS.at['State24s', 'S1'] = np.complex(np.real(dfS.S1.State24s), Q1ref+dQ1) # reactive power hand-adjusted to balance the converter ratings
nh24 = CreateNetwork4PQQComp(Uspu, Urpu)
nh24 = AddGeneratorsUsUr(nh24, np.real(dfS.Ss.State24s))
nh24 = AddGenerators4PQQComp(nh24, np.real(dfS.S2.State24s), np.imag(dfS.S1.State24s), np.imag(dfS.S2.State24s))
nh24.pf()
StoreSolutions(nh24, 'State24')
CalculateHPFCop('State24')
dfS.at['State24', 'Note'] = 'Solved circuit compensated by an HPFC with adjusted Q1cmd to balance the ratings of HPFC converters'

#%% Set up and solve the UPFC-compensated system to match the operating point from State24 (optimized HPFC)
print('Solving State32')
Q1ref = np.imag(dfS.S1.State01a) + (np.imag(dfS.S2.State02s) - np.imag(dfS.S2.State01a))
dQ1 = 20
dfS.at['State32s'] = dfS.loc['State12']
dfS.at['State32s', 'S1'] = np.complex(np.real(dfS.S1.State32s), Q1ref+dQ1) # reactive power hand-adjusted to balance the converter ratings
nu32 = CreateNetwork4PQQComp(Uspu, Urpu)
nu32 = AddGeneratorsUsUr(nu32, np.real(dfS.Ss.State32s))
nu32 = AddGenerators4PQQComp(nu32, np.real(dfS.S2.State32s), np.imag(dfS.S1.State32s), np.imag(dfS.S2.State32s))
nu32.pf()
StoreSolutions(nu32, 'State32')
CalculateUPFCop('State32')
dfS.at['State32', 'Note'] = 'Solved circuit compensated by a UPFC to match the operating point of an optimized HPFC'

#%% Set up and solve the UPFC-compensated system to match the operating point from State24 (optimized HPFC)
print('Solving State25')
dfS.at['State25s'] = dfS.loc['State12']
ns25 = CreateNetwork4SVCComp(Uspu, Urpu, U2pu=(np.abs(dfU.U1.State24)+np.abs(dfU.U2.State24))/2.)
ns25 = AddGeneratorsUsUr(ns25, np.real(dfS.Ss.State25s))
ns25 = AddGenerators4SVCComp(ns25)
ns25.pf()
StoreSolutions(ns25, 'State25')
dfU.at['State25', 'U1'] = dfU.U2.State25 # The circuit for SVC compensation does not include bus U1 (because U1=U2), define it to enable solving HPFC operating point
CalculateHPFCop('State25')
dfS.at['State25', 'Note'] = 'Solved circuit compensated by an SVC to approximate the operating point of an optimized HPFC'

#%% Sensitivity case
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
    nh6.pf('b')
    StoreSolutions(nh6, 'State06b', snapshot='b')
    dfU.at['State06b', 'U1'] = dfU.U2.State06b # The circuit for SVC compensation does not include bus U1 (because U1=U2), define it to enable solving HPFC operating point
    CalculateHPFCop('State06b')
    dfS.at['State06b', 'Note'] = 'Solved circuit compensated by SVC using PV commands, P=0 V2pu+0.2% of uncompensated system'

#%% Sensitivity case
if False:
    print('Solving State07')
    Q1ref = np.imag(dfS.S1.State06a) - 30.
    Q2ref = np.imag(dfS.S2.State06a)
    dfS.at['State07s'] = dfS.loc['State03s']
    dfS.at['State07s', 'S1'] = np.complex(-np.real(dfS.S2.State03s), Q1ref) # reactive power hand adjusted to balance the converter ratings
    dfS.at['State07s', 'S2'] = np.complex( np.real(dfS.S2.State03s), Q2ref) # reactive power hand adjusted to balance the converter ratings
    dfS.at['State07s', 'Ss'] = dfS.Ss.State07s + 2.227  # hand corrected to restore ang(Us) = 18.8deg
    nh7 = CreateNetwork4PQQComp(Uspu, Urpu)
    nh7 = AddGeneratorsUsUr(nh7, np.real(dfS.Ss.State07s))
    nh7 = AddGenerators4PQQComp(nh7, np.real(dfS.S2.State07s), np.imag(dfS.S1.State07s), np.imag(dfS.S2.State07s))
    nh7.pf()
    StoreSolutions(nh7, 'State07')
    CalculateHPFCop('State07')
    dfS.at['State07', 'Note'] = 'Solved circuit compensated by HPFC with adjusted Q1cmd, Q2cmd to balance HPFC converters'

#%% Save results to Excel
print('Saving to Excel')
# caselist = dfS.index.tolist() 
caselist = ['State01a', 'State01s', 'State02', 'State03', 'State04', 
            'State11', 'State12', 
            'State23', 'State24', 'State25',
            'State32']
SaveToExcel(caselist, dirout='Results/', SortCases=True)

if OutputPlots:
    # Opening plot files
    foutLog.write('Starting to plot at: %s\n' %(str(datetime.now())))
    print('Opening plot files')     
    pltPdf1 = dpdf.PdfPages(os.path.join(dirout,fnamePlt))

    for case in ['State01a', 'State02', 'State03', 'State04', 'State11', 'State23', 'State24', 'State25', 'State32']:
        OutputVectorsPage(pltPdf1, case, 
                          pageTitle='Calculated by '+codeName+' v'+codeVersion+'\n\n'+r'$\bf{' + case + '}$')

    # Closing plot files
    print("Closing plot files")
    pltPdf1.close()

#%% time stamp and close log file
codeTfinish = datetime.now()
foutLog.write('\n\nRun finished at: %s\n' %(str(codeTfinish)))
codeTdelta = codeTfinish - codeTstart
foutLog.write('Run Lasted: %.3f seconds\n' %(codeTdelta.total_seconds()))
foutLog.close()
