#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 3 18:53:53 2018

@author: Jovan Z. Bebic

v1.0 JZB 20180803
Library functions for FACTS sizing to facilitiate scenario analysis

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
codeVersion = '1.0'
codeCopyright = 'GNU General Public License v3.0'
codeAuthors = 'Jovan Z Bebic\n'
codeName = 'SizeFACTS.py'
dirout = './'
fnameLog = 'SizeFACTS.log'
fnamePlt = 'SizeFACTS.pdf'
OutputPlots = True

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
            pd.Series({'Ss':Ss, 'S0':S0, 'S1':S1, 'Sm':Sm, 'Sm\'':Smm, 'S2':S2, 'S3':S3, 'S4':S4, 'Sr':Sr})]
    
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


#%% Create a baseline network: Bus1 is the same as Bus2
def CreateNetwork(Uspu, Urpu, UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data

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

#% Creates a network with separate buses 1 and 2 that are coupled via PQ specs at the terminals
def CreateNetwork4Comp(Uspu, Urpu, UbLL=220.):

    # create network object
    nw = pypsa.Network() # holds network data

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

#%% Add Generators
def AddGeneratorsUsUr(nw, Ps):
    #%% sets up system generators
    nw.add("Generator", "Us", bus="Us", control="PV", p_set=Ps)
    nw.add("Generator", "Ur", bus="Ur", control="Slack")
    return nw

def AddGenerators4Comp(nw, P, Q1, Q2):
    #%% set up equivlent generators for the compensator
    nw.add("Generator", "U1", bus="U1", control="PQ", p_set=-P, q_set=Q1)
    nw.add("Generator", "U2", bus="U2", control="PQ", p_set=P, q_set=Q2)
    return nw

#%% Store load flow solutions into dataframes
def StoreSolutions(nw, caseIx):
    # Store solved voltages all at once
    dfU.at[caseIx]=nw.buses_t.v_mag_pu.loc['now'] * np.exp(nw.buses_t.v_ang.loc['now']*1.j)
    # Store solved aparent powers, pull them out of active and reactive flows of circuit elements
    dfS.at[caseIx, 'Ss'] = np.complex(nw.lines_t.p0.Zs.now, nw.lines_t.q0.Zs.now)
    dfS.at[caseIx, 'S0'] = np.complex(nw.lines_t.p1.Zs.now, nw.lines_t.q1.Zs.now)
    dfS.at[caseIx, 'Sm'] = np.complex(nw.lines_t.p0.Zm.now, nw.lines_t.q0.Zm.now)
    dfS.at[caseIx, 'Sm\''] = np.complex(-nw.lines_t.p1.Zm.now, -nw.lines_t.q1.Zm.now)
    dfS.at[caseIx, 'S2'] = np.complex(nw.lines_t.p0.ZL1.now + nw.lines_t.p0.ZL2.now, nw.lines_t.q0.ZL1.now + nw.lines_t.q0.ZL2.now)
    dfS.at[caseIx, 'S3'] = np.complex(-nw.lines_t.p1.ZL1.now - nw.lines_t.p1.ZL2.now, -nw.lines_t.q1.ZL1.now - nw.lines_t.q1.ZL2.now)
    dfS.at[caseIx, 'S4'] = np.complex(nw.lines_t.p0.Zr.now, nw.lines_t.q0.Zr.now)
    dfS.at[caseIx, 'Sr'] = np.complex(-nw.lines_t.p1.Zr.now, -nw.lines_t.q1.Zr.now)
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
    dfHPFC.at[caseIx, 'SM'] = -3.*UM*np.conj(IM)/1000.

    UX = U1-UM
    UY = U2-UM

    dfHPFC.at[caseIx, 'SX'] = 3.*UX*np.conj(I1)/1000.
    dfHPFC.at[caseIx, 'SY'] = 3.*UY*np.conj(I2)/1000.
    dfHPFC.at[caseIx, 'UX'] = UX/Ub
    dfHPFC.at[caseIx, 'UY'] = UY/Ub

    return

#%% Save the results to Excel
def SaveToExcel(dirout='./', fnameXlsx='Results.xlsx', SortCases=False):

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
    for case in dfS.index.tolist():
        worksheet1.write_string(row, 0, case)
        worksheet1.write_string(row, 1, dfS.Note[case])
        row +=1

    # Preparing real and reactive power flows
    if SortCases: dfS.sort_index(inplace=True)
    dfP = dfS.applymap(lambda x: np.real(x)).round(1)
    dfQ = dfS.applymap(lambda x: np.imag(x)).round(1)

    dfP.columns = ['Re('+x+')' for x in dfS.columns.tolist()]
    dfQ.columns = ['Im('+x+')' for x in dfS.columns.tolist()]

    row = 0
    dfx1 = pd.concat([dfP, dfQ], axis=1, join_axes=[dfS.index])
    dfx1.to_excel(writer, 'Results', startrow=row)
    row += dfx1.index.size + 2

    # Preparing voltages 
    if SortCases: dfU.sort_index(inplace=True)
    dfUmagpu = dfU.applymap(lambda x: np.abs(x)).round(4)
    dfUang = dfU.applymap(lambda x: np.angle(x, deg=True)).round(2)
    dfUmag = dfU.applymap(lambda x: np.abs(x)*Ub*np.sqrt(2.)).round(1)

    dfUmagpu.columns = ['|'+x+'pu|' for x in dfU.columns.tolist()]
    dfUang.columns = ['ang('+x+')' for x in dfU.columns.tolist()]
    dfUmag.columns = ['|'+x+'|' for x in dfU.columns.tolist()]

    dfx2 = pd.concat([dfUmagpu, dfUang, dfUmag], axis=1, join_axes=[dfU.index])
    dfx2.to_excel(writer, 'Results', startrow=row)
    row += dfx2.index.size + 2

    # Preparing UPFC sizing info
    # dfUPFC = pd.DataFrame(columns=['Ssh', 'Sser', 'Ush', 'User'])
    if SortCases: dfUPFC.sort_index(inplace=True)
    df1 = dfUPFC.loc[:, ['Ssh', 'Sser']]
    df1re = df1.applymap(lambda x: np.real(x)).round(2)
    df1im = df1.applymap(lambda x: np.imag(x)).round(1)

    df1re.columns = ['Re('+x+')' for x in df1.columns.tolist()]
    df1im.columns = ['Im('+x+')' for x in df1.columns.tolist()]

    df2 = dfUPFC.loc[:, ['Ush', 'User']]
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
    # dfHPFC = pd.DataFrame(columns=['SM', 'SX', 'SY', 'UM', 'UX', 'UY'])
    if SortCases: dfHPFC.sort_index(inplace=True)
    df1 = dfHPFC.loc[:, ['SM', 'SX', 'SY']]
    df1re = df1.applymap(lambda x: np.real(x)).round(2)
    df1im = df1.applymap(lambda x: np.imag(x)).round(1)

    df1re.columns = ['Re('+x+')' for x in df1.columns.tolist()]
    df1im.columns = ['Im('+x+')' for x in df1.columns.tolist()]

    df2 = dfHPFC.loc[:, ['UM', 'UX', 'UY']]
    df2magpu = df2.applymap(lambda x: np.abs(x)).round(4)
    df2ang = df2.applymap(lambda x: np.angle(x, deg=True)).round(2)
    df2mag = df2.applymap(lambda x: np.abs(x)*Ub*np.sqrt(2.)).round(1)

    df2magpu.columns = ['|'+x+'pu|' for x in df2.columns.tolist()]
    df2ang.columns = ['ang('+x+')' for x in df2.columns.tolist()]
    df2mag.columns = ['|'+x+'|' for x in df2.columns.tolist()]

    dfx4 = pd.concat([df1re, df1im, df2magpu, df2ang, df2mag], axis=1, join_axes=[df1re.index])
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
dfS = pd.DataFrame(columns=['Ss', 'S0', 'S1', 'Sm', 'Sm\'', 'S2', 'S3', 'S4', 'Sr', 'Note'])
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

#%% Solve for accurate State1 flows: 'State1a'
[dfU.at['State1a'], dfS.at['State1a']] = SolveBaselineFlows(Us, Ur)
dfS.at['State1a', 'Note'] = "Solved baseline circuit ('a' = accurate)"
#%% Specified apparent powers
State1s = {'S0': -733   - 161.3j, 
           'S1': -701.8 - 154.4j,
           'S2':  701.8 + 154.4j,
           'S3':  700.6 + 128.5j,
           'S4':  731.7 + 134.2j}

State2s = {'S0': -881.6 - 111.5j, 
           'S1': -903.4 - 107.1j,
           'S2':  900.0 + 200.0j,
           'S3':  899.8 + 158.4j,
           'S4':  878.0 + 162.3j}

#%% Store specified values into dataframes
dfS.at['State1s'] = pd.Series(State1s)
dfS.at['State1s', 'Note'] = 'Apparent powers before compensation, as specified by NARI'
dfS.at['State2s'] = pd.Series(State2s)
dfS.at['State2s', 'Note'] = 'Apparent powers after compensation by UPFC, as specified by NARI and adjusted by Jovan to achieve ang(Us)=18.8deg'
dfS.at['State3s'] = pd.Series(State2s) # the same spec for HPFC as for the UPFC
dfS.at['State3s', 'Note'] = 'Apparent powers for compensation by HPFC, adjusted by Jovan to balance HPFC converter ratings.'
dfU.at['State1s'] = pd.Series({'Us': Us/Ub, 'Ur': Ur/Ub})
dfU.at['State2s'] = pd.Series({'Us': Us/Ub, 'Ur': Ur/Ub})
dfU.at['State3s'] = pd.Series({'Us': Us/Ub, 'Ur': Ur/Ub})

#%% Define per unit voltages used to set up load flows
Uspu = np.abs(dfU.Us.State1s)
Urpu = np.abs(dfU.Ur.State1s)

#%% Set up and solve baseline load flow 'State1'
n4c = CreateNetwork4Comp(Uspu, Urpu)
n4c = AddGeneratorsUsUr(n4c, np.real(dfS.Ss.State1a))
n4c = AddGenerators4Comp(n4c, np.real(dfS.S2.State1a), np.imag(dfS.S1.State1a), np.imag(dfS.S2.State1a))
n4c.pf()
StoreSolutions(n4c, 'State1')
dfS.at['State1', 'Note'] = 'Solved baseline circuit with apparent power set points from State1a'

#%% Increase flow by taking 200MW more from Us without any compensation
n11 = CreateNetwork(Uspu, Urpu)
n11 = AddGeneratorsUsUr(n11, np.real(dfS.Ss.State1)+200)
n11.pf()
StoreSolutions(n11, 'State11')
dfU.at['State11', 'U1'] = dfU.U2.State11
dfS.at['State11', 'Note'] = 'Solved baseline circuit with increased dispatch of Us units by 200MW'

#%% Set up and solve the UPFC-compensated load flow: State2
dfS.Ss.State2s = SolveSs(Uspu, dfS.S0.State2s)
dfS.Sr.State2s = SolveSr(Urpu, dfS.S4.State2s)
dfS.at['State2s', 'Ss'] = dfS.Ss.State2s - 3.431  # hand corrected to restore ang(Us) = 18.8deg
nu = CreateNetwork4Comp(Uspu, Urpu)
nu = AddGeneratorsUsUr(nu, np.real(dfS.Ss.State2s))
nu = AddGenerators4Comp(nu, np.real(dfS.S2.State2s), np.imag(dfS.S1.State2s), np.imag(dfS.S2.State2s))
nu.pf()
StoreSolutions(nu, 'State2')
CalculateUPFCop('State2')
dfS.at['State2', 'Note'] = 'Solved circuit compensated by UPFC'

#%% Set up and solve the HPFC-compensated system: 'State3'
dfS.Ss.State3s = SolveSs(Uspu, dfS.S0.State3s)
dfS.Sr.State3s = SolveSr(Urpu, dfS.S4.State3s)
Q1ref = np.imag(dfS.S1.State1) + (np.imag(dfS.S2.State2) - np.imag(dfS.S2.State1))
Q2ref = np.imag(dfS.S2.State3s)
dfS.at['State3s', 'S1'] = np.complex(-np.real(dfS.S2.State3s), Q1ref) # reactive power hand adjusted to balance the converter ratings
dfS.at['State3s', 'S2'] = np.complex( np.real(dfS.S2.State3s), Q2ref) # reactive power hand adjusted to balance the converter ratings
dfS.at['State3s', 'Ss'] = dfS.Ss.State3s - 3.407  # hand corrected to restore ang(Us) = 18.8deg
nh = CreateNetwork4Comp(Uspu, Urpu)
nh = AddGeneratorsUsUr(nh, np.real(dfS.Ss.State3s))
nh = AddGenerators4Comp(nh, np.real(dfS.S2.State3s), np.imag(dfS.S1.State3s), np.imag(dfS.S2.State3s))
nh.pf()
StoreSolutions(nh, 'State3')
CalculateHPFCop('State3')
dfS.at['State3', 'Note'] = 'Solved circuit compensated by HPFC with original Q1cmd, Q2cmd as were used in UPFC'

#%% Sensitivity case
dQ = 30
dfS.at['State4s'] = dfS.loc['State3s']
dfS.at['State4s', 'S1'] = np.complex(-np.real(dfS.S2.State3s), Q1ref) # reactive power hand adjusted to balance the converter ratings
dfS.at['State4s', 'S2'] = np.complex( np.real(dfS.S2.State3s), Q2ref+dQ) # reactive power hand adjusted to balance the converter ratings
dfS.at['State4s', 'Ss'] = dfS.Ss.State4s + 2.227  # hand corrected to restore ang(Us) = 18.8deg
nh4 = CreateNetwork4Comp(Uspu, Urpu)
nh4 = AddGeneratorsUsUr(nh4, np.real(dfS.Ss.State4s))
nh4 = AddGenerators4Comp(nh4, np.real(dfS.S2.State4s), np.imag(dfS.S1.State4s), np.imag(dfS.S2.State4s))
nh4.pf()
StoreSolutions(nh4, 'State4')
CalculateHPFCop('State4')
dfS.at['State4', 'Note'] = 'Solved circuit compensated by HPFC with adjusted Q1cmd, Q2cmd to balance HPFC converters'

#%% Sensitivity case
dfS.at['State5s'] = dfS.loc['State3s']
dfS.at['State5s', 'S1'] = np.complex(-np.real(dfS.S2.State3s), Q1ref) # reactive power hand adjusted to balance the converter ratings
dfS.at['State5s', 'S2'] = np.complex( np.real(dfS.S2.State3s), Q2ref+1.5*dQ) # reactive power hand adjusted to balance the converter ratings
nh5 = CreateNetwork4Comp(Uspu, Urpu)
nh5 = AddGeneratorsUsUr(nh5, np.real(dfS.Ss.State5s))
nh5 = AddGenerators4Comp(nh5, np.real(dfS.S2.State5s), np.imag(dfS.S1.State5s), np.imag(dfS.S2.State5s))
nh5.pf()
StoreSolutions(nh5, 'State5')
CalculateHPFCop('State5')
dfS.at['State5', 'Note'] = 'Spare case'

#%% Save results to Excel
SaveToExcel()

if OutputPlots:
    foutLog.write('Starting to plot at: %s\n' %(str(datetime.now())))
    print('Opening plot files')     
    pltPdf1 = dpdf.PdfPages(os.path.join(dirout,fnamePlt))

if OutputPlots:
    for case in ['State1a', 'State2', 'State3', 'State4', 'State5']:
        OutputVectorsPage(pltPdf1, case, 
                          pageTitle='Calculated by '+codeName+' v'+codeVersion+'\n\n'+r'$\bf{' + case + '}$')

#%% Closing plot files
if OutputPlots:
    print("Closing plot files")
    pltPdf1.close()

#%% time stamp and close log file
codeTfinish = datetime.now()
foutLog.write('\n\nRun finished at: %s\n' %(str(codeTfinish)))
codeTdelta = codeTfinish - codeTstart
foutLog.write('Run Lasted: %.3f seconds\n' %(codeTdelta.total_seconds()))
foutLog.close()
