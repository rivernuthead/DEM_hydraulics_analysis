#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:16:17 2022

@author: erri
"""

import numpy as np
from scipy import optimize as opt

######################################################################################
# FUNCTIONS
######################################################################################
def interpolate(func, xData, yData, ic=None, bounds=(-np.inf, np.inf)):
    # Interpolate data by fitting a given function, then returns the interpolated curve as a 1d array.
    par, covar = opt.curve_fit(func, xData, yData, p0=ic, maxfev=8000, bounds=bounds)
    if len(par) == 2:
        intCurve = func(xData, par[0], par[1])
    elif len(par) == 3:
        intCurve = func(xData, par[0], par[1], par[2])
    elif len(par) == 4:
        intCurve = func(xData, par[0], par[1], par[2], par[3])
    else:
        print("Interpolation failed. The interpolation function must have 2 or 3 parameters")
        intCurve = -1 * np.ones(len(xData))
    return par, intCurve, covar

# Scour and deposition volumes interpolation function
def func_exp(x,A,B):
    y = A*(1-np.exp(-x/B))
    return y

def func_exp2(x,A,B,C):
    y = C + A*(1-np.exp(-x/B))
    return y

# morphW interpolation function:
def func_exp3(x,A,B):
    y = ((A + (1-np.exp(-x/B)))/(A+1))*0.8
    return y

def func_exp4(x,A,B,C):
    y = A*C**(x/C)
    return y

def func_ln(x,A,B):
    y=A*np.ln(x/B)
    return y


def GaussPoints(NG):
    '''
    Funzione per il calcolo dei punti e dei pesi di Gauss

    Argomenti
    ---------
    NG: int
       numero di punti di Gauss

    Output
    ------
    p: numpy.ndarray
      array dei punti di Gauss
    w: numpy.ndarray
      array dei pesi
    '''
    p, w = None, None
    if NG==2:
        p = np.array([ -1/np.sqrt(3),
                       +1/np.sqrt(3) ])
        w = np.array([ 1, 1 ])
    elif NG==3:
        p = np.array([-(1/5)*np.sqrt(15),
                      0,
                      (1/5)*np.sqrt(15)])
        w = np.array([5/9, 8/9, 5/9])
    elif NG==4:
        p = np.array([+(1/35)*np.sqrt(525-70*np.sqrt(30)),
                      -(1/35)*np.sqrt(525-70*np.sqrt(30)),
                      +(1/35)*np.sqrt(525+70*np.sqrt(30)),
                      -(1/35)*np.sqrt(525+70*np.sqrt(30))])
        w = np.array([(1/36)*(18+np.sqrt(30)),
                      (1/36)*(18+np.sqrt(30)),
                      (1/36)*(18-np.sqrt(30)),
                      (1/36)*(18-np.sqrt(30))])

    return p, w


# Steady flow function
def MotoUniforme( S, y_coord, z_coord, D, NG, teta_c, ds):
    '''
    Calcola i parametri di moto uniforme per assegnato tirante

    Argomenti
    ---------

    S: float
       pendenza del canale
    y_coord: numpy.ndarray
      coordinate trasversali dei punti della sezione
    z_coord: numpy.ndarray
      coordinate verticali dei punti della sezione
    D: float
      profondità alla quale calcolare i parametri di moto uniforme
    NG: int [default=2]
      numero di punti di Gauss
    teta_c: float
        parametro di mobilità critico di Shiels
    ds: float
        diamentro medio dei sedimenti

    Output
    ------
    Q: float
      portata alla quale si realizza la profondità D di moto uniforme
    Omega: float
      area sezione bagnata alla profondita' D
    b: float
      larghezza superficie libera alla profondita' D
    alpha: float
      coefficiente di ragguaglio dell'energia alla profondita' D
    beta: float
      coefficiente di ragguaglio della qdm alla profondita' D
    '''
    # Punti e pesi di Gauss
    xj, wj = GaussPoints( NG ) # Calcola i putni e i pesi di Gauss

    #Dati
    delta = 1.65
    g = 9.806
    k = 5.3 # C = 2.5*ln(11*D/(k*ds))

    # Inizializzo
    Omega = 0 # Area bagnata
    array_teta = [] # Shields parameter array
    b = 0 # Larghezza superficie libera
    sumQs = 0 # Portata solida
    B=0
    #I coefficienti di ragguaglio sono relativi a tutta la sezione, si calcolano alla fine.
    num_alpha = 0 # Numeratore di alpha
    num_beta = 0 # Numeratore di beta
    den = 0 # Base del denominatore di alpha e beta
    Di = D - (z_coord-z_coord.min())  # Distribuzione trasversale della profondita'
    N = Di.size # Numero di punti sulla trasversale

    # N punti trasversali -> N-1 intervalli (trapezi)
    for i in range( N-1 ): # Per ogni trapezio

        #    vertical stripe
        #
        #         dy
        #
        #        o-----o       <- water level
        #        |     |
        #        |     |  DR
        #        |     |
        #        |     o      zR     _ _
        #    DL  |    /       ^       |
        #        |   / dB     |       |
        #        |  /         |       |  dz
        #        | /\\ phi    |      _|_
        #    zL  o  ------    |
        #    ^                |
        #    |                |
        #    ------------------- z_coord=0

        yL, yR = y_coord[i], y_coord[i+1]
        zL, zR = z_coord[i], z_coord[i+1]
        DL, DR = Di[i], Di[i+1]
        dy = yR - yL
        dz = zR - zL
        dB = np.sqrt(dy**2+dz**2)
        cosphi = dy/dB
        # Geometric parameters:
        if DL<=0 and DR<=0:
            dy, dz = 0, 0
            DL, DR = 0, 0
        elif DL<0:
            dy = -dy*DR/dz
            dz = DR
            DL = 0
        elif DR<0:
            dy = dy*DL/dz
            dz = DL
            DR = 0

        #Metodo di Gauss:
        SUM = np.zeros(3)
        C = 0
        Dm = 0
        teta1=0

        # Gauss weight loop
        for j in range(NG):
            Dm = (DR+DL)/2# + (DR-DL)/2*xj[j]
            # print(Dm)
            # print('tirante:', Dm, '   k:', k, '   ds:', ds)

            if Dm==0 or 2.5*np.log(11*Dm/(k*ds))<0:
                C=0
            else:
                C = 2.5*np.log(11*Dm/(k*ds))

            #den
            SUM[0] += wj[j]*C*Dm**(3/2)
            #num_alpha
            SUM[1] += wj[j]*C**(3)*Dm**(2.5)
            #num_beta
            SUM[2] += wj[j]*C**(2)*Dm**(2)

        den += dy/2*cosphi**(1/2)*SUM[0]
        num_alpha += dy/2*cosphi**(3/2)*SUM[1]
        num_beta += dy/2*cosphi*SUM[2]

        dOmega = (DR + DL)*dy/2

        #Calcolo di Omega: superficie della sezione
        Omega += dOmega

        #Calcolo di B: lunghezza del perimetro bagnato

        B += dB

        #Calcolo di b: larghezza della superficie libera
        b += dy

        #Calcolo di b: larghezza della superficie libera
        #Rh=Omega/B

        #Shields parameter
        teta_primo = (Dm*cosphi)*S/(delta*ds)
        array_teta = np.append(array_teta, teta_primo)


    count_active = np.count_nonzero(np.where(array_teta>=teta_c, 1, 0))



    #Calcolo della portata Q
    Q = np.sqrt(S*g)*den

    #Calcolo della capacità di trasporto
    teta1 = (Omega/B)*S/(delta*ds)
    if teta1 >= teta_c:
        Qs = 8*(teta1-teta_c)**1.5*np.sqrt(9.81*delta*ds**3)*b
    else:
        Qs = 0
    # sumQs += qs
    Qs = sumQs

    #Condizione per procedere al calcolo anche quando il punto i è sommerso
    # mentre i+1 no.
    if den==0:
        alpha = None
        beta = None
    else:
        alpha = Omega**2*(g*S)**(3/2)*num_alpha/den**3
        beta = Omega*g*S*num_beta/den**2

    return Q, Omega, b, B, alpha, beta, Qs, count_active

