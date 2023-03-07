#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:16:17 2022

@author: erri
"""

import numpy as np

######################################################################################
# FUNCTIONS
######################################################################################
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

def MotoUniformeC( S, y_coord, z_coord, D, NG, ds):
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
    xj, wj = GaussPoints( NG ) # Calcola i punti e i pesi di Gauss

    #Dati
    k = 5.3 # C = 2.5*ln(11*D/(k*ds))
    g = 9.806
    ni = 1e-06
    # Inizializzo
    Omega = 0 # Area bagnata
    b = 0 # Larghezza superficie libera
    B=0

    #I coefficienti di ragguaglio sono relativi a tutta la sezione, si calcolano alla fine.
    num_alpha = 0 # Numeratore di alpha
    num_beta = 0 # Numeratore di beta
    den = 0 # Base del denominatore di alpha e beta
    Di = D - (z_coord-z_coord.min())  # Distribuzione trasversale della profondita'
    N = Di.size # Numero di punti sulla trasversale
    Dmed = []
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

        # Gauss weight loop
        for j in range(NG):
            Dm = (DR+DL)/2 + (DR-DL)/2*xj[j]
            if Dm==0 or 2.5*np.log(11*Dm/(k*ds))<0:
                C=0
            else:
                C = 2.5*np.log(11*Dm/(k*ds))
                Dmed = np.append(Dmed,Dm)
            dOmega = Dm*dy/NG
            #Calcolo di Omega: superficie della sezione
            Omega += dOmega
            #den
            SUM[0] += wj[j]*C*Dm**(3/2)
            #num_alpha
            SUM[1] += wj[j]*C**(3)*Dm**(2.5)
            #num_beta
            SUM[2] += wj[j]*C**(2)*Dm**(2)

        den += dy/2*cosphi**(1/2)*SUM[0]
        num_alpha += dy/2*cosphi**(3/2)*SUM[1]
        num_beta += dy/2*cosphi*SUM[2]

        
        #Calcolo di B: lunghezza del perimetro bagnato

        B += dB

        #Calcolo di b: larghezza della superficie libera
        b += dy

    #Calcolo della portata Q
    Q = np.sqrt(S*g)*den
    U = Q/Omega
    Rh = Omega/B
    #Calcolo del numero di Reynolds    
    Re = U*Rh/ni
    #Calcolo del numero di Reynolds
    Fr = U/np.sqrt(g*D)
    

    #Condizione per procedere al calcolo anche quando il punto i è sommerso
    # mentre i+1 no.
    if den==0:
        alpha = None
        beta = None
    else:
        alpha = (Omega**2)*num_alpha/(den**3)
        beta = Omega*num_beta/(den**2)

    return Q, Omega, b, B, alpha, beta, Re, Fr, Rh, Dmed

def MotoUniformeGS( S, y_coord, z_coord, D, NG, ds, ks):
    '''
    Calcola i parametri di moto uniforme per assegnato tirante
    
    Calcolo eseguito con il metodo di Gauss Legendre
    
    
    Argomenti
    ---------

    iF: float
       pendenza del canale
    y: numpy.ndarray
      coordinate trasversali dei punti della sezione
    z: numpy.ndarray
      coordinate verticali dei punti della sezione
    ks: numpy.ndarray
      coefficienti di scabrezza dei punti della sezione
    Y: float
      profondità alla quale calcolare i parametri di moto uniforme
    NG: int [default=2]
      numero di punti di Gauss

    Output
    ------
    Q: float
      portata alla quale si realizza la profondità Y di moto uniforme
    Omega: float
      area sezione bagnata alla profondita' Y
    b: float
      larghezza superficie libera alla profondita' Y
    alpha: float
      coefficiente di ragguaglio dell'energia alla profondita' Y
    beta: float
      coefficiente di ragguaglio della qdm alla profondita' Y
    '''
    # Punti e pesi di Gauss
    xj, wj = GaussPoints( NG ) # Calcola i punti e i pesi di Gauss

    #Dati
    g = 9.806
    ni = 1e-06
    # Inizializzo
    Omega = 0 # Area bagnata
    B = 0
    Q = 0
    b = 0 # Larghezza superficie libera
    num_alpha = 0 # Numeratore di alpha
    num_beta = 0 # Numeratore di beta
    den = 0 # Base del denominatore di alpha e beta
    cos_phi = 0
    Di = (D - (z_coord-z_coord.min()))  # Distribuzione trasversale della profondita'
    N = Di.size # Numero di punti sulla trasversale
    
    for i in range( N-1 ):
        yL, yR = y_coord[i], y_coord[i+1]
        zL, zR = z_coord[i], z_coord[i+1]
        DL, DR = Di[i], Di[i+1]
        dy = yR - yL
        dz = zR - zL
        dB = np.sqrt(dy**2+dz**2)
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
            
        
        b += dy 
        B += dB
        if dy != 0:
            cos_phi = dy/np.sqrt(dy**2 + dz**2)
            
        #Metodo di Gauss
        Q_hat = 0
        
        for j in range(NG):
            D_hat = (DR-DL)*0.5*xj[j]+(DR+DL)*0.5
            Omega += D_hat*dy/NG
            Q_hat += wj[j]*ks*D_hat**(5/3)
            den += wj[j]*0.5*(cos_phi**(2/3))*ks*(D_hat**(5/3))*dy
            num_alpha += wj[j]*0.5*(cos_phi**2)*(ks**3)*(D_hat**3)*dy
            num_beta += wj[j]*0.5*(cos_phi**(4/3))*(ks**2)*(D_hat**(7/3))*dy
        Q += 0.5*cos_phi**(2/3)*dy*Q_hat
   
    alpha = (Omega**2)*num_alpha/(den**3)
    beta = Omega*num_beta/(den**2)    
    Q = np.sqrt(S)*Q
    U = Q/Omega
    Rh = Omega/B
    #Calcolo del numero di Reynolds    
    Re = U*Rh/ni
    #Calcolo del numero di Reynolds
    Fr = U/np.sqrt(g*D)
    
       
    return Q, Omega, b, B, alpha, beta, Re, Fr

def QSMPM( S, y_coord, z_coord, D, NG, teta_c, ds):
    '''
    Calcola la portata solida di moto uniforme per assegnato tirante

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
    Qs: float
      portata solida che si realizza con la profondità D di moto uniforme
    '''
    # Punti e pesi di Gauss
    xj, wj = GaussPoints( NG ) # Calcola i putni e i pesi di Gauss

    #Dati
    g = 9.806
    delta = 1.65
    ni = 1e-06
    # Inizializzo
    Omega = 0 # Area bagnata
    b = 0 # Larghezza superficie libera
    aw = 0 #larghezza attiva
    B = 0 #contorno bagnato
    Di = D - (z_coord-z_coord.min())  # Distribuzione trasversale della profondita'
    N = Di.size # Numero di punti sulla trasversale
    qs_array = []
    theta_array = []
    phi_array = []
    # N punti trasversali -> N-1 intervalli (trapezi)
    for i in range( N-1 ):
        yL, yR = y_coord[i], y_coord[i+1]
        zL, zR = z_coord[i], z_coord[i+1]
        DL, DR = Di[i], Di[i+1]
        dy = yR - yL
        dz = zR - zL
        dB = np.sqrt(dy**2+dz**2)
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
        
        b += dy 
        B += dB
            
        #Metodo di Gauss
        for j in range(NG):  
            D_hat = (DR-DL)*0.5*xj[j]+(DR+DL)*0.5
            dOmega = D_hat*dy/NG
            Omega += dOmega
            #Shields parameter
            if dy != 0:
                theta1 = (dOmega/(dB/NG))*S/(delta*ds)
                
            #Calcolo della capacità di trasporto
            theta_array = np.append(theta_array, theta1)   
            if theta1 >= teta_c:
                aw += dB/NG      
            if theta1 >= 0.047:
                phi = wj[j]*8*(theta1**1.5)*(1-0.047/theta1)**1.5
                qs  = phi*np.sqrt(g*delta*ds**3)*dB/NG
            else:
                qs = 0
            qs_array = np.append(qs_array, qs)
            phi_array = np.appen(phi_array, phi)
            
    Qs = np.sum(qs_array)*aw
    active_width = aw/b
    Rp = np.sqrt(g*delta*ds**3)/ni
    
    return Qs, active_width, Rp, theta_array, np.quantile(theta_array,0.9), phi_array

def QSParker( S, y_coord, z_coord, D, NG, ds):
    '''
    Calcola la portata solida di moto uniforme per assegnato tirante

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
    Qs: float
      portata solida che si realizza con la profondità D di moto uniforme
    '''
    # Punti e pesi di Gauss
    xj, wj = GaussPoints( NG ) # Calcola i putni e i pesi di Gauss

    #Dati
    g = 9.806
    delta = 1.65
    ni = 1e-06
    # Inizializzo
    Omega = 0 # Area bagnata
    b = 0 # Larghezza superficie libera
    aw = 0 #larghezza attiva
    B = 0 #contorno bagnato
    Di = D - (z_coord-z_coord.min())  # Distribuzione trasversale della profondita'
    N = Di.size # Numero di punti sulla trasversale
    qs_array = []
    theta_array = []
    phi_array = []
    # N punti trasversali -> N-1 intervalli (trapezi)
    for i in range( N-1 ):
        yL, yR = y_coord[i], y_coord[i+1]
        zL, zR = z_coord[i], z_coord[i+1]
        DL, DR = Di[i], Di[i+1]
        dy = yR - yL
        dz = zR - zL
        dB = np.sqrt(dy**2+dz**2)
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
        
        b += dy 
        B += dB
            
        #Metodo di Gauss
        
        for j in range(NG):  
            D_hat = (DR-DL)*0.5*xj[j]+(DR+DL)*0.5
            dOmega = D_hat*dy/NG
            Omega += dOmega
            #Shields parameter
            theta1 = (dOmega/(dB/NG))*S/(delta*ds)
            theta_array = np.append(theta_array, theta1) # Add \teta value even if \teta is less than \teta_{c}
            #Calcolo della capacità di trasporto
            if theta1 >= 0.03:
                aw += dB/NG
                phi = wj[j]*11.2*(theta1**1.5)*(1-0.03/theta1)**4.5
                qs  = phi*np.sqrt(g*delta*ds**3)*dB/NG
                # theta_array = np.append(theta_array, theta1)
                phi_array = np.append(phi_array, phi)
            else:
                qs = 0
            qs_array = np.append(qs_array, qs)
        
    Qs = np.sum(qs_array)*aw
    active_width = aw/b
    Rp = np.sqrt(g*delta*ds**3)/ni
    
    return Qs, active_width, Rp, theta_array, phi_array