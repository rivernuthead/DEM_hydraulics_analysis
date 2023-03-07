# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 17:35:59 2022

@author: Marco
"""

# ------------------

# =========================
# Import Pacchetti e Moduli
# =========================
import os
import time
import numpy as np # Libreria numerica
from matplotlib import pyplot as plt # Libreria grafica
from DEM_hydraulics_analysis_functions import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes

start = time.time() # Set initial time

###############################################################################
# SETUP SCRIPT PARAMETERS and RUN MODE
###############################################################################

'''
run mode:
    1 = one run at time
    2 = batch process
DEM analysis mode:
    0 = do not perform DEM analysis
    1 = perform DEM analysis
data_interpolation_mode:
    0 = no interpolation
    1 = data interpolation
mask mode:
    1 = mask the flume edge
    2 = mask the upstream half flume
    3 = mask the downstream half flume
process mode: (NB: set DEMs name)
    1 = batch process
    2 = single run process
Resistance equation:
    1 Chezy
    2 Gauckler-Strickler
Solid Discharge Equation:
    1 MPM
    2 Parker
save mode:
    0 = save only reports
    1 = save all chart and figure
'''

run_mode = 1
DEM_analysis_mode = 1
data_interpolation_mode = 1
mask_mode = 1
process_mode = 1
resistance_eq = 1
solid_discharge_equation = 2
save_plot_mode = 1

# SINGLE RUN NAME
run = 'q07_1'
Q = 0.7 # Run discharge [l/s]

# Survey pixel dimension
px_x = 50 # [mm]
px_y = 5 # [mm]

# Not a number raster value (NaN)
NaN = -999

# Engelund-Gauss model parameters
g = 9.806 # Gravity
ds = 0.001  # Sediment grainsize [mm]
teta_c = 0.047 # Shield parameter [-]
NG = 4 # Number of Gauss points
max_iter = 100000 # Maximum number of iterations
toll = 1e-05 # Convergence tolerance
delta = 1.65
ks = 21.1/((ds)**(1/6))
S = 0.01 # Pendenza



###############################################################################
# SETUP FOLDERS and RUNS
###############################################################################
# setup working directory and DEM's name
home_dir = os.getcwd()
report_dir = os.path.join(home_dir, 'output')
plot_dir = os.path.join(home_dir, 'plot')
run_dir = os.path.join(home_dir, 'surveys')

# Create the run name list
RUNS=[]
if run_mode ==2: # batch run mode
    for RUN in sorted(os.listdir(run_dir)): # loop over surveys directories
        if RUN.startswith('q'): # Consider only folder names starting wit q
            RUNS = np.append(RUNS, RUN) # Append run name at RUNS array
elif run_mode==1: # Single run mode
    RUNS=run.split() # RUNS as a single entry array, provided by run variable

###############################################################################
# INITIALIZE OVERALL REPORT ARRAYS
###############################################################################
# Define Engelund Gauss model report matrix:
# D [m], Q [m^3/s], Wwet/W [-]
engelund_model_report = np.zeros((len(RUNS),3))
###############################################################################
# MAIN LOOP OVER RUNS
###############################################################################
for run in RUNS:

    ###########################################################################
    # SETUP FOLDERS
    ###########################################################################
    print('######')
    print(run)
    print('######')
    print()
    # setup working directory and DEM's name
    input_dir = os.path.join(home_dir, 'surveys', run)
    report_run_dir = os.path.join(report_dir, run)

    # CREATE FOLDERS
    if not(os.path.exists(report_dir)):
        os.mkdir(report_dir)
    if not(os.path.exists(plot_dir)):
        os.mkdir(plot_dir)


    # IMPORT RUN PARAMETERS from file parameters.txt
    # variable run must be as 'q' + discharge + '_' repetition number
    # Parameters.txt structure:
    # discharge [l/s],repetition,run time [min],Texner discretization [-], Channel width [m], slope [m/m]
    # Load parameter matrix:
    parameters = np.loadtxt(os.path.join(home_dir, 'parameters.txt'),delimiter=',',skiprows=1)
    # Extract run parameter depending by run name
    run_param = parameters[np.intersect1d(np.argwhere(parameters[:,0]==float(run[-4:-2])/10),np.argwhere(parameters[:,0]==float(run[1:3])/10)),:]

    # Run time data
    dt = run_param[0,2] # dt between runs [min] (real time)
    dt_xnr = run_param[0,3] # temporal discretization in terms of Exner time (Texner between runs)
    
    # Flume geometry parameters
    W = run_param[0,4] # Flume width [m]
    S = run_param[0,5] # Flume slope
    
    # Sediment parameters
    d50 = run_param[0,6]

    # Run discharge
    #Q = run_param[0,0] # Run discharge [l/s]
    
    
    files=[] # initializing filenames list
    # Creating array with file names:
    for f in sorted(os.listdir(input_dir)):
        path = os.path.join(input_dir, f)
        if os.path.isfile(path) and f.endswith('.txt') and f.startswith('matrix_bed_norm_' + run[0:5] + 's'):
            files = np.append(files, f)


    ###########################################################################
    # CHECK DEMs SHAPE
    ###########################################################################
    # Due to differences between DEMs shape (not the same ScanArea.txt laser survey file)
    # a preliminary loop over the all DEMs is required in order to define the target
    # dimension of the reshaping operation
    array_dim_x = []
    array_dim_y = []
    for f in files:
        path_DEM = os.path.join(input_dir, f)
        DEM = np.loadtxt(path_DEM,
                          # delimiter=',',
                          skiprows=8
                          )
        array_dim_x = np.append(array_dim_x, DEM.shape[0])
        array_dim_y = np.append(array_dim_y, DEM.shape[1])

    # Define target dimension:
    shp_target_x, shp_target_y = int(min(array_dim_x)), int(min(array_dim_y))

    arr_shape = np.array([shp_target_x, shp_target_y]) # Define target shape

    ###########################################################################
    # INITIALIZE ARRAYS
    ###########################################################################
    # DEFINE REPORT FILES PATH
    phi_report_path = os.path.join(report_dir, run, run + '_phi_report.txt')
    theta_report_path = os.path.join(report_dir, run, run + '_theta_report.txt')
    
    # CREATE THE REPORT DIR
    if not os.path.exists(report_run_dir):
        os.makedirs(report_run_dir)
        print(f"The folder '{report_run_dir}' has been created successfully!")
    else:
        print(f"The folder '{report_run_dir}' already exists.")
    
    # CHECK IF THE FILES ALREASY EXIST AND IF SO REMOVE THEM
    if os.path.exists(phi_report_path):
        os.remove(phi_report_path)
    if os.path.exists(theta_report_path):
        os.remove(theta_report_path)
    
    # CREATE THE REPORT FILES
    if not os.path.isfile(phi_report_path):
        open(phi_report_path, 'w').close()
    if not os.path.isfile(theta_report_path):
        open(theta_report_path, 'w').close()
    
    ###########################################################################
    # SETUP MASKS
    ###########################################################################
    # array mask for filtering data outside the channel domain
    # Different mask will be applied depending on the run due to different ScanArea
    # used during the laser surveys
    runs_list = ['q10_1', 'q10_2', 'q15_1', 'q20_1', 'q20_2'] # Old runs with old ScanArea
    array_mask_name, array_mask_path = 'array_mask.txt', home_dir # Mask for runs 07 onwards

    if run in runs_list:
        array_mask_name, array_mask_path = 'array_mask_0.txt', home_dir
        print(array_mask_name)


    # Load mask
    array_mask = np.loadtxt(os.path.join(array_mask_path, array_mask_name))
    # Reshape mask:
    array_mask_rshp = array_mask[:shp_target_x,:shp_target_y] # Array mask reshaped

    # Create array mask:
    # - array_mask: np.array with 0 and 1
    # - array_mask_nan: np.array with np.nan and 1
    array_mask_rshp = np.where(array_mask_rshp==NaN, 0, 1) # Convert in mask with 0 and 1
    array_mask_rshp_nan = np.where(array_mask_rshp==0, np.nan, 1) # Convert in mask with np.nan and 1

    # Here we can split in two parts the DEMs or keep the entire one
    if mask_mode==1:
        pass
    elif mask_mode==2: # Working downstream, masking upstream
       array_mask_rshp[:,:-int(array_mask_rshp.shape[1]/2)] = NaN
       array_mask_rshp=np.where(array_mask_rshp==NaN, np.nan, array_mask_rshp)

    elif mask_mode==3: # Working upstream, masking downstream
        array_mask_rshp[:,int(array_mask_rshp.shape[1]/2):] = NaN
        array_mask_rshp=np.where(array_mask_rshp==NaN, np.nan, array_mask_rshp)

    ###########################################################################
    # DEM ANALYSIS
    ###########################################################################
    if DEM_analysis_mode==1:
        # - Residual slope, for each DEM
        # - Bed Relief Index (BRI) averaged, for each DEM
        # - STDEV (SD) of the bed elevation, for each DEM
        # Initialize arrays
        slope_res = [] # Rsidual slope array
        BRI=[] # BRi array
        SD = [] # SD array
        engelund_model_array=[] # Engelund model array (Q, D, Wwet/w])
        water_depth_array=[] # Water depth array [m]
        discharge_array=[] # Discarge [m^3/s]
        Wwet_array = [] # Wwet array [Wwet/W]
        solid_discharge_array = []
        active_width_array = []
        Rp_array = []
        teta_array = []
        teta90_array = []
        # morphWact_values = [] # All the morphological active width values for each runs

        for f in files:
            DEM_path = os.path.join(input_dir, f) # Set DEM path
            DEM = np.loadtxt(DEM_path,          # Load DEM data
                             #delimiter=',',
                             skiprows=8)
            DEM = np.where(np.isclose(DEM, NaN), np.nan, DEM)

            # DEM reshaping according to arr_shape...
            DEM=DEM[0:arr_shape[0], 0:arr_shape[1]]

            # DEM masking...
            DEM = DEM*array_mask_rshp_nan

            # Residual slope
            # NB: this operation will be performed to detrended DEMs
            # Averaged crosswise bed elevation array:
            bed_profile = np.nanmean(DEM, axis=0) # Bed profile
            # Linear regression of bed profile:
            # Performing linear regression
            x_coord = np.linspace(0, px_x*len(bed_profile), len(bed_profile)) # Longitudinal coordinate
            linear_model = np.polyfit(x_coord, bed_profile,1) # linear_model[0]=m, linear_model[1]=q y=m*x+q
            slope_res = np.append(slope_res, linear_model[0]) # Append residual slope values

            # PLOT cross section mean values and trendline
            # fig, ax1 = plt.subplots(dpi=200)
            # ax1.plot(x_coord, bed_profile)
            # ax1.plot(x_coord, x_coord*linear_model[0]+linear_model[1], color='red')
            # ax1.set(xlabel='longitudinal coordinate (mm)', ylabel='Z (mm)',
            #        title=run+'\n'+'Residual slope:'+str(linear_model[0]))

            # BRI calculation
            BRI=np.append(BRI,np.mean(np.nanstd(DEM, axis=0))) # Overall BRI
            
            if f == files[0]:
                BRI_array = np.nanstd(DEM, axis=0) # Array crosswise for each DEM
            else:
                BRI_array = np.vstack((BRI_array, np.nanstd(DEM, axis=0)))

            # Bed elevation STDEV
            SD = np.append(SD,np.nanstd(DEM))

            # Create report matrix:
            # Structure: DEM name, residual slope [m/m], BRI [mm], SD [mm]
            matrix_DEM_analysis = np.transpose(np.stack((slope_res, BRI, SD)))

            # Build report
            report_DEM_header = 'DEM name, residual slope [m/m], BRI [mm], SD [mm]'
            report_DEM_name = run+'_DEM_report.txt'
            with open(os.path.join(report_dir, report_DEM_name), 'w') as fp:
                fp.write(report_DEM_header)
                fp.writelines(['\n'])
                for i in range(0,len(matrix_DEM_analysis[:,0])):
                    for j in range(0, len(matrix_DEM_analysis[0,:])+1):
                        if j == 0:
                            fp.writelines([files[i]+', '])
                        elif j==1:
                            # fp.writelines(["%.6f, " % float(matrix_DEM_analysis[i,j-1])])
                            fp.writelines(["{:e},".format(matrix_DEM_analysis[i,j-1])])
                        else:
                            fp.writelines(["%.3f, " % float(matrix_DEM_analysis[i,j-1])])
                    fp.writelines(['\n'])
            fp.close()

            # DEM detrending (DEM detrended both with slope and residual slope)
            DEM_detrended = DEM
            for i in range(0,DEM.shape[1]):
                DEM_detrended[:,i] = DEM[:,i]-linear_model[0]*i*px_x

#%%
            # Create equivalent cross section as sorted DEM vaues excluding NaN
            DEM_values = sorted(DEM_detrended[np.logical_not(np.isnan(DEM_detrended))]) # array with DEM values
            # cross_section_eq = DEM_values[::100] # Resize DEM value to be lighter (100 res resampling)
            cross_section_eq = np.interp(np.arange(0,len(DEM_values),len(DEM_values)/W/100), np.arange(0,len(DEM_values)), DEM_values)
            # Add cross section banks as the double of the maximum DEM's value:
            z_coord = np.pad(cross_section_eq, (1,1), mode='constant', constant_values=int(cross_section_eq.max()*2))
            z_coord = z_coord/1000 # Convert z_coord in meters
#%%
            # Create cross-wise coordination
            y_coord = np.arange(0,W*1000, W*1000/len(z_coord))
            y_coord = y_coord/1000 # Convert y_coord in meters

            # ENGENLUND-GAUSS IMPLEMENTATION

            Dmax = z_coord.max()-z_coord.min() # Maximum water depth
            Dmin = 0 # Minimum water level
            i=0 # Initialize iteration counter

            # Guess values:
            D0 = (Dmax-Dmin)/2 # Water depth
            if resistance_eq == 1:
                Qn, Omega, b, B, alpha, beta, Re, Fr, Rh, Dmed = MotoUniformeC(S, y_coord, z_coord, D0, NG, ds) # Discharge
                # Discharge extreme values
                Qmax, Omegamax, bmax, Bmax, alphamax, betamax, Remax, Frmax, Rhmax, Dmedmax = MotoUniformeC(S, y_coord, z_coord, Dmax, NG, ds)
                Qmin, Omegamin, bmin, Bmin, alphamin, betamin, Remin, Frmin, Rhmin, Dmedmin = MotoUniformeC(S, y_coord, z_coord, Dmin, NG, ds)
                
            else:
                Qn, Omega, b, B, alpha, beta, Re, Fr = MotoUniformeGS(S, y_coord, z_coord, D0, NG, ds, ks) # Discharge
                # Discharge extreme values
                Qmax, Omega, b, B, alpha, beta, Re, Fr = MotoUniformeGS(S, y_coord, z_coord, Dmax, NG, ds, ks)
                Qmin, Omega, b, B, alpha, beta, Re, Fr = MotoUniformeGS(S, y_coord, z_coord, Dmin, NG, ds, ks)
            
                
            Q_target = Q/1000 # Target discharge [m^3/s]
            if np.sign(Qmax-Q_target)==np.sign(Qmin-Q_target):
                print(' Solution out of boundaries')
            else:
                # Check if h<h_min:
                while abs(Qn - Q_target)>toll:
                    if i>max_iter:
                        print('ERROR: max iterations reached!')
                        break
                    i+=1
                    D0 = (Dmax+Dmin)/2
                    if resistance_eq == 1:
                        Q0, Omega, b, B, alpha, beta, Re, Fr, Rh, Dmed = MotoUniformeC(S, y_coord, z_coord, D0, NG, ds)
                    else:
                        Q0, Omega, b, B, alpha, beta, Re, Fr = MotoUniformeGS(S, y_coord, z_coord, D0, NG, ds, ks)
                    if Q0>Q_target:
                        Dmax=D0 # Update Dmax
                    elif Q0<Q_target:
                        Dmin=D0 # Update Dmin
                    Qn=Q0
                    if solid_discharge_equation == 1:
                        Qs, active_width, Rp, theta_array, teta90_array, phi_array = QSMPM(S, y_coord, z_coord, D0, NG, teta_c, ds)
                    else:
                        Qs, active_width, Rp, theta_array, phi_array = QSParker(S, y_coord, z_coord, D0, NG, ds)    
            
            solid_discharge_array = np.append(solid_discharge_array, Qs)   
            active_width_array = np.append(active_width_array, active_width)
            water_depth_array = np.append(water_depth_array, D0) # Water depth array
            discharge_array=np.append(discharge_array, Q0) # Discarge
            Wwet_array = np.append(Wwet_array, b/W)
            Rp_array = np.append(Rp_array, Rp)  
            

            # SAVE REPORT
            
            with open(phi_report_path, 'a') as file:
                np.savetxt(file, phi_array.reshape(1, -1), delimiter=',')
                
            with open(theta_report_path, 'a') as file:
                np.savetxt(file, theta_array.reshape(1, -1), delimiter=',')



        # BRI plot
        #TODO
        n_data = np.linspace(0,len(files)-1,len(files)) # Linspace of the number of available DoD
        c_data = np.linspace(0,1,len(files)) # c_data needs to be within 0 and 1
        colors = plt.cm.viridis(c_data)
        fig1, axs = plt.subplots(1,1,dpi=400, sharex=True, tight_layout=True, figsize=(8,6))
        #Defines the size of the zoom window and the positioning
        axins = inset_axes(axs, 2, 5, loc = 1, bbox_to_anchor=(1.3, 0.9),
                            bbox_transform = axs.figure.transFigure)
        for d, color in zip(n_data, colors):
            DoD_name = files[int(d)]
            axs.plot(np.linspace(0,DEM.shape[1]-1, DEM.shape[1])*px_x/1000, BRI_array[int(d),:], '-', c=color, label=DoD_name[21:23])
            plt.plot(np.linspace(0,DEM.shape[1]-1, DEM.shape[1])*px_x/1000, BRI_array[int(d),:], '-', c=color, label=DoD_name[21:23])
        
        # axins.scatter(x, y)
        x1, x2 = 12, 14
        y1, y2 = np.min(BRI_array[:,int(x1/px_x*1000):])*0.9, np.max(BRI_array[:,int(x1/px_x*1000):])*1.1 #Setting the limit of x and y direction to define which portion to #zoom
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        #Draw the lines from the portion to zoom and the zoom window
        mark_inset(axs, axins, loc1=1, loc2=3, fc="none", ec = "0.4")
        axs.set_title(run + ' - BRI', fontsize=14)
        axs.set_xlabel('Longitudinal coordinate [m]', fontsize=12)
        axs.set_ylabel('BRI', fontsize=12)
        # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
        plt.legend(loc='best', fontsize=8)
        
        # ax_new = fig1.add_axes([0.2, 1.1, 0.4, 0.4])
        # plt.plot(np.linspace(0,array.shape[1]-1, array.shape[1])*px_x, cross_bri_matrix[int(d),:], '-', c=color)
        
        plt.show()

        # BRI errorbar plot
        fig1, axs = plt.subplots(1,1,dpi=200, sharex=True, tight_layout=True, figsize=(8,6))
        axs.errorbar(np.linspace(0,DEM.shape[1]-1, DEM.shape[1])*px_x/1000, np.nanmean(BRI_array, axis=0), np.std(BRI_array, axis=0), linestyle='--', marker='^', color='darkcyan')
        axs.tick_params(axis='y', labelcolor='darkcyan')
        axs.set_title(run, fontsize=14)
        # axs.set_xlim(0, np.max(mean_matrix[1,:]*px_x)+1)
        axs.set_xlabel('Window analysis length [m]', fontsize=12)
        axs.set_ylabel('BRI [mm]', fontsize=12)
        # plt.savefig(os.path.join(plot_dir, run +'_morphW_interp.png'), dpi=200)
        plt.show()


        water_depth=np.mean(water_depth_array) # Average water depth
        discharge=np.mean(discharge_array) # Average discarge
        Wwet = np.mean(Wwet_array)
        solid_discharge = np.mean(solid_discharge_array)
        solid_discharge = solid_discharge*1e6*2.65
        active_width = np.mean(active_width_array)
        Rp = np.mean(Rp_array)
        teta = np.mean(teta_array)
        teta90mean = np.mean(teta90_array)
        
        print('Engelund-Gauss model results:')
        print('Reached discharge: ', discharge, ' m^3/s')
        print('Water depth: ', water_depth, ' m')
        print('Wwet/W: ', Wwet)
        print('Solid Discharge: ', solid_discharge, 'g/s')
        print('Active width: ',active_width)
        print('Reynolds di grano: ',Rp)
        print('Shields: ',teta)
        print('Shields90: ',teta90mean)

        # Append values as: run name, D [m], Q [m^3/s], Wwet/W [-]
        engelund_model_array = np.append(engelund_model_array,(water_depth, discharge, Wwet))
        if run_mode ==2:
            engelund_model_report[int(np.where(RUNS==run)[0]),:]=engelund_model_array

        # Print averaged residual slope:
        print()
        print('Averaged DEMs residual slope: ', np.average(slope_res))

    ###########################################################################
    # PLOTS
    ###########################################################################




end = time.time()
print()
print('Execution time: ', (end-start), 's')