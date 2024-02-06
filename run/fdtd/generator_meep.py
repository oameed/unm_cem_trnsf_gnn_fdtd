#####################################################################################################
### UNIVERSITY OF NEW MEXICO                                                                      ###
### COMPUTATIONAL EM LAB                                                                          ###
### EMULATING FDTD USING TRANSFORMERS AND GNNS                                                    ###
### A TYPICAL MEEP SIMULATION AS BENCHMARK FOR TIMING STUDIES                                     ###
### by: OAMEED NOAKOASTEEN                                                                        ###
###                                                                                               ###
### useful references for getting started with MEEP:                                              ###
### (1) MEEP Documentation:                      https://meep.readthedocs.io/en/latest/           ###
### (2) Matthew Baas, Getting started with MEEP: https://rf5.github.io/2019/12/22/meep-intro.html ###
#####################################################################################################

import numpy as np
import meep  as mp
import time

###########################################################
### SIMULATION CELL SIZE AND DURATION CALCULATIONS      ###
###########################################################

eps0       = 8.85419e-12                        # FREE SPACE PERMITTIVITY
mu0        = 4*np.pi*1e-7                       # FREE SPACE PERMEABILITY   
c0         = 1/np.sqrt(mu0*eps0)                # SPEED OF LIGHT IN FREE SPACE

L          = 1.5                                # LENGTH OF COMPUTATIONAL DOMAIN (m)
Fmax       = 2e9                                # LARGEST FREQUENCY IN THE PULSE (Hz)
Ncpw       = 40                                 # NUMBER OF CELLS PER WAVELENGTH
CSF        = 1/np.sqrt(2)                       # COURANT STABILITY FACTOR

Lm_min     = c0/Fmax                            # SMALLEST LAMBDA IN THE PULSE
delta      = Lm_min/Ncpw                        # LENGTH OF EACH CELL 
delta_t    = delta*CSF/c0                       # DURATION OF EACH TIMESTEP
Nc         = np.floor(L/delta)                  # TOTAL NUMBER OF CELLS ALONG EACH AXIS
Nt         = 400                                # TOTAL DURATION OF SIMULATION (PER TIMESTEP)

SRCLF      = 0.6                                # SOURCE LOCATION FACTOR
PMLF       = 0.25                               # FRACTION OF THE LENGTH OF THE CD ALLOCATED TO PML

taw        = np.sqrt(-np.log(0.1))/(np.pi*Fmax) # GAUSSIAN PULSE PARAMETER
t0         = np.sqrt(20)/taw                    # GAUSSIAN PULSE PARAMETER

###########################################################
### SIMULATION SETUP                                    ###
### Meep's unit of time (i.e., "b") is set equal to     ###
### the duratoin of each timestep (i.e., "delta_t").    ###
###########################################################

a          = delta                              # MEEP'S UNIT OF LENGTH 
b          = a/c0                               # MEEP'S UNIT OF TIME
resolution = 1
duration   = np.floor(Nt*delta_t/b)

cell_num   = np.floor(L/a)
cell       = mp.Vector3(cell_num, cell_num, 0)

src_f      = t0
src_w      = taw/np.sqrt(2)
src_lx     = cell_num/2+np.floor(SRCLF*L/a)
src_ly     = cell_num/2+np.floor(SRCLF*L/a)

sources    = [mp.Source(src      =mp.ContinuousSource(Fmax*(a/c0)), # mp.GaussianSource(frequency=src_f*(a/c0),width=src_w/b)
                        center   =mp.Vector3(50,50,0)             ,
                        component=mp.Hz                           ,
                        amplitude=1                                )]

radius     = (1*(c0/Fmax))/2
geometry   = [mp.Cylinder(radius  =radius/a         ,
                          axis    =mp.Vector3(0,0,1),
                          height  =0                ,
                          material=mp.metal          )]

pml_d      = np.floor(PMLF*L/a)
pml_layers = [mp.PML(pml_d)]

simulation =  mp.Simulation(resolution     = resolution    ,
                            cell_size      = cell          ,
                            sources        = sources       ,
                            geometry       = geometry      ,
                            boundary_layers= pml_layers    ,  
                            Courant        = CSF            )

time_start = time.time()

simulation.run(mp.at_beginning(mp.output_epsilon)                               ,
               mp.to_appended ("ex", mp.at_every(delta_t/b, mp.output_efield_x)),
               until=duration                                                    )

time_end = time.time()
run_time = (time_end-time_start)/duration
print(" RUN TIME FOR EACH TIME STEP IS {:1.2f} (ms)".format(run_time/1e-3))

print(" SIMULATION FINISHED! ")


