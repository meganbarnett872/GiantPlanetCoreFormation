
## The purpose of this code is to create a fine-mesh temperature grid within 3 Hill Radius of the protoplanet core.
## While I calculate the equivalent blackbody temperature of solids from the background irradiation outside 3R_H, inside 3R_H the opacity of the surrounding material between the core and solid can be important.
## So, I form an equal area grid around the core, calculate the distance from each coordinate to the core, and calculate the optical depth of material between the core and solid.
## That optical depth is then used to calculate an attenuated flux that reaches the solid.
## Finally, I add the attenuated flux received by the solid from the core to the flux from the background, and convert to temperature at each grid location.
## NOTE: TEMPERATURE IS THE BLACKBODY TEMPERATURE OF THE SOLID, NOT THE TEMPERATURE EACH LOCATION FEELS!! This is different by a factor of like sqrt(2) or something (see equation below)

##### Package import section ######

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.colors
import argparse
import math
from math import log
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.optimize import brentq

##### ALL CALCULATIONS IN THIS CODE ARE IN SI #####

## CONSTANTS SECTION ##

M_sun = 1.989e30 #solar mass in kg
R_sun = 6.9634e8 #solar radius in meters
L_sun = 3.827e26 #solar luminosity in W
m_earth = 5.972e24 #Earth mass in kg
r_earth = 6.378e6 #Earth radius in meters
r_jup = 6.9911e7 #Jupiter radius in meters

k = 1.38e-23 #boltzman constant in SI
G = 6.67e-11 #gravitational constant in SI
m_H = 1.67e-27
sigma = 5.67e-8 #SB constant in SI
s_in_yrs = 3.154e7
m_in_AU = 1.496e11
kappa = 0.0125 #opacity in m**2/kg

## USER SETTINGS SECTION ##
M_core_num = 1.0

a_array = np.array([40.0])
T_core_array = np.array([4000.0])

T_H2O_fo = 150.0 # rough estimate of freeze out temperature of H2O
T_CO_fo = 20.0 # rough rough estimate of freeze out temperature of CO
T_N2_fo = 8.0 # rough estimate of freeze out temperature of N2

degree_array = np.arange(0.0, 360.0, 1.0) ##only used for R_H plot line
rad_array = np.radians(degree_array) ##only used for R_H plot line

############## ~~~~~~ FOR-LOOPING SO I DON'T HAVE TO RUN THIS CODE FOR EACH INDIVIDUAL CASE ~~~~~~~ ###############
M_core = M_core_num*m_earth
R_core = r_earth*(M_core/m_earth)**0.27 #From Valencia et al. 2006

for i_as in range(len(a_array)):
	a_i = a_array[i_as]
	r_H = a_i*(M_core/(3.0*M_sun))**(1.0/3.0)
	r_H_m = r_H*m_in_AU
	r_quarter_H = r_H/4.0
	r_quarter_H_m = r_H_m/4.0
	print('Hill Radius', r_H)
	print('1/4 Hill Radius', r_quarter_H)

	r_H_line_x = r_H*np.cos(rad_array) + a_i ##R_H plot line, x coord
	r_H_line_y = r_H*np.sin(rad_array) ##R_H plot line, y coord

	Jup_loc = np.array([a_i, 0.0])
	x_Jup = Jup_loc[0]
	y_Jup = Jup_loc[1]

	a_m = a_i*m_in_AU

	x_vals = np.arange(x_Jup-3.0*r_H, x_Jup+3.0*r_H+0.005, 0.005)
	y_vals = np.arange(y_Jup-3.0*r_H, y_Jup+3.0*r_H+0.005, 0.005)

	for i in range(len(T_core_array)):

		plt.figure(1, figsize=(14.0, 10.0))
		plt.rc('font', family='Times New Roman')
		plt.tick_params(length=8, width=2.5, labelsize=24)
		plt.tick_params(which='minor', length=5, width=1.5, labelsize=24)
		plt.xlabel('X Position [w.r.t. Solar Position]', fontsize = 32)
		plt.ylabel('Y Position [w.r.t. Solar Position]', fontsize = 32)

		T_core = T_core_array[i]

		############## ~~~~~~ CALCULATIONS BEGIN HERE ~~~~~~~ ###############
		print(T_core)
		L_core = 4.0*math.pi*R_core**2.0*sigma*T_core**4.0
		T_background_at_core = 140.0*(a_i/2.0)**(-0.65)
		F_background_at_core = sigma*T_background_at_core**4.0
		T_protoatm = (L_core/(16.0*math.pi*r_quarter_H_m**2.0*sigma) + F_background_at_core/sigma)**0.25
		print(T_protoatm)
		x_full_arr = np.array([])
		y_full_arr = np.array([])
		temp_full_arr = np.array([])

		for i_x in range(len(x_vals)):
			x_i = x_vals[i_x]
			for i_y in range(len(y_vals)):
				y_i = y_vals[i_y]

				r_p_sun_AU = (x_i**2.0 + y_i**2.0)**0.5
				r_p_sun = r_p_sun_AU*m_in_AU

				r_p_jup_AU = ((x_Jup - x_i)**2.0 + (y_Jup - y_i)**2.0)**0.5
				r_p_jup = r_p_jup_AU*m_in_AU

				r_p_quarter_rH_AU = r_p_jup_AU - r_quarter_H
				r_p_quarter_rH = r_p_jup - r_quarter_H_m

				T_back = 140.0*(r_p_sun_AU/2.0)**(-0.65) ##from Oberg and Wordsworth 2019
				F_back = sigma*T_back**4.0

				m_eq = (y_Jup - y_i)/(x_Jup - x_i)
				b_eq = y_Jup - m_eq*x_Jup
				if r_p_jup <= r_quarter_H_m:
					tau_array = 0.0
				else:
					if x_Jup < x_i:
						x_eq_array = np.linspace(x_Jup, x_i, 10000)
						y_eq_array = m_eq*x_eq_array + b_eq
						r_eq_array = (x_eq_array**2.0 + y_eq_array**2.0)**0.5
						r_p_j_eq_array = ((x_Jup - x_eq_array)**2.0 + (y_Jup - y_eq_array)**2.0)**0.5
						r_eq_protoatm_array = r_eq_array[r_p_j_eq_array>r_p_quarter_rH_AU]
						chunk_num = float(len(r_eq_protoatm_array))
					elif x_i < x_Jup:
						x_eq_array = np.linspace(x_i, x_Jup, 10000)
						y_eq_array = m_eq*x_eq_array + b_eq
						r_eq_array = (x_eq_array**2.0 + y_eq_array**2.0)**0.5
						r_p_j_eq_array = ((x_Jup - x_eq_array)**2.0 + (y_Jup - y_eq_array)**2.0)**0.5
						r_eq_protoatm_array = r_eq_array[r_p_j_eq_array>r_p_quarter_rH_AU]
						chunk_num = float(len(r_eq_protoatm_array))
					elif x_i == x_Jup and y_Jup < y_i:
						y_eq_array = np.linspace(y_i, y_Jup, 10000)
						x_eq_array = (y_eq_array - b_eq)/m_eq
						r_eq_array = (x_eq_array**2.0 + y_eq_array**2.0)**0.5
						r_p_j_eq_array = ((x_Jup - x_eq_array)**2.0 + (y_Jup - y_eq_array)**2.0)**0.5
						r_eq_protoatm_array = r_eq_array[r_p_j_eq_array>r_p_quarter_rH_AU]
						chunk_num = float(len(r_eq_protoatm_array))
					else:
						y_eq_array = np.linspace(y_i, y_Jup, 10000)
						x_eq_array = (y_eq_array - b_eq)/m_eq
						r_eq_array = (x_eq_array**2.0 + y_eq_array**2.0)**0.5
						r_p_j_eq_array = ((x_Jup - x_eq_array)**2.0 + (y_Jup - y_eq_array)**2.0)**0.5
						r_eq_protoatm_array = r_eq_array[r_p_j_eq_array>r_p_quarter_rH_AU]
						chunk_num = float(len(r_eq_protoatm_array))

					r_eq_protoatm_array_m = r_eq_protoatm_array*m_in_AU
					T_back_array = 140.0*(r_eq_protoatm_array/2.0)**(-0.65) ##from Oberg and Wordsworth 2019
					scale_height_m = ((k*T_back_array/(2.3*m_H))**0.5)/((G*M_sun/(r_eq_protoatm_array_m**3.0))**0.5)
					sigma_back = 1.5e4*(r_eq_protoatm_array)**(-3.0/2.0) #surface density, in kg/m**2 ##from Oberg and Wordsworth 2019

					rho_back = sigma_back/((2.0*math.pi)**0.5*scale_height_m)

					rho_tot = np.sum(rho_back)
					tau_array = kappa*((r_p_jup-r_quarter_H_m)/chunk_num)*rho_tot

				F_at_core = sigma*T_core**4.0
				L_core = F_at_core*4.0*math.pi*R_core**2.0
				F_core = L_core/(4.0*math.pi*r_p_jup**2.0)
				T_at_r = (((np.exp(-tau_array)/4.0)*(R_core/r_p_jup)**2.0*T_core**4.0) + (F_back/sigma))**0.25

				if r_p_jup < r_quarter_H_m:
					T_at_r = T_protoatm
				if r_p_sun < R_sun:
					T_at_r = 5800.0

				x_full_arr = np.append(x_full_arr, x_i)
				y_full_arr = np.append(y_full_arr, y_i)
				temp_full_arr = np.append(temp_full_arr, T_at_r)


		x_len = len(x_vals)
		y_len = len(y_vals)

		x_contour_array = x_full_arr.reshape(x_len, y_len)
		y_contour_array = y_full_arr.reshape(x_len, y_len)
		temp_contour_array = temp_full_arr.reshape(x_len, y_len)

		save_name = 'temp_contour_' + str(a_i) + 'AU_' + str(M_core_num) + 'Mearth_' +str(kappa) + 'kappa_' + str(T_core) + 'K_zoomed_3Rhill_highres2_contour_taufix_Tfix_protoatm.png'
		save_contour_name = 'temp_contour_' + str(a_i) + 'AU_' + str(M_core_num) + 'Mearth_' +str(kappa) + 'kappa_' + str(T_core) + 'K_zoomed_3Rhill_highres2_contour_taufix_Tfix_protoatm.npz'
		plt.contourf(x_contour_array, y_contour_array, temp_contour_array, cmap='plasma', norm=matplotlib.colors.LogNorm(), levels=np.array([5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 125.0, 150.0, 175.0, 200.0]))
		cbar = plt.colorbar(ticks=np.array([5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0, 100.0, 125.0, 150.0, 175.0, 200.0]), format='%.1f', shrink=1.0, fraction=0.1, pad=0.01)
		cbar.set_label('Temperature [K]', rotation=270, fontsize=28)
		cbar.ax.tick_params(length=6, width=1.75, labelsize=18)
		cbar.ax.tick_params(which='minor', length=5, width=1.5)
		cbar.ax.get_yaxis().labelpad = 50
		plt.plot(r_H_line_x, r_H_line_y, linewidth=3, linestyle='--', c='k')
		plt.savefig(save_name)
		cbar.remove()
		plt.clf()

##### SAVING THE GENERATED TEMPERATURE PROFILE TO NUMPY COMPRESSED FILE ######
		save_array = np.array([temp_contour_array, x_contour_array, y_contour_array])
		np.savez_compressed(save_contour_name, temp_contour_array, x_contour_array, y_contour_array)




