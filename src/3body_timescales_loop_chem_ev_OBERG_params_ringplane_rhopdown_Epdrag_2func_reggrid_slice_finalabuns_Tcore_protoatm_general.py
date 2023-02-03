import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import rc
#matplotlib.use('Agg')
import argparse
import math
from math import log
import scipy.integrate
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from astropy import units

####important starting constants needed before function####

s_in_yrs = 3.154e7
G = 6.67e-11 #gravitational constant in SI
L_sun = 3.827e26 #solar luminosity in W
M_sun = 1.989e30 #solar mass in kg
m_earth = 5.972e24 #Earth mass in kg
r_earth = 6.378e6 #Earth radius in meters
#m_jup = 1.898e27 #Jup mass in kg
#r_jup = 6.9911e7 #Jup radius in meters
# m_jup = 10.0*m_earth
# r_jup = r_earth*(m_jup/m_earth)**0.27 #from Seager et al. 2007
m_in_AU = 1.496e11

k = 1.38e-23 #boltzman constant in SI
G = 6.67e-11 #gravitational constant in SI

m_H = 1.67e-27
m_H2O = 18.0*m_H
m_CO = 28.0*m_H
m_N2 = 28.0*m_H
#m_N = 14.0*m_H
#m_SO = 48.0*m_H
#m_C = 12.0*m_H
#m_HCN = 27.0*m_H
#m_C2H = 25.0*m_H
#m_CN = 26.0*m_H
#m_CS = 44.0*m_H
m_NH3 = 17.0*m_H
m_CO2 = 44.0*m_H
m_H2S = 34.0*m_H
m_CH3OH = 32.0*m_H

sigma = 5.67e-8 #SB constant in SI
rho_grain = 1000.0
rho_p = rho_grain
Ns = 1.e19 #number of binding sites per m**2 on a grain


### USER SETTINGS ###
drag_on = 1
n = 0

grain_gas_mass_ratio = 0.005 #dust:gas ratio

sun_pos_arr = np.array([-40.0])
mcore_arr = np.array([5.0])
St_arr = np.array([1.0])
T_core_array = np.array([3000.0]) 
time_setting = 10.0
slice_setting = 1
kappa = 0.0125 #opacity in m**2/kg
r_planet_start_ring = 2.0 #radius of the ring, needs to match for dynamical and thermal evolution files that should have already been generated

for i_mcore in range(len(mcore_arr)):
	mcore_i = mcore_arr[i_mcore]


	for i_sun_pos in range(len(sun_pos_arr)):
		sun_x_pos = sun_pos_arr[i_sun_pos]

		for i_r_p in range(len(St_arr)):
			#section used to calculate the particle radius resulting from the pre-set Stokes number of the solid
			St = St_arr[i_r_p]
			a_i = r_planet_start_ring + abs(sun_x_pos)
			a_m_i = a_i*m_in_AU
			v_k_i = (G*M_sun/a_m_i)**0.5
			omega_k_i = (G*M_sun/a_m_i**3.0)**0.5

			T_back_i = 140.0*(a_i/2.0)**(-0.65) ##from Oberg and Wordsworth 2019
			c_s_i = ((k*T_back_i)/(2.3*m_H))**0.5
			sigma_back_i = 1.5e4*(a_i)**(-3.0/2.0) #surface density, in kg/m**2
			scale_height_m_i = ((k*T_back_i/(2.3*m_H))**0.5)/((G*M_sun/(a_m_i**3.0))**0.5)

			rho_g_i = sigma_back_i/((2.0*math.pi)**0.5*scale_height_m_i)

			dP_dr_i = -(11.0/4.0)*c_s_i**2.0*rho_g_i/a_m_i
			v_th_i = (8.0/math.pi)**0.5*c_s_i
			delta_v_i = abs((a_m_i*((G*M_sun/a_m_i**2.0)+(dP_dr_i/rho_g_i)))**0.5 - (G*M_sun/a_m_i)**0.5) #

			Cd_i = (8.0/3.0)*v_th_i/delta_v_i #Cd for Epstein regime

			r_p_i = Cd_i*delta_v_i*(rho_g_i/rho_p)*(3.0/8.0)/omega_k_i #particle radius
			#########################################################
				
			for i_px_pos in range(0, 360):
				###Loop over particle ring location, calculating x and y locations from ring locations ##
				angle = float(i_px_pos)
				angle_rad = math.radians(angle)
				ring_rad = r_planet_start_ring + abs(sun_x_pos)
				px_pos = ring_rad*math.cos(angle_rad) - abs(sun_x_pos)
				py_pos = ring_rad*math.sin(angle_rad)
				#####################################
					
				for i_Tcore in range(len(T_core_array)):
					Tcore_i = T_core_array[i_Tcore]

					filename = 'dyn_3body_run_'+str(time_setting)+'_yrs_'+str(angle)+'_degrees_'+str(abs(sun_x_pos))+'_AU_Jupdist_'+str(mcore_i)+'_mearth_core_'+str(St)+'_Stokes_'+str(kappa)+'_kappaSI_'+str(Tcore_i)+'K_OBERG_'+str(ring_rad)+'_ringplane_rhopdown_Epdrag_2func_reggrid_slice'+str(slice_setting)+'_protoatm_general.npz'

					stripped_filename = filename.strip('_protoatm_general.npz')

					#figure filenames for saved .png images
					save_filename = stripped_filename + 't_chemical_evo_linear_finalabun_protoatm_general.png'
					save_filename_2 = stripped_filename + 't_chemical_evo_nonzoom_linear_finalabun_protoatm_general.png'

					#filename for .npz file that is saved for each particle
					save_txt_filename = stripped_filename + '_chemicalev_finalabun_protoatm_general.npz'

					############# DATA LOADING SECTION ####################
					data_array_start = np.load(filename)
					data_array = data_array_start['arr_0']

					t = data_array[:,0]
					t_secs = t*s_in_yrs

					r_p_s_AU = data_array[:,1]
					r_p_j_AU = data_array[:,2]
					rho_back_gas = data_array[:,3]

					T_background = data_array[:,4]

					print('T_background', T_background)

					T_at_R = data_array[:,5]
					print('T at R', T_at_R)
					print('Rps', r_p_s_AU)

					non_rot_x_p_rel_m = data_array[:,6]
					non_rot_y_p_rel_m = data_array[:,7]
					full_position_x_array_m = data_array[:,8]
					full_position_y_array_m = data_array[:,9]
						
					r_p_s = r_p_s_AU*m_in_AU
					r_p_j = r_p_j_AU*m_in_AU

					endtime = t[-1]

					text_bits = filename.split('_')
					start_p_x = px_pos
					start_p_y = py_pos
					solar_pos_str = text_bits[7]
					solar_pos = float(solar_pos_str)
					mcore_size = text_bits[10]
					M_core_num = float(mcore_size)

					St_str = text_bits[13]
					St_num = St
					particle_size = str(r_p_i)
					r_p = r_p_i
					kappa = float(text_bits[15])
					Tcore = Tcore_i
					M_core = M_core_num*m_earth #loose assumption where I just chose 10 Mearth for the core mass
					R_core = r_earth*(M_core/m_earth)**0.27
					m_jup = M_core
					r_jup = R_core
					r_jup_AU_approx = 7.9398e-05

					accretion_time_interp = interp1d(r_p_j_AU, t, kind='linear')
					try:
						time_of_accretion = accretion_time_interp(r_jup_AU_approx)
						print('The particle is accreted at ', time_of_accretion, ' years')
					except:
						time_of_accretion = endtime
						print('The particle is never accreted')

					epsilon = m_jup/(M_sun + m_jup)

					M_com = (M_sun*m_jup)/(M_sun + m_jup)

					particle_position_vector_AU = np.array([start_p_x, start_p_y, 0.0])
					solar_position_vector_AU = np.array([-solar_pos, 0.0, 0.0])
					sun_x_AU = solar_position_vector_AU[0]
					sun_y_AU = solar_position_vector_AU[1]
					sun_z_AU = solar_position_vector_AU[2]
					jup_position_vector_AU = np.array([0.0, 0.0, 0.0])

					#### starting chemical concentrations ####
					ML_calc = (Ns*grain_gas_mass_ratio*m_H*3.0*1.15)/(rho_grain*r_p)

					m_grain = (4.0/3.0)*math.pi*r_p**3.0*rho_grain #mass of the tracked solid

					#### START OF CHEMICAL SECTION #####
						
					Mg_H = m_grain/grain_gas_mass_ratio

					Mg_H2O = 1.6e-4*Mg_H
					Mg_CO = 8.e-5*Mg_H
					Mg_N2 = 3.e-5*Mg_H
					Mg_H2S = 1.91e-8*Mg_H
					Mg_CH3OH = 1.e-6*Mg_H
					Mg_NH3 = 7.e-6*Mg_H
					Mg_CO2 = 4.e-5*Mg_H


					sigma_H_start = Mg_H/(4.0*math.pi*r_p**2.0)
					sigma_H2O_grain_start = Mg_H2O/(4.0*math.pi*r_p**2.0)
					sigma_CO_grain_start = Mg_CO/(4.0*math.pi*r_p**2.0)
					sigma_N2_grain_start = Mg_N2/(4.0*math.pi*r_p**2.0)
					sigma_H2S_grain_start = Mg_H2S/(4.0*math.pi*r_p**2.0)
					sigma_CH3OH_grain_start = Mg_CH3OH/(4.0*math.pi*r_p**2.0)
					sigma_NH3_grain_start = Mg_NH3/(4.0*math.pi*r_p**2.0)
					sigma_CO2_grain_start = Mg_CO2/(4.0*math.pi*r_p**2.0)
						

					sigma_H2O_grain_array = np.array([sigma_H2O_grain_start])
					sigma_CO_grain_array = np.array([sigma_CO_grain_start])
					sigma_N2_grain_array = np.array([sigma_N2_grain_start])
					sigma_H2S_grain_array = np.array([sigma_H2S_grain_start])
					sigma_CH3OH_grain_array = np.array([sigma_CH3OH_grain_start])
					sigma_NH3_grain_array = np.array([sigma_NH3_grain_start])
					sigma_CO2_grain_array = np.array([sigma_CO2_grain_start])

						
					Ei_H2O = 5800.0
					Ei_CO = 1180.0
					Ei_N2 = 1050.0
					Ei_H2S = 2743.0
					Ei_CH3OH = 4930.0
					Ei_NH3 = 3800.0
					Ei_CO2 = 2700.0


					nu_i_H2O = 1.6e11*(Ei_H2O/18.0)**0.5
					nu_i_CO = 1.6e11*(Ei_CO/28.0)**0.5
					nu_i_N2 = 1.6e11*(Ei_N2/28.0)**0.5
					nu_i_H2S = 1.6e11*(Ei_H2S/34.0)**0.5
					nu_i_CH3OH = 1.6e11*(Ei_CH3OH/32.0)**0.5  
					nu_i_NH3 = 1.6e11*(Ei_NH3/17.0)**0.5
					nu_i_CO2 = 1.6e11*(Ei_CO2/44.0)**0.5

					
					#loop through each timestep to calculate the change in chemical composition and resulting composition at each timestep
					for i_chem in range(len(t)-1):
						dt_years = t[i_chem+1] - t[i_chem]
						dt = dt_years*s_in_yrs

						sigma_H2O_grain_i = sigma_H2O_grain_array[i_chem]
						sigma_CO_grain_i = sigma_CO_grain_array[i_chem]
						sigma_N2_grain_i = sigma_N2_grain_array[i_chem]
						sigma_H2S_grain_i = sigma_H2S_grain_array[i_chem]
						sigma_CH3OH_grain_i = sigma_CH3OH_grain_array[i_chem]
						sigma_NH3_grain_i = sigma_NH3_grain_array[i_chem]
						sigma_CO2_grain_i = sigma_CO2_grain_array[i_chem]


						T_r_i = T_at_R[i_chem]
						rho_gas_i = rho_back_gas[i_chem]
						r_p_s_i_AU = r_p_s_AU[i_chem]
						r_p_j_i_AU = r_p_j_AU[i_chem]


						k_evap_H2O = nu_i_H2O*math.exp(-Ei_H2O/T_r_i) 
						k_evap_CO = nu_i_CO*math.exp(-Ei_CO/T_r_i)
						k_evap_N2 = nu_i_N2*math.exp(-Ei_N2/T_r_i)
						k_evap_H2S = nu_i_H2S*math.exp(-Ei_H2S/T_r_i)
						k_evap_CH3OH = nu_i_CH3OH*math.exp(-Ei_CH3OH/T_r_i)
						k_evap_NH3 = nu_i_NH3*math.exp(-Ei_NH3/T_r_i)
						k_evap_CO2 = nu_i_CO2*math.exp(-Ei_CO2/T_r_i)


						sigma_H2O_grain_new = sigma_H2O_grain_i - Ns*k_evap_H2O*m_H2O*dt 
						sigma_CO_grain_new = sigma_CO_grain_i - Ns*k_evap_CO*m_CO*dt
						sigma_N2_grain_new = sigma_N2_grain_i - Ns*k_evap_N2*m_N2*dt
						sigma_H2S_grain_new = sigma_H2S_grain_i - Ns*k_evap_H2S*m_H2S*dt
						sigma_CH3OH_grain_new = sigma_CH3OH_grain_i - Ns*k_evap_CH3OH*m_CH3OH*dt
						sigma_NH3_grain_new = sigma_NH3_grain_i - Ns*k_evap_NH3*m_NH3*dt
						sigma_CO2_grain_new = sigma_CO2_grain_i - Ns*k_evap_CO2*m_CO2*dt

						if sigma_H2O_grain_i <= 0.0:
							sigma_H2O_grain_new = 0.0 
						if sigma_CO_grain_i <= 0.0:
							sigma_CO_grain_new = 0.0
						if sigma_N2_grain_i <= 0.0:
							sigma_N2_grain_new = 0.0
						if sigma_H2S_grain_i <= 0.0:
							sigma_H2S_grain_new = 0.0
						if sigma_CH3OH_grain_i <= 0.0:
							sigma_CH3OH_grain_new = 0.0
						if sigma_NH3_grain_i <= 0.0:
							sigma_NH3_grain_new = 0.0
						if sigma_CO2_grain_i <= 0.0:
							sigma_CO2_grain_new = 0.0

						if sigma_H2O_grain_new <= 0.0:
							sigma_H2O_grain_new = 0.0 
						if sigma_CO_grain_new <= 0.0:
							sigma_CO_grain_new = 0.0
						if sigma_N2_grain_new <= 0.0:
							sigma_N2_grain_new = 0.0
						if sigma_H2S_grain_new <= 0.0:
							sigma_H2S_grain_new = 0.0
						if sigma_CH3OH_grain_new <= 0.0:
							sigma_CH3OH_grain_new = 0.0
						if sigma_NH3_grain_new <= 0.0:
							sigma_NH3_grain_new = 0.0
						if sigma_CO2_grain_new <= 0.0:
							sigma_CO2_grain_new = 0.0
	
						sigma_H2O_grain_array = np.append(sigma_H2O_grain_array, sigma_H2O_grain_new)
						sigma_CO_grain_array = np.append(sigma_CO_grain_array, sigma_CO_grain_new)
						sigma_N2_grain_array = np.append(sigma_N2_grain_array, sigma_N2_grain_new)
						sigma_H2S_grain_array = np.append(sigma_H2S_grain_array, sigma_H2S_grain_new)
						sigma_CH3OH_grain_array = np.append(sigma_CH3OH_grain_array, sigma_CH3OH_grain_new)
						sigma_NH3_grain_array = np.append(sigma_NH3_grain_array, sigma_NH3_grain_new)
						sigma_CO2_grain_array = np.append(sigma_CO2_grain_array, sigma_CO2_grain_new)
							
					sigma_H2O_grain_array = sigma_H2O_grain_array/sigma_H_start
					sigma_CO_grain_array = sigma_CO_grain_array/sigma_H_start
					sigma_N2_grain_array = sigma_N2_grain_array/sigma_H_start
					sigma_H2S_grain_array = sigma_H2S_grain_array/sigma_H_start
					sigma_CH3OH_grain_array = sigma_CH3OH_grain_array/sigma_H_start
					sigma_NH3_grain_array = sigma_NH3_grain_array/sigma_H_start
					sigma_CO2_grain_array = sigma_CO2_grain_array/sigma_H_start
						
					sigma_H2O_grain_nozero_array = np.where(sigma_H2O_grain_array<=1.e-25, 1.e-25, sigma_H2O_grain_array) #here, we define a molecular abundance of 1.e-25 as our "zero" threshold so that we can graph abundances on a log scale
					sigma_CO_grain_nozero_array = np.where(sigma_CO_grain_array<=1.e-25, 1.e-25, sigma_CO_grain_array)
					sigma_N2_grain_nozero_array = np.where(sigma_N2_grain_array<=1.e-25, 1.e-25, sigma_N2_grain_array)
					sigma_H2S_grain_nozero_array = np.where(sigma_H2S_grain_array<=1.e-25, 1.e-25, sigma_H2S_grain_array)
					sigma_CH3OH_grain_nozero_array = np.where(sigma_CH3OH_grain_array<=1.e-25, 1.e-25, sigma_CH3OH_grain_array)
					sigma_NH3_grain_nozero_array = np.where(sigma_NH3_grain_array<=1.e-25, 1.e-25, sigma_NH3_grain_array)
					sigma_CO2_grain_nozero_array = np.where(sigma_CO2_grain_array<=1.e-25, 1.e-25, sigma_CO2_grain_array)
						
					plt.figure(figsize=(14.0,10.0))
					plt.rc('font', family='Times New Roman')
					plt.tick_params(length=8, width=2.5, labelsize=24)
					plt.tick_params(which='minor', length=5, width=1.5, labelsize=24)
					plt.xlabel('Time [yrs]', fontsize = 32)
					plt.ylabel('$\\frac{n_{x}}{n_{H}}$', fontsize = 32)
					plt.yscale('log')
					plt.ylim(1.e-9, 1.e-3)
					plt.plot(t, sigma_H2O_grain_nozero_array, linewidth=3, c='c', label='Grain H$_{2}$O Abundance')
					plt.plot(t, sigma_CO_grain_nozero_array, linewidth=3, c='b', label='Grain CO Abundance')
					plt.plot(t, sigma_N2_grain_nozero_array, linewidth=3, c='r', label='Grain N$_{2}$ Abundance')
					
					plt.scatter(t, sigma_H2O_grain_nozero_array, c='c', marker='o')
					plt.scatter(t, sigma_CO_grain_nozero_array, c='b', marker='o')
					plt.scatter(t, sigma_N2_grain_nozero_array, c='r', marker='o')
					plt.legend(loc='best', fancybox=True, framealpha=0.5, prop={'size': 18})
					#plt.savefig(save_filename, bbox_inches='tight') ##uncomment command out if you want to save this figure for each simulation!
					plt.close()


					plt.figure(figsize=(14.0,10.0))
					plt.rc('font', family='Times New Roman')
					plt.tick_params(length=8, width=2.5, labelsize=24)
					plt.tick_params(which='minor', length=5, width=1.5, labelsize=24)
					plt.xlabel('Time [yrs]', fontsize = 32)
					plt.ylabel('$\\frac{n_{x}}{n_{H}}$', fontsize = 32)
					plt.yscale('log')
					plt.plot(t, sigma_H2O_grain_nozero_array, linewidth=3, c='c', label='Grain H$_{2}$O Abundance')
					plt.plot(t, sigma_CO_grain_nozero_array, linewidth=3, c='b', label='Grain CO Abundance')
					plt.plot(t, sigma_N2_grain_nozero_array, linewidth=3, c='r', label='Grain N$_{2}$ Abundance')
					plt.scatter(t, sigma_H2O_grain_nozero_array, c='c', marker='o')
					plt.scatter(t, sigma_CO_grain_nozero_array, c='b', marker='o')
					plt.scatter(t, sigma_N2_grain_nozero_array, c='r', marker='o')
					plt.legend(loc='best', fancybox=True, framealpha=0.5, prop={'size': 18})
					#plt.savefig(save_filename_2, bbox_inches='tight') ##uncomment command out if you want to save this figure for each simulation!
					plt.close()

					stack = np.column_stack((t, r_p_s_AU))
					stack_1 = np.column_stack((stack, r_p_j_AU))
					stack_2 = np.column_stack((stack_1, T_background))
					stack_3 = np.column_stack((stack_2, T_at_R))
					stack_4 = np.column_stack((stack_3, sigma_N2_grain_nozero_array))
					stack_5 = np.column_stack((stack_4, sigma_CO_grain_nozero_array))
					stack_6 = np.column_stack((stack_5, sigma_H2O_grain_nozero_array))
					stack_7 = np.column_stack((stack_6, non_rot_x_p_rel_m))
					stack_8 = np.column_stack((stack_7, non_rot_y_p_rel_m))
					stack_9 = np.column_stack((stack_8, full_position_x_array_m))
					stack_10 = np.column_stack((stack_9, full_position_y_array_m))
					stack_11 = np.column_stack((stack_10, sigma_H2S_grain_nozero_array))
					stack_12 = np.column_stack((stack_11, sigma_CH3OH_grain_nozero_array))
					stack_13 = np.column_stack((stack_12, sigma_NH3_grain_nozero_array))
					stack_14 = np.column_stack((stack_13, sigma_CO2_grain_nozero_array))
					
					np.savez_compressed(save_txt_filename, stack_14)








					


	
