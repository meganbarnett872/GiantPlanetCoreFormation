## The purpose of this code is to calculate the thermal evolution of solids from previously generated dynamical evolution files as the particles radially drift inwards past a forming giant planet core.
## Each .npz file stores the thermal evolution information of ONE particle, and particles are evolved independently from each other, one after the other (hence the for-loop for particle position around the ring)
## In the resultling .npz file, we store the time, r, x and y position, and x and y velocities, and the temperature felt by the solid at each timestep, as well as the background disk temperature and density at each location the particle is at


import numpy as np
import h5py
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import argparse
import math
from math import log
import scipy.integrate
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from astropy import units

####important constant definitions####

s_in_yrs = 3.154e7 #number of seconds in a year
G = 6.67e-11 #gravitational constant in SI
L_sun = 3.827e26 #solar luminosity in W
M_sun = 1.989e30 #solar mass in kg
m_earth = 5.972e24 #Earth mass in kg
r_earth = 6.378e6 #Earth radius in meters
m_in_AU = 1.496e11 #number of meters in an AU
rho_p = 1000.0 #internal solid density for predominantly icy solid
k = 1.38e-23 #boltzman constant in SI
G = 6.67e-11 #gravitational constant in SI
m_H = 1.67e-27 #mass of hydrogen atom in SI
sigma = 5.67e-8 #SB constant in SI
####################################


### USER SETTINGS ###
drag_on = 1 #flag to turn on drag, drag is on if = 1
n = 0

sun_pos_arr = np.array([-40.0])
mcore_arr = np.array([5.0])
St_arr = np.array([1.0])

T_core = 3000.0 #effective core temperature setting for calc

time_setting = 10.0 #total evolution time from dynamical saved models
slice_setting = 1 #determines whether I'm saving all the generated data or not, when slice_setting is 1 then all data is being saved
r_planet_start_ring = 2.0 #radius of the ring, needs to match for dynamical evolution files that should have already been generated
kappa = 0.0125 #opacity in m**2/kg, can be varied to test effect of opacity on temperature profile

#function to calculate background temperature at each r location
def temp_func_1(r_AU):
	r_p_s_i = r_AU
	T_vals = 140.0*(r_p_s_i/2.0)**(-0.65) ##from Oberg and Wordsworth 2019
	return T_vals

#function that 
def temp_func_2(x_par_arr, y_par_arr, Jup_x_pos, Jup_y_pos, r_H, cont_file):
	cont_array_start = np.load(cont_file)
	x_val_array = np.arange(Jup_x_pos-3.0*r_H, Jup_x_pos+3.0*r_H+0.005, 0.005)
	y_val_array = np.arange(Jup_y_pos-3.0*r_H, Jup_y_pos+3.0*r_H+0.005, 0.005)
	temp_array = cont_array_start['arr_0']

	interp_temp = RegularGridInterpolator((x_val_array, y_val_array), temp_array, method='linear') 
	T_vals = interp_temp((x_par_arr, y_par_arr))
	print('----------------------------------------------------------------------------------------')
	return T_vals

for i_mcore in range(len(mcore_arr)):
	mcore_i = mcore_arr[i_mcore]

	for i_sun_pos in range(len(sun_pos_arr)):
		sun_x_pos = sun_pos_arr[i_sun_pos]

		for i_r_p in range(len(St_arr)):
			St = St_arr[i_r_p]

			for i_px_pos in range(0,360):
				angle = float(i_px_pos)
				#print(angle)
				angle_rad = math.radians(angle)
				ring_rad = r_planet_start_ring + abs(sun_x_pos)
				px_pos = ring_rad*math.cos(angle_rad) - abs(sun_x_pos)
				py_pos = ring_rad*math.sin(angle_rad)

				#St = St_arr[i_r_p]
				a_i = ring_rad
				a_m_i = a_i*m_in_AU
				v_k_i = (G*M_sun/a_m_i)**0.5
				omega_k_i = (G*M_sun/a_m_i**3.0)**0.5

				T_back_i = 140.0*(a_i/2.0)**(-0.65) ##from Oberg and Wordsworth 2019
				c_s_i = ((k*T_back_i)/(2.3*m_H))**0.5
				sigma_back_i = 1.5e4*(a_i)**(-3.0/2.0) #surface density, in kg/m**2
				scale_height_m_i = ((k*T_back_i/(2.3*m_H))**0.5)/((G*M_sun/(a_m_i**3.0))**0.5)

				rho_g_i = sigma_back_i/((2.0*math.pi)**0.5*scale_height_m_i)
				#print(rho_g_i)
				#bloop
				#mfp = m_g/(cross_section_H2*rho_g) #could be rho_g or rho_d
				dP_dr_i = -(11.0/4.0)*c_s_i**2.0*rho_g_i/a_m_i
				v_th_i = (8.0/math.pi)**0.5*c_s_i
				delta_v_i = abs((a_m_i*((G*M_sun/a_m_i**2.0)+(dP_dr_i/rho_g_i)))**0.5 - (G*M_sun/a_m_i)**0.5) #
				#---------------------------------------------------#
				Cd_i = (8.0/3.0)*v_th_i/delta_v_i #Cd for Epstein regime
				#---------------------------------------------------#
				r_p_i = Cd_i*delta_v_i*(rho_g_i/rho_p)*(3.0/8.0)/omega_k_i
				#r_p_i = r_p_arr[i_r_p]

				#################################################
				filename = 'dyn_3body_run_'+str(time_setting)+'_yrs_'+str(angle)+'_degrees_'+str(abs(sun_x_pos))+'_AU_Jupdist_'+str(mcore_i)+'_mearth_core_'+str(St)+'_Stokes_DRAG_OBERG_'+str(ring_rad)+'_ringplane_rhopdown_Epdrag_protoatm_general.npz'
				#filename_1 = 'dyn_3body_run_'+str(time_setting)+'_yrs_'+str(angle)+'_degrees_'+str(abs(sun_x_pos))+'_AU_Jupdist_'+str(mcore_i)+'_mearth_core_'+str(r_p_i)+'_particle_rad_DRAG_OBERG_'+str(ring_rad)+'_ringplane.npz'
				#m_dot_array = np.array([1.e-8, 1.e-9, 1.e-10, 1.e-11])
					
				#m_dot_array = np.array([1.e-11])

				############# DATA LOADING SECTION FROM SAVED DYNAMICAL MODEL FILES ####################
				data_array_start = np.load(filename)
				data_array = data_array_start['arr_0']

				t = data_array[:,0]

				t_secs = t*s_in_yrs

				full_position_x_array_m = data_array[:,1]
				full_position_y_array_m = data_array[:,2]
				full_position_z_array_m = data_array[:,3]
				#r_p_fromJup = (full_position_x_array_m**2.0 + full_position_y_array_m**2.0)**0.5

				#r_H_indices_arr = np.where(r_p_fromJup)

				full_vel_x_array = data_array[:,4]
				full_vel_y_array = data_array[:,5]
				full_vel_z_array = data_array[:,6]

				full_position_x_array = full_position_x_array_m/m_in_AU
				full_position_y_array = full_position_y_array_m/m_in_AU
				full_position_z_array = full_position_z_array_m/m_in_AU

				start_p_x = full_position_x_array[0]
				start_p_y = full_position_y_array[0]
				start_p_z = full_position_z_array[0]

				#endtime = t[-1]
				text_bits = filename.split('_')
				solar_pos_str = text_bits[7]
				solar_pos = float(solar_pos_str)
				mcore_size = text_bits[10]
				M_core_num = float(mcore_size) 
				#print(M_core_num)
				Stokes_num_str = text_bits[13]
				particle_size = str(r_p_i)
				##!!##r_p = r_p_i commented out on 12/18, not sure if this is gonna cause problems
				M_core = M_core_num*m_earth #loose assumption where I just chose 10 Mearth for the core mass
				R_core = r_earth*(M_core/m_earth)**0.27
				m_jup = M_core
				r_jup = R_core #from Seager et al. 2007
				r_H_start = abs(sun_x_pos)*(m_jup/(3.0*M_sun))**(1.0/3.0)
				r_H_start_m = r_H_start*m_in_AU

				r_p_fromJup = (full_position_x_array_m**2.0 + full_position_y_array_m**2.0)**0.5
				r_H_indices_arr = np.where(r_p_fromJup<r_H_start_m)
				#print(np.min(r_p_fromJup)/r_H_start_m)
				#print(r_H_indices_arr)
				#bloop
				#print('before try function')
				##section where I cut the particle evolution into the section exterior to r_H, and interior to r_H. I then take slices for the interior section.
				try:
						
					r_H_start_index = np.min(r_H_indices_arr)
					
					#### NEED TO CHANGE ARRAY INDEXES!!! ####
					#t = t[r_H_start_index:-1:10000]
					t = np.concatenate([t[0:r_H_start_index],t[r_H_start_index:-1:slice_setting]])
					print(len(t))
					#t_1 = data_array_1[:,0]
					#print(t)
					#print(t_1)
					#bloop
					t_secs = t*s_in_yrs

					full_position_x_array_m = np.concatenate([full_position_x_array_m[0:r_H_start_index],full_position_x_array_m[r_H_start_index:-1:slice_setting]])
					full_position_y_array_m = np.concatenate([full_position_y_array_m[0:r_H_start_index],full_position_y_array_m[r_H_start_index:-1:slice_setting]])
					full_position_z_array_m = np.concatenate([full_position_z_array_m[0:r_H_start_index],full_position_z_array_m[r_H_start_index:-1:slice_setting]])
					#r_p_fromJup = (full_position_x_array_m**2.0 + full_position_y_array_m**2.0)**0.5
					#r_H_indices_arr = np.where(r_p_fromJup)

					full_vel_x_array = np.concatenate([full_vel_x_array[0:r_H_start_index],full_vel_x_array[r_H_start_index:-1:slice_setting]])
					full_vel_y_array = np.concatenate([full_vel_y_array[0:r_H_start_index],full_vel_y_array[r_H_start_index:-1:slice_setting]])
					full_vel_z_array = np.concatenate([full_vel_z_array[0:r_H_start_index],full_vel_z_array[r_H_start_index:-1:slice_setting]])
					##### END OF ARRAY INDEX CHANGES!! ######

					full_position_x_array = full_position_x_array_m/m_in_AU
					full_position_y_array = full_position_y_array_m/m_in_AU
					full_position_z_array = full_position_z_array_m/m_in_AU

					start_p_x = full_position_x_array[0]
					start_p_y = full_position_y_array[0]
					start_p_z = full_position_z_array[0]
				except:
					print('continue')
				
					
					
				#print('noodle')
				#print('number of timesteps', len(t))
				endtime = t[-1]
				epsilon = m_jup/(M_sun + m_jup)

				M_com = (M_sun*m_jup)/(M_sun + m_jup)

				#print('starting particle positions: ', start_p_x, start_p_y, start_p_z)

				save_txt_filename = 'dyn_3body_run_'+str(time_setting)+'_yrs_'+str(angle)+'_degrees_'+solar_pos_str+'_AU_Jupdist_'+mcore_size+'_mearth_core_'+Stokes_num_str+'_Stokes_'+str(kappa)+'_kappaSI_'+str(T_core)+'K_OBERG_'+str(ring_rad)+'_ringplane_rhopdown_Epdrag_2func_reggrid_slice'+str(slice_setting)+'_protoatm_general.npz'

				#######################################################################

				###starting position vectors and distance between the Sun and Jupiter (a) ###
				particle_position_vector_AU = np.array([start_p_x, start_p_y, start_p_z])
				solar_position_vector_AU = np.array([-solar_pos, 0.0, 0.0])
				sun_x_AU = solar_position_vector_AU[0]
				#print(sun_x_AU)
				sun_y_AU = solar_position_vector_AU[1]
				sun_z_AU = solar_position_vector_AU[2]
				jup_position_vector_AU = np.array([0.0, 0.0, 0.0])
				solar_position_vector_m = solar_position_vector_AU*m_in_AU
				jup_position_vector_m = jup_position_vector_AU*m_in_AU

				######### Calculating related values ##########

				jup_x_radius = np.array([r_jup/m_in_AU, 0.0, -r_jup/m_in_AU, 0.0])
				jup_y_radius = np.array([0.0, r_jup/m_in_AU, 0.0, -r_jup/m_in_AU])

				a_AU = ((solar_position_vector_AU[0] - jup_position_vector_AU[0])**2.0 + (solar_position_vector_AU[1] - jup_position_vector_AU[1])**2.0 + (solar_position_vector_AU[2] - jup_position_vector_AU[2])**2.0)**0.5
				a = a_AU * m_in_AU

				r_H = a_AU*(M_core/(3.0*M_sun))**(1.0/3.0)
				#r_protoatm = r_H/4.0

				x_com_rot_frame = (M_sun*solar_position_vector_m[0] + m_jup*jup_position_vector_m[0])/(M_sun + m_jup)
				y_com_rot_frame = (M_sun*solar_position_vector_m[1] + m_jup*jup_position_vector_m[1])/(M_sun + m_jup)
				#	print('com coords AU', x_com_rot_frame/m_in_AU, y_com_rot_frame/m_in_AU)
				#	print('particle coords AU', par_x_AU, par_y_AU)

				x_com_rot_frame_AU = x_com_rot_frame/m_in_AU
				y_com_rot_frame_AU = y_com_rot_frame/m_in_AU

				r_sun_rot_frame = (solar_position_vector_m[0]**2.0 + solar_position_vector_m[1]**2.0)**0.5
				r_jup_rot_frame = (jup_position_vector_m[0]**2.0 + jup_position_vector_m[1]**2.0)**0.5

				#	print(r_sun_rot_frame/m_in_AU, r_jup_rot_frame/m_in_AU)


				r_com_neg_sun_jup = (M_sun*r_sun_rot_frame + m_jup*r_jup_rot_frame)/(M_sun + m_jup)
				r_com_sun_jup = abs(r_com_neg_sun_jup)

				theta_j = math.asin((jup_position_vector_m[1] - y_com_rot_frame)/r_com_sun_jup) #angle between sun and jup in inertial frame in radians

				period = ((4.0*math.pi**2.0*r_com_sun_jup**3.0)/(G*(M_sun + m_jup)))**0.5
				omega_0 = 2.0*math.pi/period


				v_c_jupiter = (G*M_sun/(r_com_sun_jup))**0.5

				v_x_jupiter_start = (-v_c_jupiter*math.sin(theta_j))
				v_y_jupiter_start = v_c_jupiter*math.cos(theta_j)


				r_p_s = ((full_position_x_array_m - solar_position_vector_m[0])**2.0 + (full_position_y_array_m - solar_position_vector_m[1])**2.0 + (full_position_z_array_m - solar_position_vector_m[2])**2.0)**0.5

				##################################  SECTION CALCULATING JUP AND PARTICLE MOVEMENT WRT SUN #################################

				period = ((4.0*math.pi**2.0*r_com_sun_jup**3.0)/(G*(M_sun + m_jup)))**0.5
				omega_0 = 2.0*math.pi/period
				jup_period=period

				remainder = t%jup_period

				fraction = remainder/jup_period
				#print(fraction)
				degrees = fraction*360.0
				radians = np.radians(degrees)
				#radians = t*omega_0

				v_x_jup_array = -v_c_jupiter*np.sin(radians)
				v_y_jup_array = v_c_jupiter*np.cos(radians)
				dt_array = np.array([])

				for i_time in range(0, len(t_secs)-1):
					#print(i_time)
					if i_time == 1:
						dt_array = np.append(dt_array, 0.0)
					t_secs_i = t_secs[i_time+1]
					t_secs_i_1 = t_secs[i_time]
					dt = t_secs_i - t_secs_i_1
					dt_array = np.append(dt_array, dt)


				x_dt_jup = dt_array*v_x_jup_array
				y_dt_jup = dt_array*v_y_jup_array

				x_t_jup=r_com_sun_jup*np.cos(t*omega_0*s_in_yrs)-r_com_sun_jup
				y_t_jup=r_com_sun_jup*np.sin(t*omega_0*s_in_yrs)

				x_t_jup_AU = (x_t_jup/m_in_AU)
				y_t_jup_AU = y_t_jup/m_in_AU

				r_j = ((x_t_jup_AU-solar_position_vector_AU[0])**2.0 + (y_t_jup_AU-solar_position_vector_AU[1])**2.0)**0.5
				r_j_m = r_j*m_in_AU
				r_p_j = (full_position_x_array**2.0 + full_position_y_array**2.0)**0.5



				x=full_position_x_array
				y=full_position_y_array

				non_rot_x_p = (x-solar_position_vector_AU[0])*np.cos(t*omega_0*s_in_yrs)-y*np.sin(t*omega_0*s_in_yrs)
				non_rot_y_p = (x-solar_position_vector_AU[0])*np.sin(t*omega_0*s_in_yrs)+y*np.cos(t*omega_0*s_in_yrs)
				non_rot_x_p=non_rot_x_p+solar_position_vector_AU[0]

				non_rot_x_p_rel = non_rot_x_p - solar_position_vector_AU[0]
				non_rot_y_p_rel = non_rot_y_p - solar_position_vector_AU[1]

				non_rot_x_p_rel_m = non_rot_x_p_rel*m_in_AU
				non_rot_y_p_rel_m = non_rot_y_p_rel*m_in_AU

				r_p=(non_rot_x_p_rel**2.0 + non_rot_y_p_rel**2.0)**(0.5)
				r_p=r_p


				############## STARTING TEMPERATURE CALCULATION ##################

				r_p_s_i = r_p
				r_p_j_i = r_p_j

				r_p_s_i_m = r_p_s_i*m_in_AU
				r_p_j_i_m = r_p_j_i*m_in_AU

				T_back = 140.0*(r_p_s_i/2.0)**(-0.65) ##from Oberg and Wordsworth 2019, background temperature
				F_back = sigma*T_back**4.0 ##background flux calculated from background temperature

				scale_height_m = ((k*T_back/(2.3*m_H))**0.5)/((G*M_sun/(r_p_s_i_m**3.0))**0.5)
				sigma_back = 1.5e4*(r_p_s_i)**(-3.0/2.0) #surface density, in kg/m**2 ##from Oberg and Wordsworth 2019
				rho_back = sigma_back/((2.0*math.pi)**0.5*scale_height_m)

				#for i in range(len(T_core_array)):

				M_core = M_core
				R_core = R_core

				#T_core = T_core_array[i] #effective core temperature at core surface
				F_at_core = sigma*T_core**4.0 #flux from core temperature at core surface
				L_core = F_at_core*4.0*math.pi*R_core**2.0 #luminosity from core temperature
				L_frac = L_core/L_sun #core luminosity in solar luminosity units

				tau_array = rho_back*kappa*r_p_j_i_m #optical depth calculation between core and particle at particle location
						

				F_core = L_core/(4.0*math.pi*r_p_j_i_m**2.0) #flux from core temperature at particle location
				r_cutoff = 3.0*r_H
				cont_file = "temp_contour_"+str(-sun_x_AU)+"AU_"+str(M_core_num)+"Mearth_"+str(kappa)+"kappa_"+str(T_core)+"K_zoomed_3Rhill_highres2_contour_taufix_Tfix_protoatm_general.npz" #calling fine mesh temperature grid file created from GPC_temp_dist_opacity_with_background_OBERG_params_fixedtau_contour_lookup_creation_Tcore_protoatm_finalversion.py to interpolate for temperature particle feels
				T_at_r_array = np.zeros(len(x))
				ii = np.where(r_p_j_i>r_cutoff)
				jj = np.where(r_p_j_i<r_cutoff)
				T_at_r_array[ii] = temp_func_1(r_p_s_i[ii])
				T_at_r_array[jj] = temp_func_2(x[jj]+a_AU, y[jj], -sun_x_AU, sun_y_AU, r_H, cont_file)

				stack = np.column_stack((t, r_p_s_i))
				stack_1 = np.column_stack((stack, r_p_j_i))
				stack_2 = np.column_stack((stack_1, rho_back))
				stack_3 = np.column_stack((stack_2, T_back))
				stack_5 = np.column_stack((stack_3, T_at_r_array))
				stack_6 = np.column_stack((stack_5, non_rot_x_p_rel_m))
				stack_7 = np.column_stack((stack_6, non_rot_y_p_rel_m))
				stack_8 = np.column_stack((stack_7, full_position_x_array_m))
				stack_9 = np.column_stack((stack_8, full_position_y_array_m))
				np.savez_compressed(save_txt_filename, stack_9)




















	
