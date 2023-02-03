## The purpose of this code is to track the dynamical evolution of solids as they radially drift inwards past a forming giant planet core.
## In this model, each particle is initialized in a ring separated by one degree to each other exterior to the orbit of the giant planet core, with the sun at the center of the ring. 
## Each .npz file stores the dynamical evolution information of ONE particle, and particles are evolved independently from each other, one after the other (hence the for-loop for particle position around the ring)
## In the resultling .npz file, we store the time, x and y position, and x, y, and z velocities of the solid at each timestep

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
from scipy.optimize import root
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.integrate import solve_ivp
from astropy import units

####starting constant definitions####
m_earth = 5.972e24 #mass of the earth in kg
r_earth = 6.378e6 #Earth radius in meters
s_in_yrs = 3.154e7
G = 6.67e-11 #gravitational constant in SI
M_sun = 1.989e30 #solar mass in kg
m_in_AU = 1.496e11
k = 1.38e-23
m_H = 1.67e-27
##################################

time_years_array = np.arange(0.0, 10.01, 0.01) #time array in years, inputted to ODEINT function to define which timesteps I want outputted
t = time_years_array*s_in_yrs #time array in seconds

### User defined settings ###
drag_on = 1 # Flag that determines whether drag is considered in particle dynamic evolution, drag is on if = 1
n = 0

sun_pos_arr = np.array([-40.0]) #starting position of the sun in the planet frame of reference 
mcore_arr = np.array([5.0]) #mass in Earth masses of our protoplanet, can put a variety of values to compare
St_arr = np.array([1.0]) #Starting stokes number of our particles
r_planet_start_ring = 2.0 #The distance in AU we want our ring of particles to start from the protoplanet
rho_p = 1000.0 #internal solid density of the particles

###starting position vectors and distance between the Sun and Jupiter (a) ###
#First loop to loop over the various core masses
for i_mcore in range(len(mcore_arr)):
	mcore_i = mcore_arr[i_mcore]

	m_jup = mcore_i*m_earth #10 Mearth core mass

	r_jup = r_earth*(m_jup/m_earth)**0.27 #from Seager et al. 2007
	epsilon = m_jup/(M_sun + m_jup)

	M_com = (M_sun*m_jup)/(M_sun + m_jup)

	#loop over various orbital separations we want to consider
	for i_sun_pos in range(len(sun_pos_arr)):
		sun_x_pos = sun_pos_arr[i_sun_pos]

		#loop over the various particle starting ring positions, here we use 0 to 360 to model particles at 1 degree separation around the entire ring, can switch this to be any location we want though
		for i_px_pos in range(0, 360):
			angle = float(i_px_pos)
			angle_rad = math.radians(angle)
			ring_rad = r_planet_start_ring + abs(sun_x_pos)
			px_pos = ring_rad*math.cos(angle_rad) - abs(sun_x_pos)
			py_pos = ring_rad*math.sin(angle_rad)

			#loop over various particle starting Stokes number sizes
			for i_r_p in range(len(St_arr)):
				St = St_arr[i_r_p] #assign starting St number for the solids that we will calculate the particle radius from
				a_i = ring_rad
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
				#---------------------------------------------------#
				Cd_i = (8.0/3.0)*v_th_i/delta_v_i #Drag coefficient for Epstein regime
				#---------------------------------------------------#

				r_p = St*(rho_g_i/rho_p)*v_th_i/omega_k_i #final particle radius we got from St, all calculations in between are related to getting r_p

				solar_position_vector_AU = np.array([sun_x_pos, 0.0, 0.0])
				sun_x_AU = solar_position_vector_AU[0]
				sun_y_AU = solar_position_vector_AU[1]
				sun_z_AU = solar_position_vector_AU[2]
				particle_position_vector_AU = np.array([px_pos, py_pos, 0.0])
				jup_position_vector_AU = np.array([0.0, 0.0, 0.0])
				solar_position_vector_m = solar_position_vector_AU*m_in_AU
				jup_position_vector_m = jup_position_vector_AU*m_in_AU

				a_AU = ((solar_position_vector_AU[0] - jup_position_vector_AU[0])**2.0 + (solar_position_vector_AU[1] - jup_position_vector_AU[1])**2.0 + (solar_position_vector_AU[2] - jup_position_vector_AU[2])**2.0)**0.5
				a = a_AU * m_in_AU

				T_a_start = 140.0*(a_AU/2.0)**(-0.65) ##from Oberg and Wordsworth 2019, calculating the temperature at the particle location  

				R_h_AU = a_AU*(m_jup/(3.0*M_sun))**(1.0/3.0) #Hill radius calculation of the protoplanet
				R_h = R_h_AU * m_in_AU

				R_b = G*m_jup/((k*T_a_start)/(2.3*m_H)) #Bondi radius calculation for the protoplanet
				R_b_AU = R_b/m_in_AU

				#determines whether the 1/4 of the Hill radius or Bondi radius is the radius of the protoplanet's protoatmosphere
				if R_h/4.0 < R_b:
					R_protoatm = R_h/4.0
				else:
					R_protoatm = R_b

				x_com_rot_frame = (M_sun*solar_position_vector_m[0] + m_jup*jup_position_vector_m[0])/(M_sun + m_jup)
				y_com_rot_frame = (M_sun*solar_position_vector_m[1] + m_jup*jup_position_vector_m[1])/(M_sun + m_jup)

				x_com_rot_frame_AU = x_com_rot_frame/m_in_AU
				y_com_rot_frame_AU = y_com_rot_frame/m_in_AU

				r_sun_rot_frame = (solar_position_vector_m[0]**2.0 + solar_position_vector_m[1]**2.0)**0.5
				r_jup_rot_frame = (jup_position_vector_m[0]**2.0 + jup_position_vector_m[1]**2.0)**0.5

				r_com_neg_sun_jup = (M_sun*r_sun_rot_frame + m_jup*r_jup_rot_frame)/(M_sun + m_jup)
				r_com_sun_jup = abs(r_com_neg_sun_jup)

				theta_j = math.asin((jup_position_vector_m[1] - y_com_rot_frame)/r_com_sun_jup) #angle between sun and protoplanet in inertial frame in radians

				period = ((4.0*math.pi**2.0*r_com_sun_jup**3.0)/(G*(M_sun + m_jup)))**0.5 #protoplanet period
				omega_0 = 2.0*math.pi/period #

				v_c_jupiter = (G*M_sun/(r_com_sun_jup))**0.5 #centripedal velocity of protoplanet orbiting the sun

				v_x_jupiter_start = (-v_c_jupiter*math.sin(theta_j))
				v_y_jupiter_start = v_c_jupiter*math.cos(theta_j)

				#function that calculates the starting velocities of the particle from the given starting position information
				#needed for the integrator to calculate the concurrent velocities and accelerations at each subsequent timestep
				def start_func(particle_position_vector_AU):

					particle_position_vector_m = particle_position_vector_AU*m_in_AU

					par_x = particle_position_vector_m[0]
					par_y = particle_position_vector_m[1]
					par_z = particle_position_vector_m[2]

					par_x_AU = particle_position_vector_AU[0]
					par_y_AU = particle_position_vector_AU[1]
					par_z_AU = particle_position_vector_AU[2]

					r_p_s_AU = ((par_x_AU - solar_position_vector_AU[0])**2.0 + (par_y_AU - solar_position_vector_AU[1])**2.0 + (par_z_AU - solar_position_vector_AU[2])**2.0)**0.5 
					r_p_s = r_p_s_AU*m_in_AU

					theta_p = math.asin((par_y - solar_position_vector_m[1])/r_p_s)

					##calculation correction for issue with math.asin not giving the correct angle if you are outside quad 1!! ###
					x_p_com_diff = par_x_AU - sun_x_AU
					y_p_com_diff = par_y_AU - sun_y_AU
					if x_p_com_diff < 0.0 and y_p_com_diff > 0.0: #checked and ready to roll!!
						theta_p = math.radians(180.0) - theta_p
					elif x_p_com_diff < 0.0 and y_p_com_diff < 0.0:
						theta_p = math.radians(180.0) - theta_p
					elif x_p_com_diff > 0.0 and y_p_com_diff < 0.0:
						theta_p = theta_p + math.radians(360.0)
					elif x_p_com_diff < 0.0 and y_p_com_diff == 0.0:
						theta_p = math.radians(180.0)
					elif x_p_com_diff == 0.0 and y_p_com_diff < 0.0:
						theta_p = math.radians(270.0)

					###################################################

					v_c_particle = (G*M_sun/(r_p_s))**0.5
					v_x_particle_at_orbit_start = (-v_c_particle*math.sin(theta_p))
					v_y_particle_at_orbit_start = v_c_particle*math.cos(theta_p)
					v_x_omega_cross = -omega_0*(par_y - jup_position_vector_m[1])
					v_y_omega_cross = omega_0*(par_x - jup_position_vector_m[0])
						
					v_y_diff = v_y_particle_at_orbit_start - v_y_jupiter_start - v_y_omega_cross
					v_x_diff = v_x_particle_at_orbit_start - v_x_jupiter_start - v_x_omega_cross

					particle_velocity_vector = np.array([v_x_diff, v_y_diff, 0.0])

					full_position_array = particle_position_vector_m
					full_velocity_array = particle_velocity_vector
					full_acceleration_array = np.array([0.0, 0.0, 0.0])

					full_position_x_array = np.array([particle_position_vector_AU[0]])
					full_position_y_array = np.array([particle_position_vector_AU[1]])
					full_position_z_array = np.array([particle_position_vector_AU[2]])

					output = np.array([full_position_array, full_velocity_array])

					return output


				#function that calculates the acceleration components for the particles
				def accel_calc(t,ini_cond_arr):
					### constants to start ####

					L_sun = 3.827e26 #solar luminosity in W
					R_sun = 6.9634e8 # radius of the sun in m
					#r_jup = 6.9911e7 #Jup radius in meters

					k = 1.38e-23 #boltzman constant in SI
					G = 6.67e-11 #gravitational constant in SI
					m_H = 1.67e-27 #mass of hydrogen atom in SI
					sigma = 5.67e-8 #SB constant in SI
					s_in_yrs = 3.14e7 #number of seconds in a year
					m_in_AU = 1.496e11 #number of meters in an AU

					cross_section_H2 = 2.e-20 #cross sectional area of H2 in meters^2
					m_g = 2.3*m_H #average mass of a protoplanetary disk gas molecule (85% H2, 14% He, ~1% other molecules)

					#### starting definitions ####
					particle_x = ini_cond_arr[0]
					particle_y = ini_cond_arr[1]
					particle_z = ini_cond_arr[2]

					par_x = particle_x
					par_y = particle_y
					par_z = particle_z

					par_x_AU = particle_x/m_in_AU
					par_y_AU = particle_y/m_in_AU
					par_z_AU = particle_z/m_in_AU

					v_p_x = ini_cond_arr[3]
					v_p_y = ini_cond_arr[4]
					v_p_z = ini_cond_arr[5]

					particle_velocity_vector = np.array([v_p_x, v_p_y, v_p_z])

					r_p_s = ((par_x - solar_position_vector_m[0])**2.0 + (par_y - solar_position_vector_m[1])**2.0 + (par_z - solar_position_vector_m[2])**2.0)**0.5
					r_p_s_AU = r_p_s/m_in_AU

					r_p_j = (par_x**2.0 + par_y**2.0 + par_z**2.0)**0.5

					## gas parameters ##
					T_loc = 140.0*(r_p_s_AU/2.0)**(-0.65) ##from Oberg and Wordsworth 2019
					#print(T_loc)

					scale_height_m = ((k*T_loc/(2.3*m_H))**0.5)/((G*M_sun/(r_p_s**3.0))**0.5)
					#print(scale_height_m)
					sigma_loc = 1.5e4*(r_p_s_AU)**(-3.0/2.0) #surface density, in kg/m**2 ##from Oberg and Wordsworth 2019
					#print(sigma_back)
					#print(sigma_loc)
					rho_g = sigma_loc/((2.0*math.pi)**0.5*scale_height_m)
					#rho_g = 1.e-4
					#print(rho_g)

					#rho_g = 5.e-6 #density of gas in kg per m**3, starting with placeholder value -----> need to change this so it's actually calculated!
					n_g = rho_g/(2.3*m_H)
					#print(n_g)

					rho_d_to_rho_g = 0.005
					rho_d = 0.005*rho_g

					## particle parameters ##
					rho_p = 1000.0 #in kg/m**3
					m_p = rho_p*((4.0/3.0)*math.pi*r_p**3.0)

					c_s = ((k*T_loc)/(2.3*m_H))**0.5 #sound speed in m/s

					dP_dr = -(11.0/4.0)*c_s**2.0*rho_g/r_p_s
					dP_dr_mag = abs(dP_dr)

					delta_v = (r_p_s*((G*M_sun/r_p_s**2.0)+(dP_dr/rho_g)))**0.5 - (G*M_sun/r_p_s)**0.5 #difference between azimuthal gas and dust velocities
					########################################


					## INITIAL CONDITIONS ##

					mfp = m_g/(cross_section_H2*rho_g)

					theta_p = math.asin((par_y - solar_position_vector_m[1])/r_p_s)

					##calculation correction for issue with math.asin not giving the correct angle if you are outside quad 1!! ###
					x_p_com_diff = par_x_AU - sun_x_AU
					y_p_com_diff = par_y_AU - sun_y_AU
					if x_p_com_diff < 0.0 and y_p_com_diff > 0.0:
						theta_p = math.radians(180.0) - theta_p
					elif x_p_com_diff < 0.0 and y_p_com_diff < 0.0:
						theta_p = math.radians(180.0) - theta_p
					elif x_p_com_diff > 0.0 and y_p_com_diff < 0.0:
						theta_p = theta_p + math.radians(360.0)
					elif x_p_com_diff < 0.0 and y_p_com_diff == 0.0:
						theta_p = math.radians(180.0)
					elif x_p_com_diff == 0.0 and y_p_com_diff < 0.0:
						theta_p = math.radians(270.0)


					com_position_vector = np.array([x_com_rot_frame, y_com_rot_frame, 0.0])

					momentum_p_x = v_p_x*m_p
					momentum_p_y = v_p_y*m_p
					momentum_p_z = v_p_z*m_p

					particle_velocity_mag = (v_p_x**2.0 + v_p_y**2.0 + v_p_z**2.0)**0.5


					v_c_g = (G*M_sun/(r_p_s))**0.5
						
					v_c_g_x = -v_c_g*math.sin(theta_p)
					v_c_g_y = v_c_g*math.cos(theta_p)


					v_x_omega_cross = -omega_0*(par_y - jup_position_vector_m[1])
					v_y_omega_cross = omega_0*(par_x - jup_position_vector_m[0])

					v_g_x = v_c_g_x + delta_v*(-math.sin(theta_p)) - v_x_jupiter_start - v_x_omega_cross
					v_g_y = v_c_g_y + delta_v*(math.cos(theta_p)) - v_y_jupiter_start - v_y_omega_cross
						
					vrelx=abs(v_p_x)-abs(v_g_x)
					vrely=abs(v_p_y)-abs(v_g_y)

					v_g_mag = (v_g_x**2.0 + v_g_y**2.0)**0.5
					vrel = abs(particle_velocity_mag) - abs(v_g_mag)
						

					v_g_vector = np.array([v_g_x, v_g_y, 0.0]) #gas velocity in meters per second, starting with placeholder value
					v_g_mag = (v_g_vector[0]**2.0 + v_g_vector[1]**2.0 + v_g_vector[2]**2.0)**0.5


					u_mag = ((v_p_x-v_g_x)**2.0 + (v_p_y-v_g_y)**2.0)**0.5
					#print(u_mag)

					if u_mag < 0.0:
						print('gas faster than particle', u_mag, particle_velocity_mag, v_g_mag)

					R_num = (2.0*(u_mag)*r_p)/(0.353*(8.0/math.pi)**0.5*c_s*mfp)
					M_num = u_mag/c_s

					if R_num > 2.e5:
						w = 0.2
					else:
						w = 0.4

					Cd = (8.0/3.0)*(8.0/math.pi)**0.5*c_s/u_mag
					accel_drag_vector = -(Cd*rho_g*math.pi*r_p**2.0*u_mag*(particle_velocity_vector - v_g_vector))/(2.0*m_p)

					accel_grav_x_sun = -((G*M_sun*(particle_x + a))/(((particle_x+a)**2.0 + particle_y**2.0 + particle_z**2.0)**1.5))
					accel_grav_x_jup = -((G*m_jup*particle_x)/(((particle_x)**2.0 + particle_y**2.0 + particle_z**2.0)**1.5))

					accel_grav_y_sun = -((G*M_sun)/(((particle_x+a)**2.0 + particle_y**2.0 + particle_z**2.0)**1.5))*particle_y
					accel_grav_y_jup = -((G*m_jup)/(((particle_x)**2.0 + particle_y**2.0 + particle_z**2.0)**1.5))*particle_y

					accel_grav_z_sun =  -((G*M_sun)/(((particle_x+a)**2.0 + particle_y**2.0 + particle_z**2.0)**1.5))*particle_z
					accel_grav_z_jup = -((G*m_jup)/(((particle_x)**2.0 + particle_y**2.0 + particle_z**2.0)**1.5))*particle_z


					accel_grav_x = accel_grav_x_sun + accel_grav_x_jup
					accel_grav_y = accel_grav_y_sun + accel_grav_y_jup
					accel_grav_z = accel_grav_z_sun + accel_grav_z_jup

					accel_centrip_x = omega_0**2.0*(particle_x + a*(1.0-epsilon))
					accel_centrip_y = omega_0**2.0*particle_y


					accel_cori_x = 2.0*omega_0*v_p_y
					accel_cori_y = -2.0*omega_0*v_p_x

					accel_drag_x = accel_drag_vector[0]
					accel_drag_y = accel_drag_vector[1]
					accel_drag_z = accel_drag_vector[2]

					if drag_on == 0:
						accel_x_tot = accel_grav_x + accel_cori_x + accel_centrip_x
						accel_y_tot = accel_grav_y + accel_cori_y + accel_centrip_y
						accel_z_tot = accel_grav_z
					else:
						accel_x_tot = accel_grav_x + accel_cori_x + accel_centrip_x + accel_drag_x
						accel_y_tot = accel_grav_y + accel_cori_y + accel_centrip_y + accel_drag_y
						accel_z_tot = accel_grav_z + accel_drag_z

					if r_p_s <= R_sun:
						accel_x_tot = 0.0
						accel_y_tot = 0.0
						accel_z_tot = 0.0

						v_p_x = 0.0
						v_p_y = 0.0
						v_p_z = 0.0
						print('Accreted by Sun!!')
					if r_p_j <= R_protoatm:
						print(r_p_j)
						accel_x_tot = 0.0
						accel_y_tot = 0.0
						accel_z_tot = 0.0

						v_p_x = 0.0
						v_p_y = 0.0
						v_p_z = 0.0 
						print('Accreted by Protoplanet!!!')

					dr_dt = np.array([v_p_x, v_p_y, v_p_z])

					dv_dt = np.array([accel_x_tot, accel_y_tot, accel_z_tot])

					return np.array([v_p_x, v_p_y, v_p_z, accel_x_tot, accel_y_tot, accel_z_tot])



				starting_output = start_func(particle_position_vector_AU)

				input_1 = starting_output[0][0]
				input_2 = starting_output[0][1]
				input_3 = starting_output[0][2]
				input_4 = starting_output[1][0]
				input_5 = starting_output[1][1]
				input_6 = starting_output[1][2]

				ini_cond_arr2 = np.array([input_1, input_2, input_3, input_4, input_5, input_6]) #initial conditions for the particle to be passed to the integrator

				sol = solve_ivp(accel_calc,[0,np.amax(t)],ini_cond_arr2,rtol=1.e-10) #calling the integrator function, also uses the acceleration calc function to determine the particle velocity and particle position at each timestep from acceleration


				y=sol.y
				t_secs = sol.t
				t_boi=sol.t/s_in_yrs

				full_position_x_array = y[0,:]/m_in_AU
				full_position_y_array = y[1,:]/m_in_AU
				full_position_z_array = y[2,:]/m_in_AU

				full_position_x_array_m = full_position_x_array*m_in_AU
				full_position_y_array_m = full_position_y_array*m_in_AU
				full_position_z_array_m = full_position_z_array*m_in_AU

				full_vel_x_array = y[3,:]
				full_vel_y_array = y[4,:]
				full_vel_z_array = y[5,:]

				stack_1 = np.column_stack((t_boi, full_position_x_array_m))
				stack_1_5 = np.column_stack((stack_1, full_position_y_array_m))
				stack_2 = np.column_stack((stack_1_5, full_position_z_array_m))
				stack_3 = np.column_stack((stack_2, full_vel_x_array))
				stack_4 = np.column_stack((stack_3, full_vel_y_array))
				stack_5 = np.column_stack((stack_4, full_vel_z_array))
				#print(len(t), len(full_position_x_array_m))

				text_name = 'dyn_3body_run_' + str(time_years_array[-1]) + '_yrs_'+str(angle)+'_degrees_'+str(r_sun_rot_frame/m_in_AU)+'_AU_Jupdist_'+str(m_jup/m_earth)+'_mearth_core_'+str(St)+'_Stokes_DRAG_OBERG_'+str(ring_rad)+'_ringplane_rhopdown_Epdrag_protoatm_general.npz'
				np.savez_compressed(text_name, stack_5)




				
