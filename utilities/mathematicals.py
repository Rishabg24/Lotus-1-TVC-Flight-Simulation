"""
Houses all mathematical functions for the linear-quadratic regulator simulation 
using a class based approach. Each class represents the main components of the simulation,
with each mathematical function modeling a certain aspect of the specific object the class is representing. 
"""

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_continuous_are
import json
from utilities.read_config import data

# ALL UNITS IN SI UNITS (kg, m, s, N) unless otherwise specified

class Motor:
    
    def __init__(self, thrust_csv, time_csv, motor_config_data):
        # Assuming config data is list: 
        # [SPECIFIC_IMPULSE, PROP_CG, BURN_TIME, INITIAL_MASS_PROPELLANT, dry_mass, prop_mass,]

        # extracting parameter level data 
        self.thrust_data = thrust_csv
        self.specific_impulse = motor_config_data[0]
        self.prop_cg = motor_config_data[1]
        self.burn_time = motor_config_data[2]
        self.initial_mass_prop = motor_config_data[3]
        self.dry_mass = motor_config_data[4]
        self.prop_mass = motor_config_data[5]
        self.time_data = time_csv

        # computing non-parameter level data
        self.t_min = self.time_data.min()
        self.t_max = self.time_data.max()

        
        self.thrust_array =self.compute_thrust(self.time_data)
        self.mass_burned_array = self.compute_mass_burned()
        
    def compute_thrust(self, t):
        if np.isscalar(t):
            if t < self.t_min or t > self.t_max:
                return 0.0
            return float(np.interp(t, self.time_data, self.thrust_data))
        else:
            thrust_values = np.interp(t, self.time_data, self.thrust_data)
            thrust_values[(t < self.t_min) | (t > self.t_max)] = 0.0
            return thrust_values
        
    def compute_mass_burned(self):

        thrust_array = self.thrust_array
        mdot_array = thrust_array / (self.specific_impulse * 9.81)  # mdot = F(t) / (I_sp * g_0)

        # integrate mdot from 0 to t to get mass burned
        mass_burned = cumulative_trapezoid(mdot_array, x=self.time_data[self.time_data <= self.time_data[-1]], initial=0) 
        return mass_burned

    def compute_mass_propellant(self, t,):
        if np.isscalar(t):
            if t <= 0:
                return self.initial_mass_prop
            elif t > self.burn_time:
                return 0.0
            burned = np.interp(t, self.time_data, self.mass_burned_array)
            return float(
                np.clip(self.initial_mass_prop - burned, 0, self.initial_mass_prop)
            )
        else:
            burned = np.interp(t, self.time_data,self.mass_burned_array)
            return np.clip(self.initial_mass_prop - burned, 0, self.initial_mass_prop)
        

class Rocket:
    def __init__(self, airframe_data):
        # data = [airframe_length, airframe_mass, dry_cg]
        self.airframe_length = airframe_data[0]
        self.airframe_mass = airframe_data[1]
        self.dry_cg = airframe_data[2]

    def compute_dist_to_COM(
        self,
        t,
        motor

    ):
        dry_mass = motor.dry_mass
        DRY_CG = self.dry_cg
        PROP_CG = motor.prop_cg

        if np.isscalar(t):
            prop_mass = motor.compute_mass_propellant(t)
            total_mass = dry_mass + prop_mass

            if total_mass == 0:
                return 0.0

            X_COG = (dry_mass * DRY_CG + prop_mass * PROP_CG) / total_mass
            return float(X_COG)
        
        else:
            prop_mass = motor.compute_mass_propellant(t)
            total_mass = dry_mass + prop_mass

            X_COG = np.where(
                total_mass > 0, (dry_mass * DRY_CG + prop_mass * PROP_CG) / total_mass, 0.0
            )
            return X_COG


    def compute_inertia(
        self, t, motor, X_CG = None
        ):

        dry_mass = motor.dry_mass
        m_prop = motor.compute_mass_propellant(t)
        DRY_CG = self.dry_cg
        PROP_CG = motor.prop_cg

        airframe_mass = self.airframe_mass
        airframe_length = self.airframe_length

        if X_CG is None:
            X_CG = self.compute_dist_to_COM(t, motor)
        else:
            X_CG = X_CG

        total_dry_mass = dry_mass + airframe_mass

        
        # Inertia of the dry mass (modeled as thin rod)
        I_dry_cm = (1 / 12) * total_dry_mass * airframe_length**2
        d_dry = np.abs(DRY_CG - X_CG)
        I_dry = I_dry_cm + total_dry_mass * d_dry**2

        # Inertia of the propellant (modeled as a point mass at its CG)
        d_prop = np.abs(PROP_CG - X_CG)
        I_prop = m_prop * d_prop**2

        total_inertia = I_dry + I_prop
        return total_inertia
    
    def compute_plant_gain(self, t, motor):

        # thrust, CG and inertia are all functions that take time as input and return corresponding value.
        thrust = motor.compute_thrust(t)
        COM = self.compute_dist_to_COM(t, motor)
        inertia = self.compute_inertia(t, motor, COM)

        return (thrust * COM) / inertia
    

def precompute_plant_gains(time_array, rocket, motor):
    """
    Pre-compute plant gains for all timesteps in t_array using vectorized operations.
    Optimized for Monte Carlo simulations by computing everything once at startup.
    
    """
    # Vectorized computation of all properties across the entire time array
    g_array = rocket.compute_plant_gain(time_array, motor)
    
    return g_array


def compute_LQR(g, Q, R):
    # for a 1 state system: gimbal angle and PID controller
    A = np.array([[0,1], [0,0]])
    B = np.array([[0], [g]]) # input matrix - only affects y_ddot
    P = solve_continuous_are(A, B, Q, R) # solve the continuous-time algebraic Riccati equation
    K = np.linalg.inv(R) @ B.T @ P # compute LQR gain

    return K # returns KP and KD gains for the controller


def dynamics(t, x, K, g, motor=None, rocket=None, phase="powered"):
    theta, theta_dot, v, h = x
    attitude_state = x[:2]
    u = float((-K @ attitude_state).item())
    u = np.clip(u, -np.deg2rad(20), np.deg2rad(20))
    theta_ddot = g * u

    if phase == "powered":
        mass = rocket.airframe_mass + motor.compute_mass_propellant(t)
        v_dot = (motor.compute_thrust(t) - mass * 9.81) / mass
    else:
        v_dot = -9.81
    h_dot = v

    return np.array([theta_dot, theta_ddot, v_dot, h_dot])


def coasting_dynamics(t, x, K_alpha, K_d, g, theta_wind, inertia):
    """
    using an aerodynamic restoring model to represent wind
    basically wind tries to align rocket with the direction of wind
    """
    theta, theta_dot , v, h = x

    delta_theta = theta - theta_wind
    tau = -K_alpha * delta_theta - K_d * theta_dot
    theta_ddot = tau / inertia

    v_dot = g
    h_dot = v

    return np.array([theta_dot, theta_ddot, v_dot, h_dot])

# def vertical_dynamics(t, x, rocket, phase, motor = None, ):
#     v = x[0]
#     h = x[1]
#     if phase == "powered":
#         mass = rocket.airframe_mass + motor.compute_mass_propellant(t)
#         v_dot = (motor.compute_thrust(t) - mass * 9.81) / mass
#     elif phase == "coast":
#         v_dot = -9.81
#     else:
#         raise ValueError(" Phase Not Defined. Choose either 'power' or 'coast' ")
#     h_dot = v

#     x = np.array([v_dot, h_dot])

#     return x
    


def RK4(f, t, x, h, *args, **kwargs):

    k1 = f(t, x, *args, **kwargs)
    k2 = f(t + h / 2, x + h / 2 * k1, *args, **kwargs)
    k3 = f(t + h / 2, x + h / 2 * k2, *args, **kwargs)
    k4 = f(t + h, x + h * k3, *args, **kwargs)

    x_next = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next

def compute_hoverslam(velocity, accel):
    h = velocity**2 / (2 * accel)
    return h


