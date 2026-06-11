import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_continuous_are
import json
from utilities.read_config import data

# ALL UNITS IN SI UNITS (kg, m, s, N) unless otherwise specified

# Variables for thrust computation

thrust_data = pd.read_csv("motor_thrust_curve.csv", skiprows=1, header=None, names=["time", "thrust"])
t_array = thrust_data["time"].values
y_array = thrust_data["thrust"].values

t_min = t_array.min()
t_max = t_array.max()

CONFIG_FILE = "config.json"

with open(CONFIG_FILE, "r") as f:
    data = json.load(f)  # load motor data

L_NOZZLE_CG = data["Rocket_1"]["F15"]["thrust"]["L_NOZZLE_CG"]  # Distance between COM of Rocket and gimbal pivot (m)
SPECIFIC_IMPULSE = data["Rocket_1"]["F15"]["thrust"]["isp_s"]  # (s) for an F15-0 rocket motor
DRY_CG = data["Rocket_1"]["dimensions_weight"]["dry_CG"]  # (m) CG of dry mass from the nose tip
PROP_CG = data["Rocket_1"]["dimensions_weight"]["prop_CG"]  # (m) CG of propellant from the nose tip
BURN_TIME = data["Rocket_1"]["F15"]["thrust"]["burn_time_s"]  # (s)
INITIAL_MASS_PROPELLANT = data["Rocket_1"]["dimensions_weight"]["propellant_weight"]

def compute_thrust(t):
    if np.isscalar(t):
        if t < t_min or t > t_max:
            return 0.0
        return float(np.interp(t, t_array, y_array))
    else:
        thrust_values = np.interp(t, t_array, y_array)
        thrust_values[(t < t_min) | (t > t_max)] = 0.0
        return thrust_values


thrust_array = compute_thrust(t_array)
mdot_array = thrust_array / (SPECIFIC_IMPULSE * 9.81)  # mdot = F(t) / (I_sp * g_0)

# integrate mdot from 0 to t to get mass burned
mass_burned = cumulative_trapezoid(mdot_array, x=t_array[t_array <= t_array[-1]], initial=0)  


def compute_mass_propellant(t):
    global INITIAL_MASS_PROPELLANT, BURN_TIME, mass_burned, t_array
    if np.isscalar(t):
        if t <= 0:
            return INITIAL_MASS_PROPELLANT
        elif t > BURN_TIME:
            return 0.0
        burned = np.interp(t, t_array, mass_burned)
        return float(
            np.clip(INITIAL_MASS_PROPELLANT - burned, 0, INITIAL_MASS_PROPELLANT)
        )
    else:
        burned = np.interp(t, t_array, mass_burned)
        return np.clip(INITIAL_MASS_PROPELLANT - burned, 0, INITIAL_MASS_PROPELLANT)


def compute_dist_to_COM(
    t,
    dry_mass,

):
    global DRY_CG, PROP_CG

    if np.isscalar(t):
        prop_mass = compute_mass_propellant(t)
        total_mass = dry_mass + prop_mass

        if total_mass == 0:
            return 0.0

        X_COG = (dry_mass * DRY_CG + prop_mass * PROP_CG) / total_mass
        return float(X_COG)
    
    else:
        prop_mass = compute_mass_propellant(t)
        total_mass = dry_mass + prop_mass

        X_COG = np.where(
            total_mass > 0, (dry_mass * DRY_CG + prop_mass * PROP_CG) / total_mass, 0.0
        )
        return X_COG


def compute_inertia(
    X_CG, m_prop, dry_mass=0.034, airframe_mass=0.970, airframe_length=0.407
):

    total_dry_mass = dry_mass + airframe_mass

    # Inertia of the dry mass (modeled as a point mass at its CG)
    I_dry_cm = (1 / 12) * total_dry_mass * airframe_length**2
    d_dry = np.abs(DRY_CG - X_CG)
    I_dry = I_dry_cm + total_dry_mass * d_dry**2

    # Inertia of the propellant (modeled as a point mass at its CG)
    d_prop = np.abs(PROP_CG - X_CG)
    I_prop = m_prop * d_prop**2

    total_inertia = I_dry + I_prop
    return total_inertia


def compute_plant_gain(thrust, CG, inertia):

    # thrust, CG and inertia are all functions that take time as input and return corresponding value.

    return (thrust * CG) / inertia


def precompute_plant_gains(dry_mass, airframe_mass=0.970, airframe_length=0.407):
    """
    Pre-compute plant gains for all timesteps in t_array using vectorized operations.
    Optimized for Monte Carlo simulations by computing everything once at startup.
    
    """
    # Vectorized computation of all properties across the entire time array
    thrust_array = compute_thrust(t_array)
    mass_propellant = compute_mass_propellant(t_array)
    X_COM = compute_dist_to_COM(t_array, dry_mass)
    inertial = compute_inertia(X_COM, mass_propellant, dry_mass, airframe_mass, airframe_length)
    g_array = compute_plant_gain(thrust_array, X_COM, inertial)
    
    return g_array


def compute_LQR(g, Q, R):
    # for a 1 state system: gimbal angle and PID controller
    A = np.array([[0,1], [0,0]])
    B = np.array([[0], [g]]) # input matrix - only affects y_ddot
    P = solve_continuous_are(A, B, Q, R) # solve the continuous-time algebraic Riccati equation
    K = np.linalg.inv(R) @ B.T @ P # compute LQR gain

    return K # returns KP and KD gains for the controller
