import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utilities.read_config import data
from utilities.mathematicals import Motor, Rocket
from utilities.mathematicals import compute_LQR, precompute_plant_gains, dynamics, coasting_dynamics, compute_hoverslam,  RK4
from utilities.barrowman_coeffs import compute_k_alpha, compute_total_cnalpha_and_cp

# All angles in RADIANS unless otherwise specified

F15_data = pd.read_csv(
    "data_processed/motor_thrust_curve.csv",
    skiprows=1,
    header=None,
    names=["time", "thrust"],
)
time_f_array = F15_data["time"].values
thrust_f_array = F15_data["thrust"].values

# D12_data = pd.read_csv(
#     "data_processed/motor_thrust_curve_d12.csv",
#     skiprows=1,
#     header=None,
#     names=["time", "thrust"],
# )
# time_d_array = D12_data["time"].values
# thrust_d_array = D12_data["thrust"].values

CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r") as f:
    data = json.load(f)  # load rocket system data

# =========================
# Motor Data
# =========================

f_data = data["Rocket_1"]["F15"]
d_data = data["Rocket_1"]["D12"]

# =========================
# F15 Configuration
# =========================

f_motor_config = [
    f_data["thrust"]["isp_s"],  # specific_impulse
    f_data["prop_cg"],  # prop_cg
    f_data["thrust"]["burn_time_s"],  # burn_time
    f_data["prop_mass"],  # initial_mass_prop
    f_data["dry_mass"],  # dry_mass
    f_data["prop_mass"],  # prop_mass
]

# =========================
# D12 Configuration
# =========================

d_motor_config = [
    d_data["thrust"]["isp_s"],  # specific_impulse
    d_data["prop_cg"],  # prop_cg
    d_data["thrust"]["burn_time_s"],  # burn_time
    d_data["prop_mass"],  # initial_mass_prop
    d_data["dry_mass"],  # dry_mass
    d_data["prop_mass"],  # prop_mass
]

# =========================
# Airframe Data
# =========================

airframe_data_json = data["Rocket_1"]["dimensions_weight"]

airframe_data = [
    airframe_data_json["airframe_length_m"],
    airframe_data_json["airframe_mass_kg"],
    airframe_data_json["dry_CG"],
]

F15_A = Motor(
    thrust_csv=thrust_f_array,
    time_csv=time_f_array,
    motor_config_data=f_motor_config,
) # Ascent Motor

F15_D = Motor(
    thrust_csv=thrust_f_array,
    time_csv=time_f_array,
    motor_config_data=d_motor_config,
) # Descent Motor

Lotus = Rocket(airframe_data=airframe_data)



sigma_theta = np.deg2rad(0.5)
sigma_omega = np.deg2rad(1.5)

def sim(
    motor,
    rocket,
    inital_state,
    sigma_theta=sigma_theta,
    sigma_omega=sigma_omega,
    thrust_scale=1.0,
    gain_scale=1.0,
    inertia_scale=1.0,
    phase = "powered"
):
    if motor is not None and rocket is not None:
        # Pre-compute plant gains once using vectorized operations (optimized for Monte Carlo)
        time_array = motor.time_data
        g_array = precompute_plant_gains(time_array, rocket, motor)
        g_scaled = g_array * thrust_scale * gain_scale / inertia_scale

        LQR_K_array = np.zeros(
            (len(time_array), 1, 2)
        )  # Pre-allocate array for LQR gains
        trajectory = np.zeros((len(time_array), 4))
        x = inital_state
        trajectory[0] = x
        max_tilt = 0.0    

        Q = np.diag([100.0, 10.0])  # [theta penalty, theta_dot penalty]
        R = np.array([[1.0]])

        # Now iterate through timesteps using pre-computed gains
        for i in range(1, len(time_array)):

            # NOTE: state is 2 dimensional: [theta, theta_dot]
            g = g_scaled[i]

            if np.abs(g) <= 1e-6:
                continue

            LQR_K = compute_LQR(g, Q, R)
            LQR_K_array[i] = LQR_K

            h = time_array[i] - time_array[i - 1]  # always positive
            y = RK4(dynamics, time_array[i - 1], x, h, LQR_K, g, motor, rocket, phase)

            max_tilt = max(max_tilt, np.abs(y[0]))


            x = np.array(
                [
                    y[0] + np.random.normal(0, sigma_theta),
                    y[1] + np.random.normal(0, sigma_omega),
                    y[2],
                    y[3]
                ]
            )
            trajectory[i] = y

        KD = LQR_K_array[:, 0, 1]  # Extract the derivative gain (K_d) for plotting
        KP = LQR_K_array[:, 0, 0]  # Extract the proportional gain (K_p) for plotting

        return trajectory, KD, KP, max_tilt,  
    else:
        raise ValueError("Motor or Rocket Cannot be None")

    # ==========================================
    # Uncomment for plotting LQR gains over time
    # ==========================================

    # print("KD array:", KD)
    # print("KP array:", KP)

    # print("beginning to plot")

    # plt.plot(t_array, KD)
    # plt.plot(t_array, KP)
    # plt.xlabel("Time (s)")
    # plt.ylabel("LQR Gains")
    # plt.title("LQR Gains over Time")
    # plt.legend(["K_d (Derivative Gain)", "K_p (Proportional Gain)"])
    # plt.grid()
    # plt.show()


def compute_wind_force(rho, v, C_d, A):

    # computes wind force as if its or

    F = 0.5 * rho * C_d * A * (v**2)
    return F


def generate_wind_velocity(v1, v2, theta_1, theta_2):
    # generates a wind velocity, which is treated as a horizontal wind, relative to the rocket body's axis

    v = np.random.uniform(v1, v2)
    theta = np.random.uniform(theta_1, theta_2)
    v_horz = v * theta
    return v_horz


def sim_wind(state, rocket=Lotus):

    ''' 
    simulating the rocket free falling downward from apogee.
    Because its from apogee the initial velocity is assumed to be 0 m/s. 
    state is now 4D: [theta, theta_dot, v, h]
    '''
    if rocket is not None:
        K_d = 0
        C_NA, C_P = compute_total_cnalpha_and_cp()
        cg_nose = 0.1387  # m
        v_current = 0.0
        v_initial = 0.0
        g = -9.81
        max_tilt = 0.0
        inertia = (
            (1 / 12) * rocket.airframe_mass * rocket.airframe_length**2
            + rocket.airframe_mass * (cg_nose - rocket.airframe_length / 2) ** 2
        )
       
        wind_v = generate_wind_velocity(
            8.0, 15.0, 35.0, 80.0
        )  # effective_wind_velocity

        dt = 0.01
        t_current = 0.0
            
        hoverslam_altitude = 0.0
        current_altitude = state[3]
        altitude_buffer = 0.3

        while (current_altitude > hoverslam_altitude + altitude_buffer):

            v_current = state[2]

            theta_wind = np.arctan(wind_v / (v_current + 1e-3))
            theta_wind_final = theta_wind if abs(theta_wind) < np.deg2rad(90) else 0

            k_alpha = compute_k_alpha(v_current, cg_nose, C_NA, C_P)

            state = RK4(coasting_dynamics, t_current, state, dt, k_alpha, K_d, g, theta_wind_final, inertia)

            hoverslam_altitude = compute_hoverslam(np.abs(state[3]),3.48) # computed using A_NET = (F_avg - airframe_mass) / g

            max_tilt = max(max_tilt, np.abs(state[0]))
            v_current = v_initial

        return state, max_tilt
    else:
        raise ValueError("No Rocket Specified. Please specify a rocket object")


def sim_wrapper(
    Rocket,
    Motor_ascent,
    Motor_descent,
    initial_state,
    thrust_scale,
    gain_scale,
    inertia_scale,
):
    if Motor_descent is not None:
        sim_ascent, ascent_kp, ascent_kd, ascent_max_tilt = sim(
            motor=Motor_ascent,
            rocket=Rocket,
            inital_state=initial_state,
            thrust_scale=thrust_scale,
            gain_scale=gain_scale,
            inertia_scale=inertia_scale,
            phase="powered",
        
        )
        sim_coasting, max_tilt_coasting = sim_wind(
            sim_ascent[-1],   # last full state row: [theta, theta_dot, v, h]
            rocket=Rocket,
        )

        sim_descent, descent_kp, descent_kd, descent_max_tilt = sim(
            motor=Motor_descent,
            rocket=Rocket,
            inital_state=sim_coasting,
            thrust_scale=thrust_scale,
            gain_scale=gain_scale,
            inertia_scale=inertia_scale,
            phase="powered",
        )

        return sim_descent, descent_kp, descent_kd, ascent_max_tilt, descent_max_tilt
    else:
        sim_ascent, ascent_kp, ascent_kd, ascent_max_tilt = sim(
            motor=Motor_ascent,
            rocket=Rocket,
            inital_state=initial_state,
            thrust_scale=thrust_scale,
            gain_scale=gain_scale,
            inertia_scale=inertia_scale,
            phase="powered",
        
        )
        return sim_ascent, ascent_kp, ascent_kd, ascent_max_tilt





if __name__ == "__main__":

    # Monte Carlo simulation to estimate what state will look like after burn with noise

    rng = np.random.default_rng(seed=42)

    NUM_SIMULATIONS = 1500

    final_states = np.zeros((NUM_SIMULATIONS, 4))
    max_tilts = np.zeros(NUM_SIMULATIONS)
    kp_descent_list = []
    kd_descent_list = []

    for i in range(NUM_SIMULATIONS):
        initial_state = np.array(
            [
                rng.uniform(-0.087, 0.087),  # initial theta in rad
                rng.uniform(-0.5, 0.5),  # initial theta_dot in rad/s
                0.0, # Vertical Velocity, 
                0.0 # Vertical Altitude
            ]
        )

        thrust_scale = rng.uniform(0.9, 1.1)  # ±10%
        gain_scale = rng.uniform(0.9, 1.1)  # ±10%
        inertia_scale = rng.uniform(0.9, 1.1)  # ±10%

        if i == 0:

            print("==================")
            print("Starting Simulation")
            print("==================")

        trajectory, kp, kd, max_tilt_ascent, max_tilt_descent = sim_wrapper(
            Lotus,
            F15_A,
            F15_D,
            initial_state,
            thrust_scale=thrust_scale,
            gain_scale=gain_scale,
            inertia_scale=inertia_scale,
        )

        kp_descent_list.append(kp)
        kd_descent_list.append(kd)
        final_states[i] = trajectory[-2]  # intentionally storing second-to-last state
        max_tilts[i] = max_tilt_descent  # assumed in radians and specifically looking at the the landing angle

    final_angles_deg = np.rad2deg(np.abs(final_states[:, 0]))
    max_tilts_deg = np.rad2deg(np.abs(max_tilts))

    p95_final = np.percentile(final_angles_deg, 95)
    p99_final = np.percentile(final_angles_deg, 99)
    p95_max = np.percentile(max_tilts_deg, 95)
    p99_max = np.percentile(max_tilts_deg, 99)
    max_tilt_deg = np.max(max_tilts_deg)

    print(f"Kp Descent: {np.max(kp_descent_list)}\n Kd: {np.max(kd_descent_list)}")
    print(f"95th percentile final tilt: {p95_final:.2f}°")
    print(f"99th percentile final tilt: {p99_final:.2f}°")
    print(f"95th percentile max tilt:   {p95_max:.2f}°")
    print(f"99th percentile max tilt:   {p99_max:.2f}°")
    print(f"Maximum tilt:               {max_tilt_deg:.2f}°")

    # ================================
    # Plotting all the Data
    #=================================

    BG = "#071018"  # almost black navy
    AX_BG = "#0C1D2E"  # mission control blue
    TEXT = "#EAE7D6"  # off-white paper
    GRID = "#36516B"  # muted blue-gray
    CYAN = "#4FD5FF"  # CRT cyan
    AMBER = "#FFB347"  # terminal amber
    RED = "#FF6B6B"  # warning red

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.35,
            "font.family": "DejaVu Sans",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        "TVC MONTE CARLO DISPERSION ANALYSIS",
        color=TEXT,
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    for ax in axes:
        ax.set_facecolor(AX_BG)
        ax.grid(True, linewidth=0.8)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_color(TEXT)
            spine.set_linewidth(1.0)

    # Final tilt histogram
    ax = axes[0]
    ax.hist(
        final_angles_deg, bins=50, color=CYAN, edgecolor=TEXT, linewidth=0.6, alpha=0.88
    )

    # glow line + main percentile line
    ax.axvline(p95_final, color=RED, linewidth=6, alpha=0.16)
    ax.axvline(
        p95_final,
        color=RED,
        linestyle="--",
        linewidth=2.2,
        label=f"95th pct: {p95_final:.1f}°",
    )
    ax.axvline(p99_final, color=AMBER, linewidth=6, alpha=0.14)
    ax.axvline(
        p99_final,
        color=AMBER,
        linestyle="--",
        linewidth=2.2,
        label=f"99th pct: {p99_final:.1f}°",
    )

    ax.set_title(
        "FINAL TILT ANGLE DISTRIBUTION", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Final tilt angle (deg)")
    ax.set_ylabel("Count")

    ax.text(
        0.98,
        0.95,
        f"N = {NUM_SIMULATIONS}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(facecolor=BG, edgecolor=TEXT, boxstyle="round,pad=0.3", alpha=0.9),
    )

    leg = ax.legend(facecolor=AX_BG, edgecolor=TEXT, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # Max tilt histogram
    ax = axes[1]
    ax.hist(
        max_tilts_deg, bins=50, color=AMBER, edgecolor=TEXT, linewidth=0.6, alpha=0.88
    )

    ax.axvline(p95_max, color=CYAN, linewidth=6, alpha=0.16)
    ax.axvline(
        p95_max,
        color=CYAN,
        linestyle="--",
        linewidth=2.2,
        label=f"95th pct: {p95_max:.1f}°",
    )
    ax.axvline(max_tilt_deg, color=RED, linewidth=6, alpha=0.14)
    ax.axvline(
        max_tilt_deg,
        color=RED,
        linestyle="--",
        linewidth=2.2,
        label=f"Max: {max_tilt_deg:.1f}°",
    )

    ax.set_title("MAXIMUM TILT DURING BURN", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Max tilt during burn (deg)")
    ax.set_ylabel("Count")

    ax.text(
        0.98,
        0.95,
        f"N = {NUM_SIMULATIONS}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(facecolor=BG, edgecolor=TEXT, boxstyle="round,pad=0.3", alpha=0.9),
    )

    leg = ax.legend(facecolor=AX_BG, edgecolor=TEXT, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
   