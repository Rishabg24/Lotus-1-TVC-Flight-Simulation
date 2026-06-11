import numpy as np
import matplotlib.pyplot as plt
from utilities.read_config import data
from utilities.mathematicals import t_array
from utilities.mathematicals import compute_LQR, precompute_plant_gains

MASS_DRY = data["Rocket_1"]["dimensions_weight"][
    "dry_mass"
]  # Mass of empty rocket motor (kg)
AIRFRAME_LENGTH = data["Rocket_1"]["dimensions_weight"]["airframe_length_m"]  # (m)
AIRFRAME_MASS = data["Rocket_1"]["dimensions_weight"]["airframe_mass_kg"]  # (kg)

INITIAL_STATE = np.array([0.087, 0.0])  # Initial state: [theta (rad), theta_dot (rad/s)]

sigma_theta = np.deg2rad(0.5)
sigma_omega = np.deg2rad(1.5)


def dynamics(t, x, K, g):

    theta_dot = x[1]

    u = float((-K @ x).item())

    u = np.clip(u, -np.deg2rad(20), np.deg2rad(20)) # Max Gimbal angle for rocket

    theta_ddot = g * u

    return np.array([theta_dot, theta_ddot])


def RK4(f, t, x, h, K, g):

    k1 = f(t, x, K, g)
    k2 = f(t + h / 2, x + h / 2 * k1, K, g)
    k3 = f(t + h / 2, x + h / 2 * k2, K, g)
    k4 = f(t + h, x + h * k3, K, g)

    x_next = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


def sim(inital_state=INITIAL_STATE, sigma_theta=sigma_theta, sigma_omega=sigma_omega, thrust_scale = 1.0, gain_scale = 1.0, inertia_scale = 1.0):
    # Pre-compute plant gains once using vectorized operations (optimized for Monte Carlo)

    g_array = precompute_plant_gains(MASS_DRY, AIRFRAME_MASS, AIRFRAME_LENGTH)
    g_scaled = g_array * thrust_scale * gain_scale / inertia_scale

    LQR_K_array = np.zeros((len(t_array), 1, 2))  # Pre-allocate array for LQR gains
    trajectory = np.zeros((len(t_array), 2))
    x = inital_state
    trajectory[0] = x
    max_tilt = 0.0

    Q = np.diag([100.0, 10.0])  # [theta penalty, theta_dot penalty]
    R = np.array([[1.0]])

    # Now iterate through timesteps using pre-computed gains
    for i in range(1, len(t_array)):  

        # NOTE: state is 2 dimensional: [theta, theta_dot]
        g = g_scaled[i]

        if np.abs(g) <= 1e-6:
            continue

        LQR_K = compute_LQR(g, Q, R)
        LQR_K_array[i] = LQR_K

        h = t_array[i] - t_array[i-1]   # always positive
        y = RK4(dynamics, t_array[i-1], x, h, LQR_K, g)

        max_tilt = max(max_tilt, np.abs(y[0]))

        x = np.array([y[0] + np.random.normal(0, sigma_theta),
                    y[1] + np.random.normal(0, sigma_omega)])
        trajectory[i] = y

    KD = LQR_K_array[:, 0, 1]  # Extract the derivative gain (K_d) for plotting
    KP = LQR_K_array[:, 0, 0]  # Extract the proportional gain (K_p) for plotting

    return trajectory, KD, KP, max_tilt

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

BG     = "#071018"   # almost black navy
AX_BG  = "#0C1D2E"   # mission control blue
TEXT   = "#EAE7D6"   # off-white paper
GRID   = "#36516B"   # muted blue-gray
CYAN   = "#4FD5FF"   # CRT cyan
AMBER  = "#FFB347"   # terminal amber
RED    = "#FF6B6B"   # warning red

if __name__ == "__main__":
    # Monte Carlo simulation to estimate what state will look like after burn time with noise
    rng = np.random.default_rng(seed=42)

    NUM_SIMULATIONS = 1500
    final_states = np.zeros((NUM_SIMULATIONS, 2))
    max_tilts = np.zeros(NUM_SIMULATIONS)

    for i in range(NUM_SIMULATIONS):
        initial_state = np.array([
            rng.uniform(-0.087, 0.087),   # initial theta in rad
            rng.uniform(-0.5, 0.5)         # initial theta_dot in rad/s
        ])

        thrust_scale = rng.uniform(0.9, 1.1)    # ±10%
        gain_scale = rng.uniform(0.9, 1.1)      # ±10%
        inertia_scale = rng.uniform(0.9, 1.1)   # ±10%

        if i == 0:

            print("==================")
            print("Running Simulation")
            print("==================")

        trajectory, _, _, max_tilt = sim(
            inital_state = initial_state,
            thrust_scale = thrust_scale,
            gain_scale = gain_scale,
            inertia_scale = inertia_scale
        )

        final_states[i] = trajectory[-2]   # intentionally storing second-to-last state
        max_tilts[i] = max_tilt            # assumed in radians

    final_angles_deg = np.rad2deg(np.abs(final_states[:, 0]))
    max_tilts_deg = np.rad2deg(np.abs(max_tilts))

    p95_final = np.percentile(final_angles_deg, 95)
    p99_final = np.percentile(final_angles_deg, 99)
    p95_max = np.percentile(max_tilts_deg, 95)
    p99_max = np.percentile(max_tilts_deg, 99)
    max_tilt_deg = np.max(max_tilts_deg)

    print(f"95th percentile final tilt: {p95_final:.2f}°")
    print(f"99th percentile final tilt: {p99_final:.2f}°")
    print(f"95th percentile max tilt:   {p95_max:.2f}°")
    print(f"99th percentile max tilt:   {p99_max:.2f}°")
    print(f"Maximum tilt:               {max_tilt_deg:.2f}°")

    plt.style.use("default")
    plt.rcParams.update({
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
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)
    fig.patch.set_facecolor(BG)

    fig.suptitle(
        "TVC MONTE CARLO DISPERSION ANALYSIS",
        color=TEXT,
        fontsize=16,
        fontweight="bold",
        y=0.98
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
        final_angles_deg,
        bins=50,
        color=CYAN,
        edgecolor=TEXT,
        linewidth=0.6,
        alpha=0.88
    )

    # glow line + main percentile line
    ax.axvline(p95_final, color=RED, linewidth=6, alpha=0.16)
    ax.axvline(p95_final, color=RED, linestyle="--", linewidth=2.2, label=f"95th pct: {p95_final:.1f}°")
    ax.axvline(p99_final, color=AMBER, linewidth=6, alpha=0.14)
    ax.axvline(p99_final, color=AMBER, linestyle="--", linewidth=2.2, label=f"99th pct: {p99_final:.1f}°")

    ax.set_title("FINAL TILT ANGLE DISTRIBUTION", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Final tilt angle (deg)")
    ax.set_ylabel("Count")

    ax.text(
        0.98, 0.95,
        f"N = {NUM_SIMULATIONS}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(facecolor=BG, edgecolor=TEXT, boxstyle="round,pad=0.3", alpha=0.9)
    )

    leg = ax.legend(facecolor=AX_BG, edgecolor=TEXT, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(TEXT)

    # Max tilt histogram
    ax = axes[1]
    ax.hist(
        max_tilts_deg,
        bins=50,
        color=AMBER,
        edgecolor=TEXT,
        linewidth=0.6,
        alpha=0.88
    )

    ax.axvline(p95_max, color=CYAN, linewidth=6, alpha=0.16)
    ax.axvline(p95_max, color=CYAN, linestyle="--", linewidth=2.2, label=f"95th pct: {p95_max:.1f}°")
    ax.axvline(max_tilt_deg, color=RED, linewidth=6, alpha=0.14)
    ax.axvline(max_tilt_deg, color=RED, linestyle="--", linewidth=2.2, label=f"Max: {max_tilt_deg:.1f}°")

    ax.set_title("MAXIMUM TILT DURING BURN", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Max tilt during burn (deg)")
    ax.set_ylabel("Count")

    ax.text(
        0.98, 0.95,
        f"N = {NUM_SIMULATIONS}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color=TEXT,
        bbox=dict(facecolor=BG, edgecolor=TEXT, boxstyle="round,pad=0.3", alpha=0.9)
    )

    leg = ax.legend(facecolor=AX_BG, edgecolor=TEXT, framealpha=0.95)
    for t in leg.get_texts():
        t.set_color(TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()