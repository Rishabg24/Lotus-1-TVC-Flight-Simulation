# Self-Lander LQR Simulation

A Python-based simulation for controlling the Lotus 1 self landing TVC rocket using Linear Quadratic Regulator (LQR) optimal control. The simulation models the dynamics of a small rocket during powered descent and computes optimal gimbal angles to maintain vertical stability.

## Overview

This simulator models the attitude dynamics of a self-landing rocket equipped with a gimbaled thrust vector control (TVC) system. The control law is computed in real-time using LQR, which generates optimal gimbal commands based on the current orientation (tilt angle) and angular rate. The system includes thrust modeling from motor burn data, mass dynamics as propellant is consumed, and sensor noise simulation.

## Key Features

- **LQR Control**: Optimal feedback control law computed at each timestep accounting for varying dynamics
- **Realistic Thrust Modeling**: Interpolated thrust curve from motor data with propellant mass depletion
- **TVC Limits**: Gimbal angle constraints (±20°) representing physical hardware limits
- **Noise Simulation**: Gaussian noise on angle and angular rate measurements
- **Scalable Parameters**: Thrust, gain, and inertia scaling for robustness analysis

## Project Structure

```
├── lqr_sim.py                    # Main simulation engine
├── config.json                   # Rocket and motor specifications
├── motor_thrust_curve.csv        # F15 motor thrust vs. time data
├── Estes_F15.eng                 # Motor engine file
├── utilities/
│   ├── read_config.py            # Configuration loader
│   ├── mathematicals.py          # LQR, dynamics, and mass calculations
│   └── convert_eng_to_csv.py     # Motor data format conversion
```

## Dependencies

- **numpy** - Numerical computations
- **scipy** - ODE integration and LQR gain computation
- **pandas** - Data processing for motor thrust curves
- **matplotlib** - Trajectory visualization

## Configuration

The `config.json` file specifies:
- **Motor specs**: Estes F15 (thrust profile, specific impulse, burn time)
- **Rocket dimensions**: Airframe length, mass, CG location
- **Propellant properties**: Mass, center of gravity, mass fraction

Motor thrust data is loaded from `motor_thrust_curve.csv` (time vs. thrust in Newtons).

## Usage

```python
from lqr_sim import sim

# Run simulation with default parameters
trajectory, max_tilt, KD, t_array = sim()

# Run with custom scaling factors
trajectory, max_tilt, KD, t_array = sim(
    thrust_scale=1.0,      # Thrust multiplier
    gain_scale=1.0,        # LQR gain multiplier
    inertia_scale=1.0      # Moment of inertia multiplier
)
```

The simulation returns:
- `trajectory`: State history [theta, theta_dot] over time.
- `max_tilt`: Maximum tilt angle reached (radians)
- `KD`: Derivative gain schedule over time
- `t_array`: Time vector


## Control Algorithm

The LQR controller solves the continuous-time algebraic Riccati equation at each timestep:

$$K(t) = R^{-1}B^T P(t)$$

where the system has:
- **State**: [θ, θ̇] (tilt angle and angular rate)
- **Input**: u (gimbal angle)
- **Cost weights**: Q = diag([100, 10]), R = [1]

Gimbal commands are clipped to ±20° to respect hardware constraints.

## Files Explained

- **lqr_sim.py**: Core simulation with dynamics (RK4 integration), LQR control, and noise injection
- **mathematicals.py**: Thrust curve interpolation, mass depletion tracking, plant gain scheduling, and LQR gain computation
- **read_config.py**: JSON configuration loader for motor and rocket properties
- **convert_eng_to_csv.py**: Utility to convert Estes .eng motor files to CSV format
