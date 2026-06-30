"""
Barrowman-method aerodynamic coefficient estimator.

Computes C_Nalpha (normal force coefficient slope) and center of pressure (CP)
location for a rectangular-finned rocket, then derives the aerodynamic
weathercocking stiffness K_alpha = q * S * C_Nalpha * (CP - CG) used in the
coast-phase rotational dynamics:

    K_alpha(v) = 0.5 * rho * v**2 * S_ref * C_Nalpha_total * (CP - CG)

All linear inputs are in meters, areas in m^2, angles in radians unless noted.
Reference point for all "distance from nose" measurements is the nose tip.

References:
    Barrowman, J. "The Theoretical Prediction of the Center of Pressure",
    NARAM-8, 1966. (Standard rocketry CP/C_Nalpha method.)
"""

import numpy as np


# ============================================================
# 1. Geometry inputs 
# ============================================================

# --- Fins (rectangular, 4x) ---
N_FINS = 4
ROOT_CHORD = 0.005         # m (c_r)
TIP_CHORD = 0.005          # m (c_t)  -- rectangular fin, c_t == c_r
SEMI_SPAN = 0.144254         # m (s) -- span from body surface to fin tip
SWEEP_DIST = 0.0           # m (x_f) -- leading-edge sweep distance, root to tip
FIN_LE_FROM_NOSE = 0.008     # m -- distance from nose tip to fin root leading edge

# --- Body / airframe ---
AIRFRAME_DIAMETER = 0.102  # m -- body diameter at fin attachment (reference diameter)
AIRFRAME_LENGTH = 0.389    # m
NOSE_LENGTH = 0.03442      # m
 
# --- Nose cone shape: one of "conical", "ogive", "elliptical", or None ---
# Affects the nose's own (small) C_Nalpha contribution and its CP location.
# Set HAS_NOSE = False if the airframe has no nose cone (e.g. a flat fin hub
# at the top, like an open-truss design) -- in that case the nose contributes
# ZERO C_Nalpha and ZERO CP weight, rather than faking a shape/length.
HAS_NOSE = False
NOSE_SHAPE = "conical"
 
# --- Atmospheric / flight condition (for a given instant; recompute per timestep) ---
RHO_AIR = 1.225            # kg/m^3, sea-level standard (adjust for altitude if needed)
 
 
# ============================================================
# 2. Derived geometry
# ============================================================
 
def reference_area(diameter):
    """Body cross-sectional reference area, S_ref = pi * (d/2)^2."""
    return np.pi * (diameter / 2.0) ** 2
 
 
def fin_cnalpha_single(root_chord, tip_chord, semi_span, sweep_dist, body_radius):
    """
    Barrowman normal-force-slope contribution for ONE fin, referenced to the
    BODY reference area (so it can be summed directly with the body/nose terms
    and multiplied by S_ref later).
 
    This is the standard Barrowman fin term:
 
        (C_Nalpha)_fin = (4 * N * (s/d)^2) / (1 + sqrt(1 + (2*l_m / (c_r + c_t))^2))
 
    but for a single fin (N=1 folded into the per-fin term here; multiply by
    N_FINS outside), where:
        s      = semi-span (fin span, body surface to tip)
        d      = body diameter at fin root
        l_m    = mid-chord sweep length = sweep_dist + 0.5*tip_chord - 0.5*root_chord
        c_r,c_t= root/tip chord
 
    Returns a dimensionless slope contribution per fin (already normalized by
    body reference area, NOT fin area).
    """
    d = 2.0 * body_radius
    l_m = sweep_dist + 0.5 * tip_chord - 0.5 * root_chord
 
    numerator = 4.0 * (semi_span / d) ** 2
    denominator = 1.0 + np.sqrt(1.0 + (2.0 * l_m / (root_chord + tip_chord)) ** 2)
 
    return numerator / denominator
 
 
def fin_cp_from_root_le(root_chord, tip_chord, semi_span, sweep_dist):
    """
    Distance from the fin's ROOT leading edge to the fin's own CP (chordwise),
    per Barrowman:
 
        X_f = (sweep_dist * (c_r + 2*c_t)) / (3 * (c_r + c_t)) + (1/6) * (c_r + c_t - (c_r*c_t)/(c_r+c_t))
 
    For a rectangular fin (c_r == c_t, sweep_dist == 0) this reduces to the
    chord midpoint, c_r / 2, as expected.
    """
    term1 = (sweep_dist * (root_chord + 2.0 * tip_chord)) / (3.0 * (root_chord + tip_chord))
    term2 = (1.0 / 6.0) * (
        root_chord + tip_chord - (root_chord * tip_chord) / (root_chord + tip_chord)
    )
    return term1 + term2
 
 
def nose_cnalpha(has_nose=True):
    """
    Nose cone contribution to C_Nalpha, referenced to body reference area.
    Barrowman result: (C_Nalpha)_nose = 2 for an actual nose cone
    (independent of shape, in this normalization -- shape only affects the
    nose's CP location, not its C_Nalpha slope).
 
    If the airframe has no nose cone at all (e.g. a flat hub/fin-mount at
    the top with no pointed forebody), this contribution is physically zero
    -- there's no lifting surface there to generate it.
    """
    return 2.0 if has_nose else 0.0
 
 
def nose_cp_from_tip(nose_length, shape="conical"):
    """
    Distance from the nose tip to the nose's own CP, depending on nose shape.
    Standard Barrowman results:
        conical:     0.666 * L_nose
        ogive:       0.466 * L_nose
        elliptical:  0.500 * L_nose
    """
    shape = shape.lower()
    if shape == "conical":
        return 0.666 * nose_length
    elif shape == "ogive":
        return 0.466 * nose_length
    elif shape == "elliptical":
        return 0.500 * nose_length
    else:
        raise ValueError(f"Unknown nose shape '{shape}'. Use conical, ogive, or elliptical.")
 
 
# ============================================================
# 3. Combine into total C_Nalpha and CP location
# ============================================================
 
def compute_total_cnalpha_and_cp():
    """
    Returns (C_Nalpha_total, CP_from_nose_m).
 
    C_Nalpha_total is dimensionless, normalized by body reference area S_ref
    (i.e. use it directly in K_alpha = q * S_ref * C_Nalpha_total * (CP - CG)).
 
    CP_from_nose_m is the overall center of pressure location, measured from
    the nose tip, found via the weighted average of each component's CP
    location weighted by its own C_Nalpha contribution (standard Barrowman
    CP combination rule).
    """
    body_radius = AIRFRAME_DIAMETER / 2.0
 
    # --- Nose contribution (zero if there's no actual nose cone) ---
    cna_nose = nose_cnalpha(has_nose=HAS_NOSE)
    cp_nose = nose_cp_from_tip(NOSE_LENGTH, NOSE_SHAPE) if HAS_NOSE else 0.0
 
    # --- Fin contribution (all N fins together) ---
    cna_fin_single = fin_cnalpha_single(
        ROOT_CHORD, TIP_CHORD, SEMI_SPAN, SWEEP_DIST, body_radius
    )
    cna_fins_total = N_FINS * cna_fin_single
 
    fin_cp_chordwise = fin_cp_from_root_le(ROOT_CHORD, TIP_CHORD, SEMI_SPAN, SWEEP_DIST)
    cp_fins = FIN_LE_FROM_NOSE + fin_cp_chordwise
 
    # --- Body tube contribution: ~0 for a straight cylindrical body at small
    #     angle of attack (Barrowman assumes zero unless there's a boattail
    #     or significant body lift, which we neglect here) ---
    cna_body = 0.0
    cp_body = 0.0  # unused since cna_body == 0
 
    # --- Total C_Nalpha: simple sum of component slopes ---
    cna_total = cna_nose + cna_body + cna_fins_total
 
    # --- Total CP: weighted average by each component's C_Nalpha contribution ---
    cp_total = (
        cna_nose * cp_nose + cna_body * cp_body + cna_fins_total * cp_fins
    ) / cna_total
 
    return cna_total, cp_total
 
 
# ============================================================
# 4. K_alpha and K_d (damping) computation
# ============================================================
 
def compute_k_alpha(velocity, cg_from_nose, cna_total=None, cp_total=None, rho=RHO_AIR):
    """
    Aerodynamic weathercocking stiffness at a given instant.
 
    K_alpha = q * S_ref * C_Nalpha_total * (CP - CG)
 
    velocity      : current airspeed magnitude (m/s) -- drives dynamic pressure q
    cg_from_nose  : current CG location from nose tip (m) -- changes over burn,
                    pull this from your Rocket/Motor compute_dist_to_COM at
                    the relevant timestep
    cna_total/cp_total : pass these in if precomputed once (geometry is fixed,
                    so you should compute these ONCE outside any time loop,
                    not recompute every timestep)
    """
    if cna_total is None or cp_total is None:
        cna_total, cp_total = compute_total_cnalpha_and_cp()
 
    S_ref = reference_area(AIRFRAME_DIAMETER)
    q = 0.5 * rho * velocity ** 2
 
    static_margin = cp_total - cg_from_nose  # meters; positive = stable (CP aft of CG)
 
    k_alpha = q * S_ref * cna_total * static_margin
    return k_alpha, static_margin
 
 
def compute_k_d(inertia, k_alpha, zeta=0.7):
    """
    Synthetic damping coefficient chosen to achieve a target damping ratio
    zeta for the coast-phase restoring oscillation:
 
        K_d = 2 * zeta * sqrt(I * K_alpha)
 
    zeta in [0.5, 1.0] per your earlier note; 0.7 is a reasonable default
    (near-critical without overshoot risk). K_alpha must be >= 0 (statically
    stable) for this to produce a real K_d -- a negative K_alpha means the
    rocket is statically UNSTABLE at that instant and this damping model
    doesn't apply.
    """
    if k_alpha < 0:
        raise ValueError(
            "K_alpha is negative (CP ahead of CG) -- rocket is statically "
            "unstable at this condition. Check CG location or fin sizing."
        )
    return 2.0 * zeta * np.sqrt(inertia * k_alpha)
 
 
# ============================================================
# 5. Example usage / sanity check
# ============================================================
 
if __name__ == "__main__":
    cna_total, cp_total = compute_total_cnalpha_and_cp()
 
    print("=" * 50)
    print("BARROWMAN AERO ESTIMATE")
    print("=" * 50)
    print(f"C_Nalpha (total, per rad):     {cna_total:.4f}")
    print(f"CP location from nose:         {cp_total*1000:.2f} mm")
    print(f"Reference area (S_ref):        {reference_area(AIRFRAME_DIAMETER)*1e4:.2f} cm^2")
    print()
 
    # Example: evaluate K_alpha at a representative descent speed.
    # Replace these with real values pulled from your sim at each timestep.
    example_velocity = 15.0       # m/s, representative coast/descent speed
    example_cg_from_nose = 0.20   # m, placeholder -- pull from compute_dist_to_COM
    example_inertia = 0.05        # kg*m^2, placeholder -- pull from compute_inertia
 
    k_alpha, static_margin = compute_k_alpha(
        example_velocity, example_cg_from_nose, cna_total, cp_total
    )
    print(f"--- Example at v = {example_velocity} m/s, CG = {example_cg_from_nose*1000:.1f} mm from nose ---")
    print(f"Static margin (CP - CG):        {static_margin*1000:.2f} mm")
    print(f"K_alpha:                        {k_alpha:.6f} N*m/rad")
 
    try:
        k_d = compute_k_d(example_inertia, k_alpha, zeta=0.7)
        print(f"K_d (zeta=0.7):                  {k_d:.6f} N*m*s/rad")
    except ValueError as e:
        print(f"K_d: SKIPPED -- {e}")