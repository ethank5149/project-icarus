import numpy as np


def newtonian_cd_exo(mach, alpha, beta):
    """Newtonian impact theory for exo-atmospheric drag coefficient."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    aoa_eff = np.sqrt(alpha_rad**2 + beta_rad**2)
    cd = 2.0 * np.sin(aoa_eff) ** 2
    cd = np.clip(cd, 0.0, 2.0)
    return cd


def newtonian_sideforce_moments_exo(mach, alpha, beta):
    """Newtonian side force and moments for exo-atmospheric regime."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    cy = 2.0 * np.sin(alpha_rad) * np.cos(alpha_rad)
    cn = 2.0 * np.sin(beta_rad) * np.cos(beta_rad)
    cl_roll = 0.0
    return cy, cn, cl_roll


def linear_viscous_endo(mach, alpha, beta, ref_area=0.1, ref_length=1.0, rho=1.225, viscosity=1.78e-5):
    """Linear + viscous aero model for endo-atmospheric regime."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)

    Re = rho * mach * ref_length / max(viscosity, 1e-10)
    Re = np.maximum(Re, 10.0)
    logRe = np.log10(Re)
    Cf = np.where(np.abs(logRe) > 1e-12, 0.455 / (logRe ** 2.58), 0.0)

    S_wet = np.pi * ref_length * np.sqrt(ref_area / np.pi) * 2.0
    Cd_friction = Cf * (S_wet / ref_area)

    # Compressibility correction (vectorized over mach array).
    mach = np.asarray(mach, dtype=float)
    beta_pg = np.ones_like(mach)
    sub = mach < 0.6
    pg = (mach >= 0.6) & (mach < 0.7)
    kt = mach >= 0.7
    beta_pg = np.where(pg, np.sqrt(np.maximum(1.0 - mach**2, 1e-6)), beta_pg)
    b_kt = (mach**2 + 2.0) / (mach**2 * np.sqrt(np.maximum(1.0 - 0.2 * mach**2, 1e-6)) + 2.0)
    beta_pg = np.where(kt, b_kt, beta_pg)
    beta_pg = np.where(sub, 1.0, beta_pg)
    Cd_friction = Cd_friction / np.where(beta_pg > 1e-6, beta_pg, 1.0)

    Cd_induced = (np.deg2rad(alpha) ** 2 + np.deg2rad(beta) ** 2)
    Cd = Cd_friction + Cd_induced
    Cd = np.clip(Cd, 0.0, 2.0)

    Cl_lin = 2.0 * np.pi * alpha_rad
    Cl_visc = -0.1 * Cf * np.sign(alpha_rad)
    cl_side = Cl_lin + Cl_visc

    Cm_lin = -1.2 * alpha_rad
    Cm_visc = 0.05 * Cf * np.sign(alpha_rad)
    cm_pitch = Cm_lin + Cm_visc

    cn_yaw = 2.0 * np.pi * beta_rad
    cl_roll = 0.0

    return Cd, cl_side, cm_pitch, cn_yaw, cl_roll


def blended_aero(mach, alpha, beta, altitude, boundary_alt=100e3, taper_width=5e3):
    """Blend endo/exo aero with smooth taper around boundary altitude."""
    taper_low = boundary_alt - taper_width
    taper_high = boundary_alt + taper_width
    blend = np.clip((altitude - taper_low) / (taper_high - taper_low), 0.0, 1.0)
    blend = 0.5 * (1.0 - np.cos(np.pi * blend))

    cd_exo = newtonian_cd_exo(mach, alpha, beta)
    cy_exo, cn_exo, cl_roll_exo = newtonian_sideforce_moments_exo(mach, alpha, beta)

    Cd_endo, cy_endo, cm_endo, cn_endo, cl_roll_endo = linear_viscous_endo(mach, alpha, beta)

    cd_exo_arr = np.full_like(Cd_endo, cd_exo)
    cy_exo_arr = np.full_like(cy_endo, cy_exo)
    cm_exo_arr = np.full_like(cm_endo, 0.0)
    cn_exo_arr = np.full_like(cn_endo, cn_exo)
    cl_roll_exo_arr = np.full_like(cl_roll_endo, cl_roll_exo)

    Cd = Cd_endo * (1.0 - blend) + cd_exo_arr * blend
    Cy = cy_endo * (1.0 - blend) + cy_exo_arr * blend
    Cm = cm_endo * (1.0 - blend) + cm_exo_arr * blend
    Cn = cn_endo * (1.0 - blend) + cn_exo_arr * blend
    Cl_roll = cl_roll_endo * (1.0 - blend) + cl_roll_exo_arr * blend

    return Cd, Cy, Cm, Cn, Cl_roll
