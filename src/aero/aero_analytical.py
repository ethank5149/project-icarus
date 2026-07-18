import numpy as np


def newtonian_cd_exo(mach, alpha, beta):
    """Newtonian impact theory for exo-atmospheric drag coefficient."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    aoa_eff = np.sqrt(alpha_rad**2 + beta_rad**2)
    cd = 2.0 * np.sin(aoa_eff) ** 2
    cd = np.clip(cd, 0.0, 2.0)
    return cd


def newtonian_cl_exo(mach, alpha, beta):
    """Newtonian lift/side-force coefficient for exo-atmospheric regime."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    cl = 2.0 * np.sin(alpha_rad) * np.cos(alpha_rad)
    cm = 2.0 * np.sin(beta_rad) * np.cos(beta_rad)
    return cl, cm


def linear_viscous_endo(mach, alpha, beta, ref_area=0.1, ref_length=1.0, rho=1.225, viscosity=1.78e-5):
    """Linear + viscous aero model for endo-atmospheric regime."""
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)

    Re = rho * mach * ref_length / max(viscosity, 1e-10)
    Cf = 0.455 / (np.log10(Re) ** 2.58)

    Cd_par = Cf * (ref_area / ref_length**2)
    Cd_induced = (np.deg2rad(alpha) ** 2 + np.deg2rad(beta) ** 2)
    Cd = Cd_par + Cd_induced
    Cd = np.clip(Cd, 0.0, 2.0)

    Cl_lin = 2.0 * np.pi * alpha_rad
    Cl_visc = -0.1 * Cf * np.sign(alpha_rad)
    Cl = Cl_lin + Cl_visc

    Cm_lin = -1.2 * alpha_rad
    Cm_visc = 0.05 * Cf * np.sign(alpha_rad)
    Cm = Cm_lin + Cm_visc

    return Cd, Cl, Cm


def blended_aero(mach, alpha, beta, altitude, boundary_alt=100e3, taper_width=5e3):
    """Blend endo/exo aero with smooth taper around boundary altitude."""
    taper_low = boundary_alt - taper_width
    taper_high = boundary_alt + taper_width
    blend = np.clip((altitude - taper_low) / (taper_high - taper_low), 0.0, 1.0)
    blend = 0.5 * (1.0 - np.cos(np.pi * blend))

    cd_exo = newtonian_cd_exo(mach, alpha, beta)
    cl_exo, cm_exo = newtonian_cl_exo(mach, alpha, beta)

    Cd, Cl, Cm = linear_viscous_endo(mach, alpha, beta)
    cd_exo_arr = np.full_like(Cd, cd_exo)
    cl_exo_arr = np.full_like(Cl, cl_exo)
    cm_exo_arr = np.full_like(Cm, cm_exo)

    Cd = Cd * (1.0 - blend) + cd_exo_arr * blend
    Cl = Cl * (1.0 - blend) + cl_exo_arr * blend
    Cm = Cm * (1.0 - blend) + cm_exo_arr * blend

    return Cd, Cl, Cm
