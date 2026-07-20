import h5py
import numpy as np
from .aero_analytical import blended_aero


def generate_aero_data(
    filename="aero_data.h5",
    mach_range=(0.5, 5.0, 50),
    alpha_range=(0.0, 15.0, 20),
    beta_range=(-5.0, 5.0, 11),
    boundary_alt=100e3,
    taper_width=5e3,
    noise_level=0.02,
):
    """
    Generate synthetic aero data for Mach, alpha, beta, and altitude.
    Stores Cd, Cy, Cm with regime flags and per-point uncertainty bounds in HDF5.
    """
    mach = np.linspace(mach_range[0], mach_range[1], mach_range[2])
    alpha = np.linspace(alpha_range[0], alpha_range[1], alpha_range[2])
    beta = np.linspace(beta_range[0], beta_range[1], beta_range[2])

    Mach, Alpha, Beta = np.meshgrid(mach, alpha, beta, indexing="ij")

    n_pts = Mach.size
    alt = np.linspace(0, 150e3, n_pts)

    Cd, Cy, Cm, Cn, Cl_roll = blended_aero(Mach.flatten(), Alpha.flatten(), Beta.flatten(), alt, boundary_alt, taper_width)

    Cd += np.random.normal(0.0, noise_level, size=n_pts)
    Cy += np.random.normal(0.0, noise_level, size=n_pts)
    Cm += np.random.normal(0.0, noise_level * 0.5, size=n_pts)

    Cd = np.clip(Cd, 0.0, 2.0)
    sigma = np.full((n_pts, 3), noise_level, dtype=float)

    with h5py.File(filename, "w") as hf:
        hf.create_dataset("mach", data=Mach.flatten())
        hf.create_dataset("alpha", data=Alpha.flatten())
        hf.create_dataset("beta", data=Beta.flatten())
        hf.create_dataset("altitude", data=alt)
        hf.create_dataset("Cd", data=Cd)
        hf.create_dataset("Cy", data=Cy)
        hf.create_dataset("Cm", data=Cm)
        hf.create_dataset("sigma", data=sigma)
        regime = np.where(alt < boundary_alt, 0, 1).astype(np.int8)
        hf.create_dataset("regime", data=regime)
        hf.attrs["boundary_alt"] = boundary_alt
        hf.attrs["taper_width"] = taper_width

    print(f"Dataset generated: {filename} ({n_pts} samples, 3 outputs)")


if __name__ == "__main__":
    generate_aero_data()
