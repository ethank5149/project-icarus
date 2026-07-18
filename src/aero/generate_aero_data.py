import h5py
import numpy as np

def generate_fenics_data(filename="aero_data.h5"):
    """
    Simulates high-fidelity CFD results from a FEniCS/DOLFINx solver.
    In practice, solve your Navier-Stokes equations here.
    """
    mach = np.linspace(0.5, 5.0, 50)
    alpha = np.linspace(0, 15, 20)
    
    # Create meshgrid
    M, A = np.meshgrid(mach, alpha)
    
    # Synthetic CFD output (Coefficient generation)
    Cd = 0.05 + 0.1 * M**2 + 0.01 * A
    Cl = 0.5 * A - 0.02 * M
    
    with h5py.File(filename, 'w') as hf:
        hf.create_dataset('mach', data=M.flatten())
        hf.create_dataset('alpha', data=A.flatten())
        hf.create_dataset('Cd', data=Cd.flatten())
        hf.create_dataset('Cl', data=Cl.flatten())
    print(f"Dataset generated: {filename}")

generate_fenics_data()