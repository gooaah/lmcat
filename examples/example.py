"""
Example: generate a Cu(111) slab NVT trajectory with ASE + EMT and compute
the Gibbs dividing surface position.

Workflow:
  1. Build a Cu(111) slab.
  2. Run an NVT MD simulation with EMT to generate a trajectory.
  3. Histogram + Gaussian smoothing -> density profile.
  4. erfc fit -> Gibbs interface position.
  5. Plot results.
"""
import numpy as np
import matplotlib.pyplot as plt

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

import lmcat

# --- 1. Build Cu(111) slab ---
atoms = fcc111('Cu', size=(4, 4, 6), vacuum=30.0)
atoms.calc = EMT()

# --- 2. NVT MD simulation (Langevin thermostat, T=700 K) ---
T = 1600  # K
MaxwellBoltzmannDistribution(atoms, temperature_K=T)

dyn = Langevin(atoms, timestep=5.0 * units.fs, temperature_K=T, friction=0.01)

traj = []

def collect():
    traj.append(atoms.copy())

print("Running MD")

dyn.attach(collect, interval=10)
dyn.run(2000)   

traj = traj[100:]

print(f"Collected {len(traj)} frames")

# --- 3. Parameters ---
cell = traj[0].get_cell()
z_max = cell[2, 2]
hist_range = (0.0, z_max)
bins = int(z_max / 0.1)          # resolution ~0.1 Angstrom per bin
gauss_width = 1.0                 # Gaussian smoothing width [Angstrom]
w_bin = z_max / bins

# --- 4. Histogram + smoothing ---
zs, cu_hist = lmcat.trajectory_histogram(traj, hist_range, bins, slab_sym='Cu')
cu_density = lmcat.smooth_density(cu_hist, gauss_width, w_bin)

# --- 5. Locate the top interface (Gibbs dividing surface) ---
# The top surface of a Cu(111) slab lies roughly above z_max / 2
z_mid = z_max / 2
sample_range = np.linspace(z_mid, z_max - 2.0, 300)
interface_z, fit_curve = lmcat.fit_interface(
    cu_density, slab_sigma=gauss_width, sample_range=sample_range, zs=zs
)
print(f"Gibbs dividing surface (top) at z = {interface_z:.3f} Angstrom")

# --- 6. Fit double erfc to extract slab thickness ---
result = lmcat.fit_slab_interfaces(zs, cu_density)
print(f"Slab thickness: {result['z2'] - result['z1']:.3f} Angstrom")
print(f"Interface width (sigma): {result['sigma']:.3f} Angstrom")
print(f"Bulk density (rho): {result['rho']:.4f} atoms/Angstrom^3")

# --- 7. Plot ---
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(zs, cu_density, label='Cu density profile')
ax.plot(fit_curve[:, 0], fit_curve[:, 1], '--', label='erfc fit (top surface)')
ax.axvline(interface_z, color='red', linestyle=':', label=f'Interface z = {interface_z:.2f} A')
ax.set_xlabel('z [Angstrom]')
ax.set_ylabel('Density [atoms/Angstrom^3]')
ax.set_title('Cu slab - Gibbs dividing surface')
ax.legend()
plt.tight_layout()
plt.savefig('interface_profile.png', dpi=150)
print("Plot saved to interface_profile.png")
