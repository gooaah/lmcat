"""
Example: fit a symmetric double-erf model to a Cu(111) slab density profile
and extract slab thickness and interface positions.

Workflow:
  1. Build a Cu(111) slab and run NVT MD with EMT.
  2. Histogram + Gaussian smoothing -> density profile.
  3. fit_slab_interfaces -> lower/upper interface positions, slab thickness.
  4. Plot density profile with double-erf fit overlay.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

import lmcat

# --- 1. Build Cu(111) slab and run MD ---
atoms = fcc111('Cu', size=(4, 4, 6), vacuum=30.0)
atoms.calc = EMT()

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

# --- 2. Histogram + smoothing ---
cell = traj[0].get_cell()
z_max = cell[2, 2]
hist_range = (0.0, z_max)
bins = int(z_max / 0.1)
gauss_width = 1.0  # Angstrom
w_bin = z_max / bins

zs, cu_hist = lmcat.trajectory_histogram(traj, hist_range, bins, slab_sym='Cu')
cu_density = lmcat.smooth_density(cu_hist, gauss_width, w_bin)

# --- 3. Fit double-erf to get slab thickness ---
# sigma=None lets the fit optimize the interface width
result = lmcat.fit_slab_interfaces(zs, cu_density)

z1 = result['z1']
z2 = result['z2']
rho = result['rho']
sigma = result['sigma']
thickness = z2 - z1

print(f"Lower interface (z1): {z1:.3f} Angstrom")
print(f"Upper interface (z2): {z2:.3f} Angstrom")
print(f"Slab thickness:       {thickness:.3f} Angstrom")
print(f"Interface width:      {sigma:.3f} Angstrom")
print(f"Bulk density:         {rho:.4f} atoms/Angstrom^3")

# With fixed sigma
result_fixed = lmcat.fit_slab_interfaces(zs, cu_density, sigma=gauss_width)
print(f"\nWith fixed sigma={gauss_width}:")
print(f"  Slab thickness: {result_fixed['z2'] - result_fixed['z1']:.3f} Angstrom")

# --- 4. Plot ---
# Reconstruct the double-erf model curve
zs_fine = np.linspace(zs[0], zs[-1], 1000)
arg1 = (zs_fine - z1) / (np.sqrt(2) * sigma)
arg2 = (zs_fine - z2) / (np.sqrt(2) * sigma)
fit_density = 0.5 * rho * (erf(arg1) - erf(arg2))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(zs, cu_density, label='Cu density profile')
ax.plot(zs_fine, fit_density, '--', label='Double-erf fit')
ax.axvline(z1, color='green', linestyle=':', label=f'z1 = {z1:.2f} A')
ax.axvline(z2, color='red', linestyle=':', label=f'z2 = {z2:.2f} A')
ax.set_xlabel('z [Angstrom]')
ax.set_ylabel('Density [atoms/Angstrom^3]')
ax.set_title(f'Cu slab thickness = {thickness:.2f} A')
ax.legend()
plt.tight_layout()
plt.savefig('slab_thickness.png', dpi=150)
print("Plot saved to slab_thickness.png")
