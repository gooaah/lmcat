# lmcat Design Notes

## Project Purpose

`lmcat` computes the position of the **Gibbs dividing surface** at a
liquid-metal / adsorbate interface from molecular-dynamics trajectories.

Algorithm pipeline:

1. Read MD trajectory (list of ASE `Atoms` snapshots).
2. Build a z-density histogram for each atomic species.
3. Apply Gaussian smoothing to obtain a continuous density profile.
4. Fit a complementary error function (erfc) to the profile edge to locate
   the Gibbs dividing surface.

The Gibbs dividing surface at position μ satisfies:

    ρ(z) = factor · erfc((z − μ) / (σ√2))

where σ is the interface width and factor is proportional to the bulk density.

---

## Module Layout

```
lmcat/
├── io.py        — I/O primitives: atom counting, slab–layer distances
├── profile.py   — Histogram construction and Gaussian-smoothed profiles
├── fitting.py   — erfc / double-erf fitting to locate Gibbs surfaces
└── analysis.py  — High-level analysis: profile alignment, half-max, distances
```

The four-layer split reflects the natural data-flow order:

| Layer | Responsibility | Dependency |
|---|---|---|
| `io` | Load/count atoms | none |
| `profile` | Build histograms, smooth | `io` indirectly |
| `fitting` | Fit erfc, extract interface position | `profile` output |
| `analysis` | Combine fitted results, align to experiment | all of the above |

---

## Public API

Functions exposed via `lmcat.__init__` (and `__all__`):

| Function | Module | Purpose |
|---|---|---|
| `count_element` | `io` | Count atoms of one element |
| `layer_slab_distances` | `io` | Layer–slab vertical distance statistics |
| `trajectory_histogram` | `profile` | Area-normalized z-histogram for one species |
| `smooth_density` | `profile` | Gaussian-smooth an area-normalized histogram |
| `density_profile` | `profile` | Full pipeline: traj → smooth density profiles |
| `fit_interface` | `fitting` | Fit single erfc → Gibbs surface position |
| `fit_slab_interfaces` | `fitting` | Fit double erfc → slab thickness + interfaces |
| `align_profiles` | `analysis` | Align computed profiles to experiment |
| `element_density` | `analysis` | Evaluate element profiles at given z-values |
| `interface_distances` | `analysis` | Peak/half-max distances between layer and slab |

Internal (private) functions use an underscore prefix, e.g. `_double_erf_sym`.

---

## Unit Conventions

| Quantity | Unit |
|---|---|
| Length / position | Angstrom (Å) |
| Density | atoms/Å³ |
| Interface width σ | Angstrom |
| Area (xy cross-section) | Å² |

---

## Known Limitations and Design Choices

### `trajectory_histogram` vs `trajectory_histogram_full`

- `trajectory_histogram` returns an **area-normalized** histogram
  [atoms/Å³ per bin width].  Feed into `smooth_density`.
- `trajectory_histogram_full` returns a **raw** (un-normalized) histogram
  [counts per bin, averaged over frames].  Feed into
  `smooth_density_two_species`.

The two paths exist for historical reasons; prefer `trajectory_histogram` +
`smooth_density` for new code.

### shgo seed

`fit_interface` and `align_profiles` use `scipy.optimize.shgo` with
`options={'seed': 42}` to make the global optimization reproducible.
Results may change slightly between SciPy versions.

### Orthogonal cell assumption

`trajectory_histogram` and `trajectory_histogram_full` compute the xy area
as `cell[0,0] * cell[1,1]`.  This is only correct for orthogonal simulation
cells.  Non-orthogonal (triclinic) cells are not supported.
