# Implementation source ledger

The water lab uses papers and inspectable source implementations as design
references. It does not copy code from copyleft projects. The native runtime is
TypeScript and WGSL and remains MIT licensed.

## Nearshore solver, wet/dry shoreline, and run-up

- Celeris GPU extended Boussinesq solver: [paper](https://arxiv.org/abs/1611.05984), [source](https://github.com/SasanTV/Celeris) at `3fb2f6d25314490f20c76b393c79448d919d0a25` (GPL-3.0). Used as a clean-room architectural reference for hydrostatic reconstruction, predictor/corrector staging, friction, sponge boundaries, wet/dry limiting, and breaking indicators.
- Adaptive third-order Celeris: [paper](https://arxiv.org/abs/1909.04153). Used to set the longer-term accuracy target after the first robust finite-volume pass.
- XBeach: [source](https://github.com/openearth/xbeach). Used as a reference for wave-action coupling, run-up, backwash, and wetting/drying validation scenarios.

Acceptance gates: lake-at-rest preservation over variable bathymetry, no
negative depth at wet/dry fronts, bounded momentum, conservation in closed
tests, stable 60-second headed run, and visible shoaling without an edge seam.

## Local breaking topology and wave transport

- Water Wave Packets: [paper and project](https://visualcomputing.ist.ac.at/publications/2017/WWP/), [source](https://github.com/jeschke/water-wave-packets) at `1487cbbafab47c5854ff9f8b9b717051ee4e4d7f`. Used for packet energy, group velocity, Snell refraction, boundary reflection, and local wave transport.
- Horizon Forbidden West water: [SIGGRAPH 2022 slides](https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Water-Malan.pdf). Used for the hybrid far field/local deformation architecture and controllable breaking fronts.
- Real-Time Water Waves with Wave Particles: [paper/project](https://www.cemyuksel.com/research/waveparticles/). Used for event-local wave generation and propagation.

Acceptance gates: breaking appears only after an instability threshold, shares
positions/normals with the base surface at patch boundaries, has no detached
sheet or reset pop, and remains stable from side, front, overhead, and underwater
views.

## Persistent foam

- Crest: [source](https://github.com/wave-harmonic/crest) at `db0658ff0b2e93e4a9e28cc2867509658b0ecc00` (MIT), especially `UpdateFoam.compute`. Used for backtraced flow advection, multi-source foam birth, shoreline contribution, decay, and LOD fallback.
- Dupuy and Bruneton, Real-time Animation and Rendering of Ocean Whitecaps: [paper](https://hal.science/hal-00967078). Used for deformation-driven whitecap selection.

Acceptance gates: foam is born from compression/breaking/shoreline events,
moves with the simulated flow, expands and decays over time, and does not glow
or remain camera-locked.

## Reflection, refraction, lighting, and atmosphere

- FidelityFX Stochastic Screen Space Reflections: [source](https://github.com/GPUOpen-Effects/FidelityFX-SSSR) (MIT). Used to define the scene-color/depth/normal integration hooks and hierarchical ray-march path.
- Bruneton atmospheric scattering: [reference implementation](https://ebruneton.github.io/precomputed_atmospheric_scattering/). Used to unify sun radiance, sky reflection, aerial perspective, and underwater illumination under one atmosphere.
- Epic Single Layer Water: [documentation](https://dev.epicgames.com/documentation/unreal-engine/single-layer-water-shading-model-in-unreal-engine). Used for the reflection/transmission/volume composition order.

Acceptance gates: external scene color and depth can replace the lab fallback,
reflected features track the scene, refraction varies with thickness, and the sun
direction/color is identical in sky, glint, seabed, fog, and underwater shading.

## Geometry LOD and temporal stability

- Projected grid for water rendering: [thesis](https://fileadmin.cs.lth.se/graphics/theses/projects/projgrid/projgrid-hq.pdf). Used for camera-projected sampling density and stable horizon coverage.
- FidelityFX FSR2: [source and algorithm](https://github.com/GPUOpen-Effects/FidelityFX-FSR2). Used as a reference for motion-vector conventions, reactive masks, history rejection, and temporal stabilization; the lab does not embed FSR2.
- Bruneton, Neyret, and Holzschuch, seamless geometry-to-BRDF transition: [paper](https://evasion.inrialpes.fr/Membres/Fabrice.Neyret/images/fluids-nuages/waves/Jonathan/articlesCG/real-time-realistic-ocean-lighting-using-seamless-transitions-from-geometry-to-BRDF-10.pdf). Used for moving unresolved wave energy into the BRDF instead of aliasing geometry.

Acceptance gates: near-camera triangle density increases without raising the
full-domain count, horizon coverage survives camera motion, temporal shimmer is
lower in fixed-path comparisons, and GPU render time does not regress by more
than 10% against the retained baseline.
