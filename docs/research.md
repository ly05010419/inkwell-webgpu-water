# Tethys spectral-water research pass

The target is an animated ocean surface that reads as water before it reads as a shader: irregular swell, nonlinear shallow-water response, stable distant shading, advected whitewater, and a quiet procedural seabed. Exposed terrain exists only in the island validation scene used to test shoreline wet/dry behavior and scene refraction.

## What the screenshot passes rejected

The first implementation summed seven hand-authored Gerstner waves, added sine/cellular micro-normal patterns, painted cellular caustics on the floor, and rendered detached Bezier crest sheets. Deterministic 1600×1000 captures exposed the failure modes:

- the repeated micro waves read as a wavy-line texture;
- cellular caustics and sinusoidal sand bands looked stamped under the water;
- crest sheets produced disconnected foam bars instead of breaking water;
- one smooth wave band alternated between flat/plastic and implausibly sharp;
- mirrored sand imagery added detail but not physical coherence.

Those approaches were removed, not hidden behind the optimized mode. The preserved comparison is in `benchmarks/visual/baseline-20260811-141446/`.

## Sources that changed the architecture

### Spectral evolution instead of a small wave recipe

Tessendorf represents the ocean statistically in frequency space, evolves each Fourier mode with the gravity-wave dispersion relation, and recovers spatial displacement with an inverse FFT. Horvath extends the practical spectrum choices with empirical directional models, including JONSWAP/TMA-style sea states and controllable directional spreading. This is now the base simulation rather than an approximation used only for normals.

- [Jerry Tessendorf: Simulating Ocean Water](https://3dmodelizm.ru/ext/graphics.ucsd.edu/courses/rendering/2005/jdewall/tessendorf.pdf)
- [Christopher Horvath: Empirical Directional Wave Spectra for Computer Graphics](https://dl.acm.org/doi/10.1145/2791261.2791267)

### Several wavelength bands, each with its own dispersion

Production systems do not ask one periodic field to carry the full ocean. NVIDIA WaveWorks exposes dual customizable JONSWAP spectra, multiple simulation LODs, foam, and an anisotropic PBR-ready BRDF. Crest similarly uses a multi-resolution data structure for displacement, dynamic waves, flow, and foam. The WebGPU lab therefore evaluates three independent 128² bands:

1. 240 m long-wave domain, including a second directional swell spectrum;
2. 64 m wind-wave domain;
3. 12 m capillary-gravity domain, reaching a 24 rad/m cutoff (roughly decimetre wavelengths).

Cutoff bands prevent double-counting. Each band evolves independently and runs fourteen Stockham inverse-FFT stages. Four packed complex fields produce horizontal displacement, height, slopes, cross derivatives, and horizontal displacement derivatives in parallel.

- [NVIDIA WaveWorks](https://developer.nvidia.com/waveworks)
- [NVIDIA: Three Things You Need to Know About WaveWorks 2.0](https://developer.nvidia.com/blog/three-things-you-need-to-know-about-wave-works-2-0/)
- [SIGGRAPH 2019: Multi-resolution Ocean Rendering in Crest](https://www.advances.realtimerendering.com/s2019/index.htm)
- [ARM: OpenGL ES FFT Ocean](https://arm-software.github.io/opengl-es-sdk-for-android/ocean_f_f_t.html)

### Transition from resolved geometry to a slope BRDF

Bruneton, Neyret, and Holzschuch show that ocean geometry, normals, and the BRDF must transition coherently across scale and distance. Directly turning sub-pixel waves into normals creates shimmer and painted streaks. The lab displaces geometry with the long and medium bands, attenuates resolved short-wave slopes with distance, and moves their remaining energy into roughness. The result is stable at the horizon without discarding the animated short-wave simulation.

- [Bruneton, Neyret, Holzschuch: Real-time Realistic Ocean Lighting using Seamless Transitions from Geometry to BRDF](https://evasion.inrialpes.fr/Membres/Fabrice.Neyret/images/fluids-nuages/waves/Jonathan/articlesCG/real-time-realistic-ocean-lighting-using-seamless-transitions-from-geometry-to-BRDF-10.pdf)
- [SIGGRAPH 2017: Crest — Novel Ocean Rendering Techniques](https://www.advances.realtimerendering.com/s2017/index.html)
- [SIGGRAPH 2013: Oceans on a Shoestring](https://advances.realtimerendering.com/s2013/index.html)

### Weak nonlinearity before explicit breaking

A purely linear random sea is vertically symmetric. Real gravity waves develop sharper crests, broader troughs, non-Gaussian slopes, and horizontal skew before breaking. The choppy-wave literature shows that local Lagrangian deformation is a computationally efficient weakly nonlinear correction, while the improved model also corrects time evolution through nonlinear dispersion. The lab retains its established horizontal spectral displacement and adds a restrained low-order bound-harmonic height/slope correction per long and medium cascade. It is intentionally kept below the phase screenshots that produced near-singular folded triangles.

- [Nouguier, Guérin, Chapron: “Choppy wave” model for nonlinear gravity waves](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JC004984)
- [An improved Lagrangian model for the time evolution of nonlinear surface waves](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/0A67400DB48E55B66CC08F97DD00751A/S0022112019005196a.pdf/an-improved-lagrangian-model-for-the-time-evolution-of-nonlinear-surface-waves.pdf)

### Whitewater is selected by deformation

Dupuy and Bruneton derive whitecap coverage from unstable wave deformation rather than unrelated noise. The lab reconstructs the horizontal Jacobian from FFT derivatives and permits foam only where that mapping compresses strongly and the crest is high. Multi-scale value noise breaks coverage after the physical selection; it never chooses the crest location.

- [Dupuy and Bruneton: Real-time Animation and Rendering of Ocean Whitecaps](https://hal.science/hal-00967078)
- [Crest foam documentation](https://crest.readthedocs.io/en/latest/user/foam.html)

### Water is a participating volume

The surface separates Fresnel reflection from RGB transmission, absorption, and in-scattering. Optical distance is taken through the water, not from a generic transparency slider. The floor uses the same procedural bathymetry as the render mesh. Exposed islands add only broad domain-warped dune relief and organic sediment variation; fine geometric ripples were rejected after they aliased at grazing angles. Artificial coordinate jitter, cellular networks, sinusoidal sand stripes, and image textures were removed after screenshot review.

- [Epic Games: Single Layer Water Shading Model](https://dev.epicgames.com/documentation/unreal-engine/single-layer-water-shading-model-in-unreal-engine)
- [GDC 2017: Shore to Horizon notes](https://media.gdcvault.com/gdc2017/Presentations/Longchamps_Nicolas_ShoreToHorizon_NOTES.pdf)
- [NVIDIA GPU Gems: Rendering Water Caustics](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics)

### Ocean reflection is not a tinted plastic lobe

The first spectral result still looked synthetic because it tinted and softened the reflected sky, added height-based crest colour, and used too much bright cyan in-scattering. The revised interface uses exact dielectric Fresnel for a 1.333 index of refraction and a wind-aligned Cox-Munk facet-slope distribution for direct sun glitter. Reflected radiance keeps the contrast of the procedural atmosphere, while transmitted open-water energy remains deliberately dark. The short spectral cascade now extends into capillary-gravity scales so the glitter path breaks into animated facets rather than a broad satin highlight.

- [Cox and Munk: Measurement of the Roughness of the Sea Surface from Photographs of the Sun's Glitter](https://elischolar.library.yale.edu/journal_of_marine_research/824/)
- [Bréon and Henriot: Spaceborne observations of ocean glint reflectance and modeling of wave slope distributions](https://arxiv.org/abs/2210.05456)
- [An analytical model for the bidirectional reflectance distribution of the sea surface](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2007JC004137)

The physically separated terms are deliberately art-directed rather than locked to photographic navy water. The current grade raises the atmosphere and transmitted water into a high-key cerulean/turquoise range, with pale warm-cyan highlights and darker teal troughs. The Fresnel weighting, capillary breakup, and low diffuse energy remain intact so the brighter palette does not revert to the earlier plastic sheet.

### Breaking fronts need local nonlinear deformation

A Fourier height field remains single-valued. It can produce steep crests and whitecaps, but not a plunging lip with an air cavity. Horizon Forbidden West uses localized deformations assembled into controllable wavefronts; wave-particle work likewise treats breaking motion as more than the base wave equation. The lab blends one travelling, meandering nonlinear displacement front into the same water parameterization. Its horizontal and vertical motion steepen the attached surface, while analytic derivatives update its normal and deformation-driven foam selection.

The screenshot pass tested two stronger local-topology profiles and rejected both: a folded strip formed a bright cylindrical tube, while a reduced fold formed a long horizontal shelf when viewed exactly side-on. The retained version first redistributes base-grid samples toward the front, then replaces only a narrow band of base fragments with a 256×48 bufferless ribbon. That ribbon evaluates the same FFT fields, depth attenuation, local wake, material, and foam as the surrounding ocean; its extra displacement fades to zero at shared boundaries. It only rounds and noses high-breakup crest sections rather than pretending to simulate an unsupported air cavity. Front, profile, pre-reset, post-reset, and underwater captures showed no detached sheet, hole, doubled surface, or reset pop. A true plunging air cavity still requires a localized multiply connected surface or hybrid particle/volume representation.

- [SIGGRAPH 2022: Rendering Water in Horizon Forbidden West](https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Water-Malan.pdf)
- [Cem Yuksel: Real-Time Water Waves with Wave Particles](https://www.cemyuksel.com/research/waveparticles/)
- [Hybrid FFT and wave-particle simulation](https://arxiv.org/abs/2511.02852)

## Current GPU path

1. Precompute one 513² procedural terrain height/normal field.
2. Upload deterministic TMA/JONSWAP initial spectra and dispersion data for three 128² cascades.
3. Each frame, evolve all spectra and transform four packed complex fields per cascade with Stockham inverse FFTs.
4. Advance a 256² conservative nearshore elevation/momentum/foam state. Hydrostatic reconstruction and a Rusanov flux preserve wet/dry fronts; spectral boundary relaxation couples it to the far field.
5. Update a persistent breaker-event field from spectral Jacobian compression plus shallow-water depth, Froude, and compression indicators. Backtrace foam through the local velocity and apply independent birth/decay rates.
6. Render the open ocean with four camera-snapped nested rings and a physically selected local breaker patch. The island validation scene switches to matching 512² terrain/water contours so the wet/dry edge is stable.
7. Resolve near capillary-gravity slopes, transition them into aggregate BRDF roughness with distance, and evaluate the shared sun through the Cox-Munk glint term.
8. Shade sky reflection, RGB extinction/scattering, procedural-floor transmission, and underwater fog. The island scene makes one color/depth capture so refracted terrain is the rendered scene rather than a duplicate analytic stand-in.

The latest measured production profile holds 60 fps with no >50 ms hitches or browser errors. Two optimized confirmations average **6.883 ms** timestamped GPU work at the surface and **6.150 ms** underwater; the matched reference cases are **7.576 ms** and **6.780 ms**. The normal scene reports 166,400 triangles. The optional shoreline stress scene renders 1,048,576 terrain + water triangles with a 512² nearshore simulation at 1.25× scale in **9.315 ms GPU total**, versus its 9.540 ms pre-pass baseline. This is an isolated renderer result, not a full-game claim.
