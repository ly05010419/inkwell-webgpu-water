/// <reference types="@webgpu/types" />

import { ShipRenderer } from "@/lib/ship-renderer";
import { COLOR_FUNCTIONS, TETHYS_AERIAL_WGSL, WORLD_UNIFORMS } from "@/lib/shared-wgsl";
import {
  TETHYS_REFERENCE_SIMULATION_RESOLUTION,
  TETHYS_WATER_FIELD_SIZE,
  TETHYS_WATER_LEVEL,
  waterSimulationBytes,
  type WaterRenderMode,
  type WaterScene,
  type WaterView,
} from "@/lib/water-profiles";

type Vec3 = [number, number, number];

export type WaterLabOptions = {
  mode: WaterRenderMode;
  view: WaterView;
  meshResolution: number;
  simulationResolution: number;
  renderScale: number;
  waveScale: number;
  distantRoughness: number;
  detailRange: number;
  swellSmoothing: number;
  longCascadeScale: number;
  mediumCascadeScale: number;
  fogReach: number;
  fixedTime?: number;
  benchmark?: boolean;
  cameraYaw?: number;
  cameraPitch?: number;
  scene: WaterScene;
};

export type WaterLabMetrics = {
  ready: boolean;
  mode: WaterRenderMode;
  view: WaterView;
  meshResolution: number;
  simulationResolution: number;
  waveScale: number;
  distantRoughness: number;
  detailRange: number;
  swellSmoothing: number;
  longCascadeScale: number;
  mediumCascadeScale: number;
  fogReach: number;
  triangles: number;
  simulationBytes: number;
  simulationSubsteps: number;
  sceneCapturePasses: number;
  disturbanceCount: number;
  particleCount: number;
  frameMeanMs: number;
  frameP95Ms: number;
  frameP99Ms: number;
  frameMaxMs: number;
  fps: number;
  hitchFrames: number;
  submitMeanMs: number;
  gpuSimulationMeanMs: number | null;
  gpuSimulationP95Ms: number | null;
  gpuRenderMeanMs: number | null;
  gpuRenderP95Ms: number | null;
  gpuTimestampSamples: number;
  adapter: string;
  error: string | null;
};

// Baseline world extent, and the reference the fog/far-plane scale divides by.
// 520 m across a 512-tap terrain field is 1.02 m per texel -- widened from the
// authored 390 m so the southern island (shelf centre at z = -196) fits inside
// with margin. Land must never reach the border row: everything beyond the
// field is border-clamped, so an island touching the edge casts a dry-land
// shadow to infinity through the clamp, discarding the water above it.
const TERRAIN_EXTENT = 520;
// The open ocean sees 10x further. This scales fog reach, the far plane and the
// water's mesh coverage -- deliberately NOT the terrain field, which stays at
// 390 m across 512 taps (0.76 m/texel). Stretching the field to match would
// have dropped it to 7.6 m/texel and flattened the seabed dunes that the
// underwater camera looks straight at. Past the authored centre the field
// clamps to its flat -8.5 m border, and water absorption hides the seabed long
// before that, so the water can extend far beyond the terrain for free.
const OPEN_WATER_VIEW_SCALE = 100;
// Orbit ceilings. The island scene keeps the authored 145 m (its camera is
// pinned to 96 m anyway). The open ocean's old 250 m ceiling guarded against
// "bare seabed emerging past ~300 m of orbit" -- that exposure was the
// breaker-warp tanh overflowing to NaN and dropping the downwind water wedge,
// fixed at the clamp in adaptiveBreakerCoordinates. With the surface intact
// the ceiling only needs to stay inside the clipmap's 16384 m reach so water
// still surrounds the camera; 12 km also keeps f32 world coordinates well
// clear of visible spectral-UV jitter.
const OPEN_WATER_MAX_ORBIT = 12000;
// A glTF hull riding the simulated surface. It is placed near the camera target
// so the default framing shows it, and offset from the wake impulse at (0, -12)
// so the two read as separate features.
// Swell amplitude multiplier. The floor is not zero: a dead-flat surface has no
// slope for the BRDF to work with and the ocean turns into a mirror.
const MIN_WAVE_SCALE = 0.15;
const MAX_WAVE_SCALE = 1.6;
// How much of the faded capillary slope is returned to BRDF roughness.
// 0 reproduces the original look, where the far surface tends toward a mirror.
const MAX_DISTANT_ROUGHNESS = 3;
// Multiplier on the 42-118 m capillary fade and the 95-188 m crest fade.
const MIN_DETAIL_RANGE = 0.4;
const MAX_DETAIL_RANGE = 8;
// Strength of the swell cascades' screen-space slope fade. 1 reproduces the
// tuned look, 0 keeps full per-fragment slope to the horizon (glittery).
const MAX_SWELL_SMOOTHING = 3;
// Runtime bounds for the two displacing cascades' tile sizes. The spectrum is
// regenerated on the CPU when either changes; shaders read the live values
// from atmosphere.zw, so no pipeline rebuild is involved.
const LONG_SCALE_RANGE = [80, 480] as const;
const MEDIUM_SCALE_RANGE = [24, 128] as const;
// Where the open ocean's radial fog closes, relative to its authored position.
// 0 removes it entirely, which is the default.
const MAX_FOG_REACH = 3;

const SHIP_MODEL_URL = "/models/dutch_ship_medium/dutch_ship_medium_2k.gltf";
// The two scenes do not share a seabed: the open ocean's is a submerged shelf
// while the island scene raises authored dunes above the waterline. A single
// position would beach the hull in one of them, so each scene gets its own spot
// in open water, angled so the broadside and bow both read from the camera.
const SHIP_PLACEMENTS = Object.freeze({
  open: Object.freeze({
    centre: Object.freeze([7, -12] as [number, number]),
    heading: 2.60,
    // The model's waterline sits at its own origin; this trims how deep it floats.
    draft: -0.65,
  }),
  shore: Object.freeze({
    // Just outside the island's shelf and inshore of it, so the hull reads
    // against the dunes instead of being lost beyond the fixed 96 m orbit.
    centre: Object.freeze([50, 60] as [number, number]),
    heading: 3.12,
    draft: -0.65,
  }),
});
const TERRAIN_FIELD_RESOLUTION = 512;
const FRAME_HISTORY = 360;
const WORLD_UNIFORM_BYTES = 256;
const SIMULATION_PARAM_BYTES = 32;
const DEPTH_FORMAT: GPUTextureFormat = "depth24plus";
const SPECTRAL_RESOLUTION = 128;
const SPECTRAL_LOG_SIZE = 7;
const BREAKER_PATCH_ALONG_RESOLUTION = 256;
const BREAKER_PATCH_ACROSS_RESOLUTION = 48;
const BREAKER_EVENT_RESOLUTION = 256;
const BREAKER_PATCH_TRIANGLES = BREAKER_PATCH_ALONG_RESOLUTION * BREAKER_PATCH_ACROSS_RESOLUTION * 2;
// The travelling localized breaker front. It is a single long crest line that
// sweeps across the open-ocean domain, so a camera aimed along its tangent sees
// one continuous ridge spanning the frame. Disabled here; the spectral cascades
// and the nearshore state keep owning the surface.
//
// This gates five coupled sites that must agree, or the surface tears:
//   1. the adaptive vertex warp that concentrates the grid on the front
//   2. the crest displacement added to the main water surface
//   3. the attached 256x48 crest patch geometry
//   4. the main-surface discard that hands the band over to that patch
//   5. the patch draw call and its triangle accounting
//   6. the per-fragment shading normal in waterFragment, which must fold the
//      breaker displacement derivatives back in once the crest returns
// Leaving 4 enabled without 3 punches a transparent hole along the crest band.
const BREAKER_ENABLED = false;
const BREAKER_SHADER_GATE = BREAKER_ENABLED ? "1.0" : "0.0";
const WATER_CLIPMAP_RESOLUTION = 64;
// Rings are 32 * 2^level metres of half-extent, so the count sets reach while
// the innermost ring keeps its cell size. Raising the base extent instead
// would have coarsened the water right under the camera.
//
// Ten levels reach 16384 m, past the 14500 m point where the open-ocean radial
// fog is already opaque. That headroom is not waste: the rings are
// snapped to the camera while the terrain field stays centred on the world, so
// at the 1450 m zoom limit the water must still span the 1950 m terrain radius
// from an off-centre origin (1450 + 1950 = 3400 m). Falling short of that lets
// the seabed and the sky show through beyond the water's edge.
const WATER_CLIPMAP_LEVELS = 10;
// Where the outermost ring's edge vertices are thrown to, in metres. Two hard
// bounds: it must exceed the outermost ring (16384 m) or the skirt would pull
// geometry inward, and its depth must stay in front of the sky, which writes
// 0.999999. With near 0.12 m and the far plane below, 20 km satisfies both.
const WATER_HORIZON_REACH = 20000;
// The open ocean's far plane is fixed rather than scaled: it has to clear the
// skirt regardless of view scale, and pushing it further only costs depth
// precision. The island scene keeps the authored 560 m, where its waterline
// needs the precision and nothing is drawn beyond a few hundred metres.
const OPEN_WATER_FAR_PLANE = 50000;
const SPECTRAL_CASCADES = [
  { lengthScale: 240, cutoffLow: 0.024, cutoffHigh: 0.36, amplitudeScale: 0.45, choppiness: 1.18, secondaryScale: 0.22, seed: 0x51f15e },
  { lengthScale: 64, cutoffLow: 0.30, cutoffHigh: 1.42, amplitudeScale: 0.45, choppiness: 1.05, secondaryScale: 0.08, seed: 0x72a93b },
  // This cascade reaches decimetre-scale capillary-gravity waves. It shades
  // the interface only; carrying it into the mesh would alias and look ridged.
  { lengthScale: 12, cutoffLow: 1.22, cutoffHigh: 24.0, amplitudeScale: 0.82, choppiness: 0.40, secondaryScale: 0, seed: 0x19ce47 },
] as const;

type SpectralFieldPingPong = [[GPUTexture, GPUTexture], [GPUTexture, GPUTexture]];


const TETHYS_TERRAIN_WGSL = /* wgsl */ `
fn tethysCoastalShelf(p: vec2<f32>, center: vec2<f32>, radiusScale: vec2<f32>, lift: f32, relief: f32, phase: f32) -> f32 {
  let delta = p - center;
  let angle = phase * 0.23;
  let local = vec2<f32>(
    delta.x * cos(angle) - delta.y * sin(angle),
    delta.x * sin(angle) + delta.y * cos(angle)
  ) / radiusScale;
  let radius = length(local);
  let coastAngle = atan2(local.y, local.x);
  let coastNoise = sin(coastAngle * 3.0 + phase) * 0.040
    + sin(coastAngle * 7.0 - phase * 0.8) * 0.022
    + sin((p.x + p.y) * 0.031 + phase) * 0.018;
  let coastalDistance = radius + coastNoise;
  let coast = 1.0 - smoothstep(0.58, 1.035, coastalDistance);
  let interior = 1.0 - smoothstep(0.12, 0.57, coastalDistance);
  let erosion = sin(p.x * 0.038 + p.y * 0.017 + phase) * 0.52
    + sin(p.x * -0.019 + p.y * 0.043 - phase * 0.7) * 0.31
    + sin((p.x + p.y) * 0.081 + phase * 1.4) * 0.17;
  let longRidge = sin(p.x * 0.014 - p.y * 0.021 + phase * 2.1);
  let highland = pow(smoothstep(-0.20, 0.85, longRidge), 1.35);
  let rollingRelief = -0.08 + erosion * 0.22 + highland * 0.86;
  return max(0.0, coast * (lift + relief * rollingRelief * interior));
}

fn terrainHeight(p: vec2<f32>, shoreMix: f32) -> f32 {
  var warped = p;
  warped.x += sin(p.y * 0.018 + 0.8) * 2.4;
  warped.y += sin(p.x * 0.016 - 0.2) * 2.1;
  var height = -8.5;
  height += sin(warped.x * 0.052 + warped.y * 0.016) * 0.62;
  height += sin(warped.x * -0.024 + warped.y * 0.046 + 1.7) * 0.39;
  height += sin((warped.x + warped.y) * 0.12) * 0.14;
  var shelfPower = 0.0;
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(0.0, 14.0), vec2<f32>(76.0, 50.0), 12.8, 8.0, 0.3), 6.0);
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(-112.0, -79.0), vec2<f32>(62.0, 40.0), 12.6, 11.0, 1.7), 6.0);
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(116.0, -92.0), vec2<f32>(65.0, 44.0), 12.5, 12.0, 3.4), 6.0);
  shelfPower += pow(tethysCoastalShelf(warped, vec2<f32>(-6.0, -196.0), vec2<f32>(112.0, 40.0), 12.9, 10.0, 5.1), 6.0);
  height += pow(max(shelfPower, 0.0), 1.0 / 6.0);
  // Build broad, domain-warped dune ridges inland. Keeping their mask above
  // the swash zone protects the waterline contour while giving the exposed
  // islands a wind-shaped silhouette instead of a smooth clay mound.
  let duneInterior = smoothstep(${(TETHYS_WATER_LEVEL + 0.62).toFixed(2)}, ${(TETHYS_WATER_LEVEL + 2.55).toFixed(2)}, height);
  let duneWarp = vec2<f32>(
    valueNoise(p * 0.021 + vec2<f32>(7.1, -3.8)) - 0.5,
    valueNoise(p * 0.024 + vec2<f32>(-5.3, 9.6)) - 0.5
  );
  let duneP = p + duneWarp * 17.0;
  let duneBands = sin(duneP.x * 0.098 + duneP.y * 0.031)
    + sin(duneP.x * 0.047 - duneP.y * 0.071 + 1.8) * 0.47;
  let duneRidges = sign(duneBands) * pow(abs(duneBands) * 0.68, 1.32);
  let broadDunes = valueNoise(duneP * 0.038 + vec2<f32>(2.3, 6.7)) - 0.5;
  let erodedDetail = valueNoise(duneP * 0.14 + vec2<f32>(4.7, -2.1)) - 0.5;
  height += shoreMix * duneInterior
    * (duneRidges * 0.52 + broadDunes * 0.82 + erodedDetail * 0.16);
  // This lab isolates the water material. Preserve Tethys' shelf contours as
  // a submerged seabed, but never expose an island or terrestrial surface.
  var seabed = min(height, -4.35);
  seabed += sin(p.x * 0.071 + p.y * 0.026) * 0.38;
  seabed += sin(p.x * -0.033 + p.y * 0.083 + 1.7) * 0.24;
  seabed += sin(p.x * 0.017 - p.y * 0.013 + 0.6) * 0.48;
  // The material lab keeps the original all-submerged view, while the coastal
  // scene restores authored Tethys islands for wet/dry and run-up validation.
  // Everything fades to the flat deep border before the field edge: exposed
  // land back to seabed, and the seabed's own dunes down to the -8.5 m floor.
  // The border row is clamp-repeated to infinity by every out-of-field sample,
  // so any relief left on it casts a visible shallow-water ray outward; see
  // the TERRAIN_EXTENT note.
  let borderFade = 1.0 - smoothstep(245.0, 258.0, max(abs(p.x), abs(p.y)));
  return mix(-8.5, mix(seabed, height, shoreMix * borderFade), borderFade);
}

fn hash21(p: vec2<f32>) -> f32 {
  var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
  p3 += vec3<f32>(dot(p3, p3.yzx + vec3<f32>(33.33)));
  return fract((p3.x + p3.y) * p3.z);
}

fn valueNoise(p: vec2<f32>) -> f32 {
  let cell = floor(p);
  var local = fract(p);
  local = local * local * (vec2<f32>(3.0) - 2.0 * local);
  return mix(
    mix(hash21(cell), hash21(cell + vec2<f32>(1.0, 0.0)), local.x),
    mix(hash21(cell + vec2<f32>(0.0, 1.0)), hash21(cell + vec2<f32>(1.0, 1.0)), local.x),
    local.y
  );
}
`;

const TERRAIN_FIELD_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
${TETHYS_TERRAIN_WGSL}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var fieldOut: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(16, 16)
fn buildTerrain(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(fieldOut);
  if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
  let uv = vec2<f32>(id.xy) / vec2<f32>(dimensions - vec2<u32>(1u));
  let p = (uv - vec2<f32>(0.5)) * uniforms.terrain.x;
  let spacing = uniforms.terrain.x / f32(dimensions.x - 1u);
  let height = terrainHeight(p, uniforms.environment.x);
  let left = terrainHeight(p - vec2<f32>(spacing, 0.0), uniforms.environment.x);
  let right = terrainHeight(p + vec2<f32>(spacing, 0.0), uniforms.environment.x);
  let back = terrainHeight(p - vec2<f32>(0.0, spacing), uniforms.environment.x);
  let front = terrainHeight(p + vec2<f32>(0.0, spacing), uniforms.environment.x);
  let normal = normalize(vec3<f32>(left - right, spacing * 2.0, back - front));
  textureStore(fieldOut, vec2<i32>(id.xy), vec4<f32>(height, normal.x, normal.z, 0.0));
}
`;

const WATER_SIMULATION_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
struct SimulationParams {
  impulse: vec4<f32>,
  stepFoamShift: vec4<f32>,
}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var<uniform> params: SimulationParams;
@group(0) @binding(2) var previousState: texture_2d<f32>;
@group(0) @binding(3) var nextState: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var terrainField: texture_2d<f32>;
@group(0) @binding(5) var longField0: texture_2d<f32>;
@group(0) @binding(6) var longField1: texture_2d<f32>;
@group(0) @binding(7) var mediumField0: texture_2d<f32>;
@group(0) @binding(8) var mediumField1: texture_2d<f32>;
@group(0) @binding(9) var spectrumSampler: sampler;

const GRAVITY = 9.81;
const MIN_DEPTH = 0.035;

struct CellState {
  eta: f32,
  q: vec2<f32>,
  foam: f32,
  bottom: f32,
  depth: f32,
}

fn clampedCoord(coord: vec2<i32>, dimensions: vec2<u32>) -> vec2<i32> {
  return clamp(coord, vec2<i32>(0), vec2<i32>(dimensions) - vec2<i32>(1));
}

fn worldPosition(coord: vec2<i32>, dimensions: vec2<u32>) -> vec2<f32> {
  let uv = (vec2<f32>(coord) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
  return uniforms.simulation.xy + (uv - vec2<f32>(0.5)) * uniforms.simulation.z;
}

fn terrainAtWorld(p: vec2<f32>) -> f32 {
  let terrainDimensions = textureDimensions(terrainField);
  let uv = clamp(p / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let coord = vec2<i32>(round(uv * vec2<f32>(terrainDimensions - vec2<u32>(1))));
  return textureLoad(terrainField, coord, 0).r;
}

fn spectralBoundaryState(p: vec2<f32>, depth: f32) -> vec4<f32> {
  let longUv = fract(p / uniforms.atmosphere.z + vec2<f32>(0.5));
  let mediumUv = fract(p / uniforms.atmosphere.w + vec2<f32>(0.5));
  let long0 = textureSampleLevel(longField0, spectrumSampler, longUv, 0.0) * uniforms.waves.x;
  let long1 = textureSampleLevel(longField1, spectrumSampler, longUv, 0.0) * uniforms.waves.x;
  let medium0 = textureSampleLevel(mediumField0, spectrumSampler, mediumUv, 0.0) * uniforms.waves.x;
  let medium1 = textureSampleLevel(mediumField1, spectrumSampler, mediumUv, 0.0) * uniforms.waves.x;
  let longHeight = long0.b;
  let mediumHeight = medium0.b;
  let eta = longHeight + mediumHeight
    + 0.14 * (longHeight * longHeight - 0.080 * uniforms.waves.y)
    + 0.32 * (mediumHeight * mediumHeight - 0.030 * uniforms.waves.y);
  // The boundary transport follows the dominant spectrum direction. Interior
  // momentum immediately becomes bathymetry-aware through the conservative
  // flux. This is a relaxation boundary, not a second rendered wave layer.
  let meanDirection = normalize(vec2<f32>(0.887, -0.462));
  let direction = normalize(meanDirection - (long1.rg + medium1.rg) * 0.055);
  let phaseSpeed = sqrt(GRAVITY * max(depth, MIN_DEPTH));
  let crossDerivative = long0.a * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium0.a * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let horizontalDerivative = long1.ba * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium1.ba * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let jacobian = (1.0 + horizontalDerivative.x) * (1.0 + horizontalDerivative.y) - crossDerivative * crossDerivative;
  return vec4<f32>(eta, direction * eta * phaseSpeed, max(0.0, 1.0 - jacobian));
}

fn loadCell(coordIn: vec2<i32>, dimensions: vec2<u32>) -> CellState {
  let coord = clampedCoord(coordIn, dimensions);
  let raw = textureLoad(previousState, coord, 0);
  let bottom = terrainAtWorld(worldPosition(coord, dimensions));
  let depth = max(uniforms.sunWater.w + raw.r - bottom, 0.0);
  var result: CellState;
  result.eta = raw.r;
  result.q = select(raw.gb, vec2<f32>(0.0), depth <= MIN_DEPTH);
  result.foam = raw.a;
  result.bottom = bottom;
  result.depth = depth;
  return result;
}

fn conservativeState(cell: CellState, reconstructedDepth: f32) -> vec3<f32> {
  let scale = select(reconstructedDepth / max(cell.depth, MIN_DEPTH), 0.0, cell.depth <= MIN_DEPTH);
  return vec3<f32>(reconstructedDepth, cell.q * scale);
}

fn physicalFluxX(state: vec3<f32>) -> vec3<f32> {
  let h = max(state.x, MIN_DEPTH);
  let velocity = state.yz / h;
  return vec3<f32>(state.y, state.y * velocity.x + 0.5 * GRAVITY * state.x * state.x, state.y * velocity.y);
}

fn physicalFluxY(state: vec3<f32>) -> vec3<f32> {
  let h = max(state.x, MIN_DEPTH);
  let velocity = state.yz / h;
  return vec3<f32>(state.z, state.z * velocity.x, state.z * velocity.y + 0.5 * GRAVITY * state.x * state.x);
}

fn hydrostaticPair(a: CellState, b: CellState) -> array<vec3<f32>, 2> {
  let interfaceBottom = max(a.bottom, b.bottom);
  let surfaceA = uniforms.sunWater.w + a.eta;
  let surfaceB = uniforms.sunWater.w + b.eta;
  let hA = max(0.0, surfaceA - interfaceBottom);
  let hB = max(0.0, surfaceB - interfaceBottom);
  return array<vec3<f32>, 2>(conservativeState(a, hA), conservativeState(b, hB));
}

fn rusanovX(a: CellState, b: CellState) -> vec3<f32> {
  let pair = hydrostaticPair(a, b);
  let left = pair[0];
  let right = pair[1];
  let uLeft = select(left.y / max(left.x, MIN_DEPTH), 0.0, left.x <= MIN_DEPTH);
  let uRight = select(right.y / max(right.x, MIN_DEPTH), 0.0, right.x <= MIN_DEPTH);
  let speed = max(abs(uLeft) + sqrt(GRAVITY * left.x), abs(uRight) + sqrt(GRAVITY * right.x));
  return 0.5 * (physicalFluxX(left) + physicalFluxX(right)) - 0.5 * speed * (right - left);
}

fn rusanovY(a: CellState, b: CellState) -> vec3<f32> {
  let pair = hydrostaticPair(a, b);
  let south = pair[0];
  let north = pair[1];
  let vSouth = select(south.z / max(south.x, MIN_DEPTH), 0.0, south.x <= MIN_DEPTH);
  let vNorth = select(north.z / max(north.x, MIN_DEPTH), 0.0, north.x <= MIN_DEPTH);
  let speed = max(abs(vSouth) + sqrt(GRAVITY * south.x), abs(vNorth) + sqrt(GRAVITY * north.x));
  return 0.5 * (physicalFluxY(south) + physicalFluxY(north)) - 0.5 * speed * (north - south);
}

fn sidePressureCorrection(originalDepth: f32, reconstructedDepth: f32) -> f32 {
  return 0.5 * GRAVITY * (originalDepth * originalDepth - reconstructedDepth * reconstructedDepth);
}

@compute @workgroup_size(16, 16)
fn simulate(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(nextState);
  if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
  let coord = vec2<i32>(id.xy);
  let center = loadCell(coord, dimensions);
  let west = loadCell(coord - vec2<i32>(1, 0), dimensions);
  let east = loadCell(coord + vec2<i32>(1, 0), dimensions);
  let south = loadCell(coord - vec2<i32>(0, 1), dimensions);
  let north = loadCell(coord + vec2<i32>(0, 1), dimensions);
  let cellSize = uniforms.simulation.z / f32(dimensions.x);
  let dt = params.stepFoamShift.x;

  let eastPair = hydrostaticPair(center, east);
  let westPair = hydrostaticPair(west, center);
  let northPair = hydrostaticPair(center, north);
  let southPair = hydrostaticPair(south, center);
  var eastFlux = rusanovX(center, east);
  var westFlux = rusanovX(west, center);
  var northFlux = rusanovY(center, north);
  var southFlux = rusanovY(south, center);
  eastFlux.y += sidePressureCorrection(center.depth, eastPair[0].x);
  westFlux.y += sidePressureCorrection(center.depth, westPair[1].x);
  northFlux.z += sidePressureCorrection(center.depth, northPair[0].x);
  southFlux.z += sidePressureCorrection(center.depth, southPair[1].x);

  var next = vec3<f32>(center.depth, center.q) - dt * ((eastFlux - westFlux) + (northFlux - southFlux)) / cellSize;
  next.x = max(next.x, 0.0);
  var nextDepth = next.x;
  var nextQ = select(next.yz, vec2<f32>(0.0), nextDepth <= MIN_DEPTH);
  let speed = length(nextQ) / max(nextDepth, MIN_DEPTH);
  let manning = 0.018;
  let friction = GRAVITY * manning * manning * speed / max(pow(max(nextDepth, MIN_DEPTH), 1.333333), 0.001);
  nextQ /= 1.0 + dt * friction;

  let uv = (vec2<f32>(id.xy) + vec2<f32>(0.5)) / vec2<f32>(dimensions);
  let radius = max(params.impulse.w, 0.0001);
  let impulseDistance = length((uv - params.impulse.xy) / radius);
  let impulse = exp(-impulseDistance * impulseDistance * 3.2);
  let ring = exp(-pow(impulseDistance - 0.72, 2.0) * 18.0);
  nextDepth = max(0.0, nextDepth + (impulse - ring * 0.28) * params.impulse.z);
  let impulseDirection = normalize(vec2<f32>(uniforms.player.z, uniforms.player.w) + vec2<f32>(0.0001, 0.0));
  nextQ += impulseDirection * ring * params.impulse.z * 1.6;

  // Couple the far-field FFT to the nonlinear domain. A strong sponge forces
  // the outer band to the incident sea state, while deeper interior cells get
  // a much weaker source during warm-up. Shallow cells are then owned by the
  // conservative solver, allowing bathymetric refraction and run-up.
  let edgeDistance = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
  let sponge = 1.0 - smoothstep(0.0, 0.085, edgeDistance);
  let stillDepth = max(uniforms.sunWater.w - center.bottom, 0.0);
  let boundary = spectralBoundaryState(worldPosition(coord, dimensions), stillDepth);
  let deepWarmup = smoothstep(4.8, 9.5, stillDepth) * (1.0 - sponge) * 0.42;
  let coupling = min(1.0, dt * (12.0 * sponge + 2.4 * deepWarmup));
  nextDepth = mix(nextDepth, max(stillDepth + boundary.x, 0.0), coupling);
  // Linear shallow-water transport is q = c * eta. Multiplying by depth a
  // second time over-forces the wet/dry front and produces a vertical wall.
  nextQ = mix(nextQ, boundary.yz, coupling);

  let velocity = nextQ / max(nextDepth, MIN_DEPTH);
  let backtraceUv = clamp(uv - velocity * dt / uniforms.simulation.z, vec2<f32>(0.002), vec2<f32>(0.998));
  let backtracedFoam = textureSampleLevel(previousState, spectrumSampler, backtraceUv, 0.0).a;
  let neighbourFoam = (west.foam + east.foam + south.foam + north.foam) * 0.25;
  var foam = mix(backtracedFoam, neighbourFoam, min(0.11, dt * 1.4));
  let froude = speed / max(sqrt(GRAVITY * nextDepth), 0.001);
  let surfaceCompression = max(0.0, -(east.q.x - west.q.x + north.q.y - south.q.y) / (2.0 * cellSize));
  let breakingBirth = smoothstep(0.58, 0.92, froude) * smoothstep(0.03, 0.32, surfaceCompression);
  let shorelineBirth = (1.0 - smoothstep(0.16, 1.7, nextDepth)) * smoothstep(0.03, 0.24, speed);
  let spectralBirth = smoothstep(0.115, 0.31, boundary.w) * smoothstep(0.27, 0.76, boundary.x);
  let shorelineWaveBirth = (1.0 - smoothstep(0.10, 1.55, nextDepth)) * smoothstep(0.18, 0.64, boundary.x);
  foam *= exp(-dt * 0.58);
  foam += dt * (spectralBirth * 0.48 + breakingBirth * 2.4 + shorelineBirth * 0.52 + shorelineWaveBirth * 1.25) * params.stepFoamShift.y;
  foam = max(foam, ring * abs(params.impulse.z) * 4.0 * params.stepFoamShift.y);

  let eta = nextDepth + center.bottom - uniforms.sunWater.w;
  textureStore(nextState, coord, vec4<f32>(clamp(eta, -1.8, 1.8), clamp(nextQ, vec2<f32>(-12.0), vec2<f32>(12.0)), clamp(foam, 0.0, 1.0)));
}
`;

const BREAKER_EVENT_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var previousEvents: texture_2d<f32>;
@group(0) @binding(2) var nextEvents: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var terrainField: texture_2d<f32>;
@group(0) @binding(4) var waterState: texture_2d<f32>;
@group(0) @binding(5) var fieldSampler: sampler;
@group(0) @binding(6) var longField0: texture_2d<f32>;
@group(0) @binding(7) var longField1: texture_2d<f32>;
@group(0) @binding(8) var mediumField0: texture_2d<f32>;
@group(0) @binding(9) var mediumField1: texture_2d<f32>;
@group(0) @binding(10) var spectrumSampler: sampler;

fn frontPosition(time: f32) -> f32 {
  let travellingPhase = time * 2.4 + 12.0;
  return travellingPhase - floor(travellingPhase / 72.0) * 72.0 - 36.0;
}

fn eventHistory(coord: i32) -> f32 {
  return textureLoad(previousEvents, vec2<i32>(clamp(coord, 0, ${BREAKER_EVENT_RESOLUTION - 1}), 0), 0).r;
}

@compute @workgroup_size(64, 1)
fn updateBreakerEvents(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= ${BREAKER_EVENT_RESOLUTION}u) { return; }
  let uv = (f32(id.x) + 0.5) / ${BREAKER_EVENT_RESOLUTION}.0;
  let along = mix(-180.0, 180.0, uv);
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let time = uniforms.cameraTime.w;
  let meander = sin(along * 0.055 + time * 0.055 + 0.7) * 3.8
    + sin(along * 0.14 - time * 0.032 - 1.3) * 1.2;
  let p = tangentDirection * along + travelDirection * (frontPosition(time) + meander);

  let terrainUv = clamp(p / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let bottom = textureSampleLevel(terrainField, fieldSampler, terrainUv, 0.0).r;
  let stillDepth = max(uniforms.sunWater.w - bottom, 0.0);
  let simulationUv = (p - uniforms.simulation.xy) / uniforms.simulation.z + vec2<f32>(0.5);
  let simulationInside = step(0.0, simulationUv.x) * step(0.0, simulationUv.y) * step(simulationUv.x, 1.0) * step(simulationUv.y, 1.0);
  let state = textureSampleLevel(waterState, fieldSampler, clamp(simulationUv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0) * simulationInside;
  let dynamicDepth = max(stillDepth + state.r, 0.035);
  let speed = length(state.gb) / dynamicDepth;
  let froude = speed / max(sqrt(9.81 * dynamicDepth), 0.001);

  let longUv = fract(p / uniforms.atmosphere.z + vec2<f32>(0.5));
  let mediumUv = fract(p / uniforms.atmosphere.w + vec2<f32>(0.5));
  let long0 = textureSampleLevel(longField0, spectrumSampler, longUv, 0.0) * uniforms.waves.x;
  let long1 = textureSampleLevel(longField1, spectrumSampler, longUv, 0.0) * uniforms.waves.x;
  let medium0 = textureSampleLevel(mediumField0, spectrumSampler, mediumUv, 0.0) * uniforms.waves.x;
  let medium1 = textureSampleLevel(mediumField1, spectrumSampler, mediumUv, 0.0) * uniforms.waves.x;
  let crossDerivative = long0.a * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium0.a * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let horizontalDerivative = long1.ba * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium1.ba * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let jacobian = (1.0 + horizontalDerivative.x) * (1.0 + horizontalDerivative.y) - crossDerivative * crossDerivative;
  let compression = max(0.0, 1.0 - jacobian);
  let slope = length(long1.rg + medium1.rg);
  let spectralInstability = smoothstep(0.035, 0.160, compression)
    * mix(0.34, 1.0, smoothstep(0.045, 0.205, slope));
  let depthRatio = abs(state.r) / max(dynamicDepth, 0.12);
  let nearshoreInstability = (1.0 - smoothstep(2.2, 6.0, dynamicDepth))
    * max(smoothstep(0.46, 0.86, froude), smoothstep(0.38, 0.76, depthRatio));
  let targetInstability = clamp(max(spectralInstability, nearshoreInstability), 0.0, 1.0);

  // Lateral history diffusion gives contiguous breaking segments. Fast attack
  // and slow release implement the persistent breaking state used by practical
  // nearshore solvers instead of flickering on a single threshold crossing.
  let center = eventHistory(i32(id.x));
  let history = center * 0.50 + eventHistory(i32(id.x) - 1) * 0.25 + eventHistory(i32(id.x) + 1) * 0.25;
  let rate = select(0.62, 7.5, targetInstability > history);
  let blend = 1.0 - exp(-rate / 60.0);
  let activation = mix(history, targetInstability, blend);
  textureStore(nextEvents, vec2<i32>(i32(id.x), 0), vec4<f32>(activation, spectralInstability, nearshoreInstability, compression));
}
`;

const SPECTRUM_EVOLUTION_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var initialSpectrum: texture_2d<f32>;
@group(0) @binding(2) var waveData: texture_2d<f32>;
@group(0) @binding(3) var field0: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var field1: texture_storage_2d<rgba16float, write>;

fn complexMultiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(8, 8)
fn evolveSpectrum(@builtin(global_invocation_id) id: vec3<u32>) {
  let dimensions = textureDimensions(field0);
  if (id.x >= dimensions.x || id.y >= dimensions.y) { return; }
  let coord = vec2<i32>(id.xy);
  let initial = textureLoad(initialSpectrum, coord, 0);
  let wave = textureLoad(waveData, coord, 0);
  let phase = wave.w * uniforms.cameraTime.w;
  let exponent = vec2<f32>(cos(phase), sin(phase));
  let h = complexMultiply(initial.xy, exponent) + complexMultiply(initial.zw, vec2<f32>(exponent.x, -exponent.y));
  let ih = vec2<f32>(-h.y, h.x);
  let displacementX = ih * wave.x * wave.y;
  let displacementY = h;
  let displacementZ = ih * wave.z * wave.y;
  let displacementXdx = -h * wave.x * wave.x * wave.y;
  let displacementYdx = ih * wave.x;
  let displacementZdx = -h * wave.x * wave.z * wave.y;
  let displacementYdz = ih * wave.z;
  let displacementZdz = -h * wave.z * wave.z * wave.y;
  let dxDz = vec2<f32>(displacementX.x - displacementZ.y, displacementX.y + displacementZ.x);
  let dyDxz = vec2<f32>(displacementY.x - displacementZdx.y, displacementY.y + displacementZdx.x);
  let dyxDyz = vec2<f32>(displacementYdx.x - displacementYdz.y, displacementYdx.y + displacementYdz.x);
  let dxxDzz = vec2<f32>(displacementXdx.x - displacementZdz.y, displacementXdx.y + displacementZdz.x);
  textureStore(field0, coord, vec4<f32>(dxDz, dyDxz));
  textureStore(field1, coord, vec4<f32>(dyxDyz, dxxDzz));
}
`;

const SPECTRAL_IFFT_SHADER = /* wgsl */ `
struct Params {
  axis: u32,
  stage: u32,
  size: u32,
  finalize: u32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var twiddleTable: texture_2d<f32>;
@group(0) @binding(2) var input0: texture_2d<f32>;
@group(0) @binding(3) var input1: texture_2d<f32>;
@group(0) @binding(4) var output0: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var output1: texture_storage_2d<rgba16float, write>;

fn complexMultiply(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
  return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn butterfly(a: vec4<f32>, b: vec4<f32>, twiddle: vec2<f32>) -> vec4<f32> {
  return vec4<f32>(a.xy + complexMultiply(twiddle, b.xy), a.zw + complexMultiply(twiddle, b.zw));
}

@compute @workgroup_size(8, 8)
fn inverseFftStage(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.size || id.y >= params.size) { return; }
  let outputCoord = vec2<i32>(id.xy);
  let transformIndex = select(id.y, id.x, params.axis == 0u);
  let data = textureLoad(twiddleTable, vec2<i32>(i32(transformIndex), i32(params.stage)), 0);
  let first = i32(round(data.z));
  let second = i32(round(data.w));
  var coord0 = vec2<i32>(i32(id.x), i32(id.y));
  var coord1 = coord0;
  if (params.axis == 0u) {
    coord0.x = first;
    coord1.x = second;
  } else {
    coord0.y = first;
    coord1.y = second;
  }
  let inverseTwiddle = vec2<f32>(data.x, -data.y);
  var value0 = butterfly(textureLoad(input0, coord0, 0), textureLoad(input0, coord1, 0), inverseTwiddle);
  var value1 = butterfly(textureLoad(input1, coord0, 0), textureLoad(input1, coord1, 0), inverseTwiddle);
  if (params.finalize == 1u) {
    let checker = 1.0 - 2.0 * f32((id.x + id.y) % 2u);
    value0 *= checker;
    value1 *= checker;
  }
  textureStore(output0, outputCoord, value0);
  textureStore(output1, outputCoord, value1);
}
`;


const SKY_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
${COLOR_FUNCTIONS}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
struct Output { @builtin(position) position: vec4<f32>, @location(0) ndc: vec2<f32> }

@vertex fn skyVertex(@builtin(vertex_index) id: u32) -> Output {
  var positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  var output: Output;
  output.position = vec4<f32>(positions[id], 0.999999, 1.0);
  output.ndc = positions[id];
  return output;
}

@fragment fn skyFragment(input: Output) -> @location(0) vec4<f32> {
  let ray = normalize(uniforms.cameraForward.xyz + input.ndc.x * uniforms.cameraRight.xyz * uniforms.cameraRight.w + input.ndc.y * uniforms.cameraUp.xyz * uniforms.cameraUp.w);
  var color = skyColor(ray, uniforms.cameraTime.w, normalize(uniforms.sunWater.xyz));
  if (uniforms.terrain.w > 0.5) {
    let upward = smoothstep(-0.42, 0.72, ray.y);
    let volume = mix(vec3<f32>(0.012, 0.155, 0.158), vec3<f32>(0.050, 0.385, 0.335), upward);
    let lightColumn = pow(max(dot(ray, normalize(uniforms.sunWater.xyz)), 0.0), 14.0) * upward;
    color = volume + vec3<f32>(0.18, 0.30, 0.23) * lightColumn * 0.22;
  }
  return vec4<f32>(linearToSrgb(aces(color)), 1.0);
}
`;

// Shared aerial perspective for both the terrain and water passes. They used to
// disagree: terrain faded into the horizon while water stayed fully saturated
// out to its mesh border. At the authored 145 m fog close that border sat well
// behind the wall and never showed, but in the 10x open ocean the water's edge
// falls inside the view, so both passes have to fade on identical terms.

const TERRAIN_RENDER_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
${TETHYS_TERRAIN_WGSL}
${COLOR_FUNCTIONS}
${TETHYS_AERIAL_WGSL}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var terrainField: texture_2d<f32>;
@group(0) @binding(2) var fieldSampler: sampler;
@group(0) @binding(3) var mediumField0: texture_2d<f32>;
@group(0) @binding(4) var mediumField1: texture_2d<f32>;
@group(0) @binding(5) var shortField0: texture_2d<f32>;
@group(0) @binding(6) var shortField1: texture_2d<f32>;
@group(0) @binding(7) var spectrumSampler: sampler;
@group(0) @binding(8) var waterState: texture_2d<f32>;

struct Output {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) fieldUv: vec2<f32>,
}

@vertex fn terrainVertex(@builtin(vertex_index) vertexId: u32) -> Output {
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u)
  );
  let resolution = u32(uniforms.environment.y);
  let cellId = vertexId / 6u;
  let cell = vec2<u32>(cellId % resolution, cellId / resolution);
  let grid = cell + corners[vertexId % 6u];
  let uv = vec2<f32>(grid) / f32(resolution);
  let sample = textureSampleLevel(terrainField, fieldSampler, uv, 0.0);
  let world = vec3<f32>((uv.x - 0.5) * uniforms.terrain.x, sample.r, (uv.y - 0.5) * uniforms.terrain.x);
  var output: Output;
  output.position = uniforms.viewProj * vec4<f32>(world, 1.0);
  output.world = world;
  output.fieldUv = uv;
  return output;
}

@fragment fn terrainFragment(input: Output) -> @location(0) vec4<f32> {
  let field = textureSample(terrainField, fieldSampler, input.fieldUv);
  let normalY = sqrt(max(1.0 - field.g * field.g - field.b * field.b, 0.0001));
  let N = normalize(vec3<f32>(field.g, normalY, field.b));
  let L = normalize(uniforms.sunWater.xyz);
  let diffuse = clamp(dot(N, L) * 0.56 + 0.48, 0.0, 1.0);
  let p = input.world.xz;
  let broad = valueNoise(p * 0.075 - vec2<f32>(8.1, -2.4));
  let grain = valueNoise(p * 0.38 + vec2<f32>(4.7, -9.2));
  let geology = valueNoise(p * 0.021 + vec2<f32>(13.2, -6.7));
  let erosion = valueNoise(vec2<f32>(p.x * 0.055 + p.y * 0.018, p.y * 0.19 - p.x * 0.025));
  let sedimentMacro = valueNoise(p * 0.018 + vec2<f32>(-11.4, 6.8));
  let sedimentMeso = valueNoise(vec2<f32>(p.x * 0.092 + p.y * 0.027, p.y * 0.105 - p.x * 0.021) + vec2<f32>(3.9, -7.1));
  let simulationUv = (p - uniforms.simulation.xy) / uniforms.simulation.z + vec2<f32>(0.5);
  let inSimulation = step(0.0, simulationUv.x) * step(simulationUv.x, 1.0)
    * step(0.0, simulationUv.y) * step(simulationUv.y, 1.0);
  let shoreState = textureSample(waterState, fieldSampler, clamp(simulationUv, vec2<f32>(0.0), vec2<f32>(1.0)));
  let localWaterLevel = uniforms.sunWater.w
    + clamp(shoreState.r, -0.16, 0.18) * inSimulation * uniforms.environment.x;
  let sandSource = mix(vec3<f32>(0.22, 0.185, 0.115), vec3<f32>(0.43, 0.345, 0.19), sedimentMacro)
    * (0.88 + broad * 0.15 + grain * 0.045 + (sedimentMeso - 0.5) * 0.18);
  let granularVariation = (broad - 0.5) * 0.07 + (grain - 0.5) * 0.025;
  var color = mix(sandSource * vec3<f32>(0.74, 0.84, 0.72), vec3<f32>(0.045, 0.17, 0.145), 0.10)
    * mix(0.68, 1.08, diffuse) * (1.0 + granularVariation);
  let depth = max(0.0, localWaterLevel - input.world.y);
  let exposed = smoothstep(localWaterLevel + 0.18, localWaterLevel + 0.64, input.world.y) * uniforms.environment.x;
  let sandBase = mix(vec3<f32>(0.235, 0.135, 0.050), vec3<f32>(0.48, 0.315, 0.115), broad);
  let rockBase = mix(vec3<f32>(0.22, 0.175, 0.125), vec3<f32>(0.39, 0.285, 0.175), geology);
  let rockMask = smoothstep(0.18, 0.58, 1.0 - N.y) * 0.72 + smoothstep(0.68, 0.90, erosion) * 0.24;
  let ripplePhase = p.x * 0.21 + p.y * 0.065 + valueNoise(p * 0.030 + vec2<f32>(2.7, -5.4)) * 5.2;
  let sandRipple = sin(ripplePhase) * 0.5 + 0.5;
  let duneMacro = valueNoise(p * 0.026 + vec2<f32>(-3.4, 7.1));
  let duneMeso = valueNoise(vec2<f32>(p.x * 0.063 + p.y * 0.017, p.y * 0.071 - p.x * 0.012) + vec2<f32>(6.2, -1.8));
  let duneTone = (duneMacro - 0.5) * 0.31
    + (duneMeso - 0.5) * 0.15
    + (sandRipple - 0.5) * 0.045;
  let sunwardCrest = clamp(dot(N.xz, normalize(vec2<f32>(-0.52, -0.80))) * 0.5 + 0.5, 0.0, 1.0);
  let windPolish = smoothstep(0.64, 0.90, N.y) * smoothstep(0.48, 0.76, duneMeso);
  let sandPalette = mix(sandBase * vec3<f32>(0.84, 0.89, 0.82), sandBase * vec3<f32>(1.13, 1.07, 0.91), duneMacro);
  let elevationTone = smoothstep(localWaterLevel + 0.30, localWaterLevel + 4.8, input.world.y);
  let elevationSand = mix(vec3<f32>(0.255, 0.145, 0.052), vec3<f32>(0.53, 0.35, 0.135), elevationTone);
  let drySand = mix(sandBase, rockBase, clamp(rockMask, 0.0, 0.82))
    * mix(0.72, 1.03, diffuse)
    * (0.84 + broad * 0.15 + grain * 0.047 + (erosion - 0.5) * 0.065 + duneTone);
  let polishedSand = mix(sandPalette, elevationSand, 0.42) * mix(0.82, 1.04, diffuse);
  let drySandLayered = mix(drySand, polishedSand, 0.28 + windPolish * 0.24)
    * mix(0.83, 1.07, sunwardCrest);
  let coast = smoothstep(localWaterLevel - 0.30, localWaterLevel + 0.34, input.world.y) * uniforms.environment.x;
  let solverWash = inSimulation * uniforms.environment.x
    * smoothstep(0.018, 0.22, shoreState.a)
    * (1.0 - smoothstep(0.05, 0.62, abs(input.world.y - localWaterLevel)));
  let wetSand = mix(vec3<f32>(0.18, 0.135, 0.088), vec3<f32>(0.255, 0.185, 0.105), broad)
    * mix(0.74, 0.98, diffuse) * (0.91 + grain * 0.055 + sandRipple * 0.025);
  color = mix(color, wetSand, coast);
  color = mix(color, wetSand * vec3<f32>(0.82, 0.88, 0.84), solverWash * 0.12);
  color = mix(color, drySandLayered, exposed);
  // Project the seabed point toward the refracted sun ray before sampling the
  // actual animated surface derivatives. This keeps the caustic tied to the
  // spectral water instead of painting a cellular texture onto the sand.
  let refractedSunOffset = L.xz / max(L.y, 0.12) * depth * 0.18;
  let surfaceP = p - refractedSunOffset;
  let mediumUv = fract(surfaceP / uniforms.atmosphere.w + vec2<f32>(0.5));
  let shortUv = fract(surfaceP / ${SPECTRAL_CASCADES[2].lengthScale.toFixed(1)} + vec2<f32>(0.5));
  let medium0 = textureSample(mediumField0, spectrumSampler, mediumUv) * uniforms.waves.x;
  let medium1 = textureSample(mediumField1, spectrumSampler, mediumUv) * uniforms.waves.x;
  let short0 = textureSample(shortField0, spectrumSampler, shortUv);
  let short1 = textureSample(shortField1, spectrumSampler, shortUv);
  let mediumCross = medium0.a * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let mediumDerivative = medium1.ba * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let shortCross = short0.a * ${SPECTRAL_CASCADES[2].choppiness.toFixed(2)};
  let shortDerivative = short1.ba * ${SPECTRAL_CASCADES[2].choppiness.toFixed(2)};
  let mediumJacobian = (1.0 + mediumDerivative.x) * (1.0 + mediumDerivative.y) - mediumCross * mediumCross;
  let shortJacobian = (1.0 + shortDerivative.x) * (1.0 + shortDerivative.y) - shortCross * shortCross;
  let surfaceFocus = max(0.0, 1.0 - mediumJacobian) * 0.48 + max(0.0, 1.0 - shortJacobian) * 0.52;
  let focusedLight = pow(smoothstep(0.060, 0.27, surfaceFocus), 2.0)
    * smoothstep(0.6, 2.2, depth) * (1.0 - smoothstep(11.0, 20.0, depth));
  color *= 0.94 + focusedLight * 0.14 * (1.0 - exposed);
  color += vec3<f32>(0.095, 0.105, 0.045) * focusedLight * 0.060 * (1.0 - exposed);
  let distanceToEye = distance(uniforms.cameraTime.xyz, input.world);
  let underwater = uniforms.terrain.w > 0.5;
  let dryLand = exposed * (1.0 - select(0.0, 1.0, underwater));
  color = tethysAerialColor(color, input.world, uniforms.cameraTime.xyz, uniforms.environment.w, underwater, dryLand);
  return vec4<f32>(linearToSrgb(aces(color)), 1.0);
}
`;

const SCENE_BLIT_SHADER = /* wgsl */ `
@group(0) @binding(0) var sceneColor: texture_2d<f32>;
@group(0) @binding(1) var sceneSampler: sampler;

struct BlitOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex fn blitVertex(@builtin(vertex_index) vertexId: u32) -> BlitOutput {
  let positions = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
  let position = positions[vertexId];
  var output: BlitOutput;
  output.position = vec4<f32>(position, 0.0, 1.0);
  output.uv = vec2<f32>(position.x * 0.5 + 0.5, 0.5 - position.y * 0.5);
  return output;
}

@fragment fn blitFragment(input: BlitOutput) -> @location(0) vec4<f32> {
  return textureSample(sceneColor, sceneSampler, input.uv);
}
`;

const WATER_RENDER_SHADER = /* wgsl */ `
override REFERENCE_MODE: bool = false;
${WORLD_UNIFORMS}
${TETHYS_TERRAIN_WGSL}
${COLOR_FUNCTIONS}
${TETHYS_AERIAL_WGSL}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var terrainField: texture_2d<f32>;
@group(0) @binding(2) var waterState: texture_2d<f32>;
@group(0) @binding(3) var fieldSampler: sampler;
@group(0) @binding(4) var longField0: texture_2d<f32>;
@group(0) @binding(5) var longField1: texture_2d<f32>;
@group(0) @binding(6) var mediumField0: texture_2d<f32>;
@group(0) @binding(7) var mediumField1: texture_2d<f32>;
@group(0) @binding(8) var shortField0: texture_2d<f32>;
@group(0) @binding(9) var shortField1: texture_2d<f32>;
@group(0) @binding(10) var spectrumSampler: sampler;
@group(0) @binding(11) var breakerEvents: texture_2d<f32>;
@group(1) @binding(0) var sceneColorTexture: texture_2d<f32>;
@group(1) @binding(1) var sceneDepthTexture: texture_depth_2d;
@group(1) @binding(2) var sceneColorSampler: sampler;

struct Output {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) fieldUv: vec2<f32>,
  @location(3) simulationUv: vec2<f32>,
  @location(4) waveHeight: f32,
  @location(5) compression: f32,
  @location(6) breakerLip: f32,
  @location(7) breakerCoord: vec2<f32>,
  @location(8) surfaceKind: f32,
}

struct SurfaceEvaluation {
  world: vec3<f32>,
  tangentPX: vec3<f32>,
  tangentPZ: vec3<f32>,
  fieldUv: vec2<f32>,
  simulationUv: vec2<f32>,
  waveHeight: f32,
  compression: f32,
  breakerLip: f32,
}

fn simulationSample(uv: vec2<f32>) -> vec4<f32> {
  let inside = step(0.0, uv.x) * step(0.0, uv.y) * step(uv.x, 1.0) * step(uv.y, 1.0);
  return textureSampleLevel(waterState, fieldSampler, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0) * inside;
}

fn dielectricFresnel(cosine: f32) -> f32 {
  let eta = 1.0 / 1.333;
  let sinTransmittedSquared = eta * eta * max(0.0, 1.0 - cosine * cosine);
  if (sinTransmittedSquared >= 1.0) { return 1.0; }
  let transmittedCosine = sqrt(max(0.0, 1.0 - sinTransmittedSquared));
  let parallel = (cosine - 1.333 * transmittedCosine) / max(cosine + 1.333 * transmittedCosine, 0.0001);
  let perpendicular = (transmittedCosine - 1.333 * cosine) / max(transmittedCosine + 1.333 * cosine, 0.0001);
  return 0.5 * (parallel * parallel + perpendicular * perpendicular);
}

fn smithVisibility(cosine: f32, meanSquareSlope: f32) -> f32 {
  let tangentSquared = max(0.0, 1.0 - cosine * cosine) / max(cosine * cosine, 0.0001);
  return 2.0 / (1.0 + sqrt(1.0 + meanSquareSlope * tangentSquared));
}

fn oceanSunGlitter(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, extraVariance: f32) -> f32 {
  let ndv = max(dot(N, V), 0.001);
  let ndl = max(dot(N, L), 0.001);
  let H = normalize(V + L);
  let ndh = max(dot(N, H), 0.001);
  let windWorld = normalize(vec3<f32>(0.887, 0.0, -0.462));
  let T = normalize(windWorld - N * dot(windWorld, N));
  let B = normalize(cross(N, T));
  let alongSlope = dot(H, T) / ndh;
  let acrossSlope = dot(H, B) / ndh;
  // Cox-Munk clean-sea mean-square slopes at an 11.5 m/s wind. extraVariance
  // carries the capillary slope that distance faded out of the normal: this
  // distribution is exactly where sub-resolution slope belongs, so returning it
  // here broadens the glitter instead of letting the far water go mirror-flat.
  let alongVariance = 0.0363 + extraVariance;
  let acrossVariance = 0.0251 + extraVariance;
  let slopePdf = exp(-0.5 * (alongSlope * alongSlope / alongVariance + acrossSlope * acrossSlope / acrossVariance))
    / (6.2831853 * sqrt(alongVariance * acrossVariance));
  let facetDistribution = slopePdf / max(ndh * ndh * ndh * ndh, 0.0001);
  let visibility = smithVisibility(ndv, 0.0307 + extraVariance) * smithVisibility(ndl, 0.0307 + extraVariance);
  return dielectricFresnel(max(dot(V, H), 0.0)) * facetDistribution * visibility / max(4.0 * ndv, 0.001);
}

fn breakerFrontPosition(time: f32) -> f32 {
  let travellingPhase = time * 2.4 + 12.0;
  return travellingPhase - floor(travellingPhase / 72.0) * 72.0 - 36.0;
}

fn breakerFrontVisibility(front: f32) -> f32 {
  // Fade the front out before its periodic reset, then reintroduce it from the
  // opposite side. This avoids a 72 m position pop in both geometry and the
  // adaptive sampling warp.
  return 1.0 - smoothstep(28.0, 35.0, abs(front));
}

fn breakerEventActivation(along: f32) -> f32 {
  let uv = vec2<f32>(clamp(along / 360.0 + 0.5, 0.0, 1.0), 0.5);
  return smoothstep(0.035, 0.68, textureSampleLevel(breakerEvents, fieldSampler, uv, 0.0).r);
}

fn breakerCoordinates(p: vec2<f32>, time: f32) -> vec2<f32> {
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let along = dot(p, tangentDirection);
  let meander = sin(along * 0.055 + time * 0.055 + 0.7) * 3.8
    + sin(along * 0.14 - time * 0.032 - 1.3) * 1.2;
  let signedDistance = dot(p, travelDirection) - breakerFrontPosition(time) - meander;
  return vec2<f32>(signedDistance, along);
}

fn adaptiveBreakerCoordinates(p: vec2<f32>, time: f32) -> vec2<f32> {
  // Concentrate the existing uniform-grid samples around the moving front.
  // The linear compensation makes the warp approach zero at the domain edge,
  // so this is a redistribution of vertices rather than an expanding patch.
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let across = dot(p, travelDirection);
  let along = dot(p, tangentDirection);
  let front = breakerFrontPosition(time);
  let domainHalfDiagonal = 276.0;
  let concentration = 8.2 * breakerFrontVisibility(front) * breakerEventActivation(along) * ${BREAKER_SHADER_GATE};
  let bandWidth = 12.5;
  // tanh must see a bounded argument: Metal's tanh overflows to NaN past
  // roughly |x| = 89, and the disabled-breaker gate multiplies by 0 only
  // *after* -- 0 * NaN is still NaN, which drops every far-field vertex it
  // touches and tears a cell-quantised wedge out of the downwind horizon.
  let correction = -concentration * tanh(clamp((across - front) / bandWidth, -30.0, 30.0))
    + concentration * across / domainHalfDiagonal;
  return p + travelDirection * correction;
}

fn localizedBreakerDisplacement(p: vec2<f32>, time: f32) -> vec4<f32> {
  // A travelling, meandering nonlinear wavefront blended into the same water
  // parameterization. Unlike the rejected detached crest sheet, its edges
  // converge to the spectral surface and its horizontal motion can fold the
  // grid naturally when the crest becomes steep.
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let breakerCoord = breakerCoordinates(p, time);
  let along = breakerCoord.y;
  let travellingFront = breakerFrontPosition(time);
  let frontVisibility = breakerFrontVisibility(travellingFront);
  let signedDistance = breakerCoord.x;
  let u = signedDistance / 9.0;
  let edgeWindow = 1.0 - smoothstep(0.82, 1.34, abs(u));
  let localEnvelope = exp(-0.55 * u * u) * edgeWindow;
  let phase = 3.14159265 * u;
  let crestProfile = cos(phase);
  let alongSignal = sin(along * 0.041 + time * 0.08 + 0.4)
    + 0.48 * sin(along * 0.097 - time * 0.035 - 1.2);
  let alongBreakup = smoothstep(-0.35, 0.72, alongSignal);
  let alongVariation = (0.42 + 0.58 * alongBreakup)
    * (0.90 + 0.10 * sin(along * 0.092 + 1.8));
  let vertical = 2.45 * localEnvelope * crestProfile * alongVariation;
  var horizontal = -travelDirection * 2.15 * localEnvelope * sin(phase) * alongVariation;
  let lip = pow(max(crestProfile, 0.0), 3.0) * localEnvelope;
  horizontal += travelDirection * 0.72 * lip * alongVariation;
  let activation = breakerEventActivation(along);
  return vec4<f32>(horizontal.x, vertical, horizontal.y, lip) * frontVisibility * activation * (1.0 - uniforms.environment.x) * ${BREAKER_SHADER_GATE};
}

fn evaluateWaterSurface(p: vec2<f32>) -> SurfaceEvaluation {
  let fieldUv = clamp(p / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let terrain = textureSampleLevel(terrainField, fieldSampler, fieldUv, 0.0);
  let depth = uniforms.sunWater.w - terrain.r;
  let shallowAttenuation = smoothstep(0.14, 2.7, depth);
  let simUv = (p - uniforms.simulation.xy) / uniforms.simulation.z + vec2<f32>(0.5);
  let longUv = fract(p / uniforms.atmosphere.z + vec2<f32>(0.5));
  let mediumUv = fract(p / uniforms.atmosphere.w + vec2<f32>(0.5));
  let long0 = textureSampleLevel(longField0, spectrumSampler, longUv, 0.0) * uniforms.waves.x;
  let long1 = textureSampleLevel(longField1, spectrumSampler, longUv, 0.0) * uniforms.waves.x;
  let medium0 = textureSampleLevel(mediumField0, spectrumSampler, mediumUv, 0.0) * uniforms.waves.x;
  let medium1 = textureSampleLevel(mediumField1, spectrumSampler, mediumUv, 0.0) * uniforms.waves.x;
  let horizontalDisplacement = long0.rg * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium0.rg * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let longHeight = long0.b;
  let mediumHeight = medium0.b;
  let spectralHeight = longHeight + mediumHeight
    + 0.14 * (longHeight * longHeight - 0.080 * uniforms.waves.y)
    + 0.32 * (mediumHeight * mediumHeight - 0.030 * uniforms.waves.y);
  let crossDerivative = long0.a * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium0.a * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let longSlope = long1.rg * (1.0 + 0.28 * longHeight);
  let mediumSlope = medium1.rg * (1.0 + 0.64 * mediumHeight);
  let spectralSlope = longSlope + mediumSlope;
  let horizontalDerivative = long1.ba * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} + medium1.ba * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)};
  let sim = simulationSample(simUv);
  let texel = uniforms.simulation.w;
  let left = simulationSample(simUv - vec2<f32>(texel, 0.0)).r;
  let right = simulationSample(simUv + vec2<f32>(texel, 0.0)).r;
  let back = simulationSample(simUv - vec2<f32>(0.0, texel)).r;
  let front = simulationSample(simUv + vec2<f32>(0.0, texel)).r;
  let worldTexel = uniforms.simulation.z * texel;
  let simulationDerivative = vec2<f32>(right - left, front - back) / max(worldTexel * 2.0, 0.001);
  let simulationEdge = min(min(simUv.x, 1.0 - simUv.x), min(simUv.y, 1.0 - simUv.y));
  let simulationCoverage = step(0.0, simulationEdge) * smoothstep(0.008, 0.055, simulationEdge);
  let baseJacobian = (1.0 + horizontalDerivative.x) * (1.0 + horizontalDerivative.y) - crossDerivative * crossDerivative;
  let breaker = localizedBreakerDisplacement(p, uniforms.cameraTime.w) * shallowAttenuation;
  let breakerStep = 0.55;
  let breakerDx = (localizedBreakerDisplacement(p + vec2<f32>(breakerStep, 0.0), uniforms.cameraTime.w) * shallowAttenuation - breaker) / breakerStep;
  let breakerDz = (localizedBreakerDisplacement(p + vec2<f32>(0.0, breakerStep), uniforms.cameraTime.w) * shallowAttenuation - breaker) / breakerStep;
  // The nonlinear field is a replacement for the far FFT within its domain,
  // not an additive wake texture. Its relaxation band already matches the FFT,
  // and this narrow geometric blend hides the finite-domain edge.
  let nearshoreOwnership = simulationCoverage * (1.0 - smoothstep(3.8, 5.55, depth));
  let wave = mix(spectralHeight * shallowAttenuation, sim.r, nearshoreOwnership) + breaker.y;
  let world = vec3<f32>(
    p.x + horizontalDisplacement.x * shallowAttenuation + breaker.x,
    uniforms.sunWater.w + wave,
    p.y + horizontalDisplacement.y * shallowAttenuation + breaker.z
  );
  let blendedSlope = mix(spectralSlope * shallowAttenuation, simulationDerivative, nearshoreOwnership);
  let tangentPX = vec3<f32>(1.0 + horizontalDerivative.x * shallowAttenuation + breakerDx.x, blendedSlope.x + breakerDx.y, crossDerivative * shallowAttenuation + breakerDx.z);
  let tangentPZ = vec3<f32>(crossDerivative * shallowAttenuation + breakerDz.x, blendedSlope.y + breakerDz.y, 1.0 + horizontalDerivative.y * shallowAttenuation + breakerDz.z);
  var result: SurfaceEvaluation;
  result.world = world;
  result.tangentPX = tangentPX;
  result.tangentPZ = tangentPZ;
  result.fieldUv = fieldUv;
  result.simulationUv = simUv;
  result.waveHeight = wave;
  result.compression = max(0.0, 1.0 - baseJacobian) * shallowAttenuation + breaker.w * 0.38;
  result.breakerLip = breaker.w;
  return result;
}

@vertex fn waterVertex(@builtin(vertex_index) vertexId: u32, @builtin(instance_index) instanceId: u32) -> Output {
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u)
  );
  var onHorizonSkirt = false;
  let resolution = ${WATER_CLIPMAP_RESOLUTION}u;
  let cellId = vertexId / 6u;
  let cell = vec2<u32>(cellId % resolution, cellId / resolution);
  let grid = cell + corners[vertexId % 6u];
  let uv = vec2<f32>(grid) / f32(resolution);
  // Both scenes share the camera-snapped clipmap: the island scene is the
  // same open ocean with the authored archipelago exposed at the world centre.
  var baseP = vec2<f32>(0.0);
  {
    let level = f32(instanceId);
    let halfExtent = 32.0 * exp2(level);
    let cellSize = halfExtent * 2.0 / f32(resolution);
    let snappedCamera = floor(uniforms.cameraTime.xz / cellSize) * cellSize;
    baseP = snappedCamera + (uv - vec2<f32>(0.5)) * halfExtent * 2.0;
    if (instanceId > 0u) {
      // Degenerate the covered centre of each coarser level. A one-cell
      // underlap keeps T-junctions hidden while all ring origins stay snapped.
      let innerHalf = halfExtent * 0.5 - cellSize;
      let cellCenter = snappedCamera + ((vec2<f32>(cell) + vec2<f32>(0.5)) / f32(resolution) - vec2<f32>(0.5)) * halfExtent * 2.0;
      if (all(abs(cellCenter - snappedCamera) < vec2<f32>(innerHalf))) {
        baseP = vec2<f32>(10000.0);
      }
    }
    // The rings are a finite square, so their outer edge is a visible cut
    // wherever fog does not reach it. Push the outermost ring of vertices out
    // to the horizon: the last row of quads becomes a skirt that closes the gap
    // to the skyline. Those triangles are enormous but land within a few pixels
    // of the horizon, where the surface is far below one sample per wave
    // anyway. Without this, removing the fog wall just exposes the cut.
    if (instanceId == ${WATER_CLIPMAP_LEVELS - 1}u) {
      onHorizonSkirt = grid.x == 0u || grid.x == resolution || grid.y == 0u || grid.y == resolution;
      if (onHorizonSkirt) {
        let outward = baseP - snappedCamera;
        let reach = max(abs(outward.x), abs(outward.y));
        baseP = snappedCamera + outward * (${WATER_HORIZON_REACH}.0 / max(reach, 1.0));
      }
    }
  }
  let coordinateStep = 0.55;
  let p = adaptiveBreakerCoordinates(baseP, uniforms.cameraTime.w);
  let pDx = (adaptiveBreakerCoordinates(baseP + vec2<f32>(coordinateStep, 0.0), uniforms.cameraTime.w) - p) / coordinateStep;
  let pDz = (adaptiveBreakerCoordinates(baseP + vec2<f32>(0.0, coordinateStep), uniforms.cameraTime.w) - p) / coordinateStep;
  let surface = evaluateWaterSurface(p);
  let tangentX = surface.tangentPX * pDx.x + surface.tangentPZ * pDx.y;
  let tangentZ = surface.tangentPX * pDz.x + surface.tangentPZ * pDz.y;
  let breakerCoord = breakerCoordinates(p, uniforms.cameraTime.w);
  var output: Output;
  if (onHorizonSkirt) {
    // Project the skirt vertex as a direction so it lands on the horizon, and
    // flatten the direction's vertical component first: from a high orbit the
    // eye-to-vertex drop would otherwise depress the skirt's square rim by
    // atan(eyeHeight / 20 km) below the true horizon, exposing its corners.
    var towardHorizon = surface.world - uniforms.cameraTime.xyz;
    towardHorizon.y = 0.0;
    var horizonClip = uniforms.viewProj * vec4<f32>(towardHorizon, 0.0);
    horizonClip.z = horizonClip.w * 0.99999;
    output.position = horizonClip;
  } else {
    output.position = uniforms.viewProj * vec4<f32>(surface.world, 1.0);
  }
  output.world = surface.world;
  output.normal = normalize(cross(tangentZ, tangentX));
  output.fieldUv = surface.fieldUv;
  output.simulationUv = surface.simulationUv;
  output.waveHeight = surface.waveHeight;
  output.compression = surface.compression;
  output.breakerLip = surface.breakerLip;
  output.breakerCoord = breakerCoord;
  output.surfaceKind = 0.0;
  return output;
}

fn breakerPatchBreakup(along: f32, time: f32) -> f32 {
  let breakupSignal = sin(along * 0.041 + time * 0.08 + 0.4)
    + 0.48 * sin(along * 0.097 - time * 0.035 - 1.2);
  return smoothstep(-0.10, 0.65, breakupSignal);
}

fn breakerPatchExtra(across: f32, along: f32, time: f32) -> vec3<f32> {
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let frontVisibility = breakerFrontVisibility(breakerFrontPosition(time));
  let u = across / 9.0;
  let edgeWindow = 1.0 - smoothstep(0.70, 1.24, abs(u));
  let alongWindow = 1.0 - smoothstep(158.0, 176.0, abs(along));
  let envelope = exp(-0.58 * u * u) * edgeWindow * alongWindow * frontVisibility * breakerEventActivation(along) * (1.0 - uniforms.environment.x) * ${BREAKER_SHADER_GATE};
  let phase = 3.14159265 * u;
  let alongVariation = 0.86 + 0.14 * sin(along * 0.092 + 1.8);
  let breakup = 0.24 + 0.76 * breakerPatchBreakup(along, time);
  let lip = pow(max(cos(phase), 0.0), 3.0);
  // A narrow crest-nose correction rather than another full wave profile.
  // The base spectral/bound-harmonic surface owns the body of the wave; this
  // only rounds and leans locally breaking sections without forming a shelf.
  let horizontalAmount = 0.62 * breakup * lip * envelope * alongVariation;
  let vertical = 0.14 * breakup * lip * envelope * alongVariation;
  return vec3<f32>(travelDirection.x * horizontalAmount, vertical, travelDirection.y * horizontalAmount);
}

@vertex fn breakerPatchVertex(@builtin(vertex_index) vertexId: u32) -> Output {
  var corners = array<vec2<u32>, 6>(
    vec2<u32>(0u, 0u), vec2<u32>(0u, 1u), vec2<u32>(1u, 0u),
    vec2<u32>(0u, 1u), vec2<u32>(1u, 1u), vec2<u32>(1u, 0u)
  );
  let alongResolution = 256u;
  let acrossResolution = 48u;
  let cellId = vertexId / 6u;
  let cell = vec2<u32>(cellId % alongResolution, cellId / alongResolution);
  let grid = cell + corners[vertexId % 6u];
  let uv = vec2<f32>(grid) / vec2<f32>(f32(alongResolution), f32(acrossResolution));
  let along = mix(-180.0, 180.0, uv.x);
  let across = mix(-12.0, 12.0, uv.y);
  let travelDirection = normalize(vec2<f32>(0.887, -0.462));
  let tangentDirection = vec2<f32>(-travelDirection.y, travelDirection.x);
  let time = uniforms.cameraTime.w;
  let meander = sin(along * 0.055 + time * 0.055 + 0.7) * 3.8
    + sin(along * 0.14 - time * 0.032 - 1.3) * 1.2;
  let meanderDerivative = cos(along * 0.055 + time * 0.055 + 0.7) * 0.209
    + cos(along * 0.14 - time * 0.032 - 1.3) * 0.168;
  let p = tangentDirection * along + travelDirection * (breakerFrontPosition(time) + meander + across);
  let pAlong = tangentDirection + travelDirection * meanderDerivative;
  let pAcross = travelDirection;
  let surface = evaluateWaterSurface(p);
  let extra = breakerPatchExtra(across, along, time);
  let derivativeStep = 0.12;
  let extraAlong = (breakerPatchExtra(across, along + derivativeStep, time) - breakerPatchExtra(across, along - derivativeStep, time)) / (2.0 * derivativeStep);
  let extraAcross = (breakerPatchExtra(across + derivativeStep, along, time) - breakerPatchExtra(across - derivativeStep, along, time)) / (2.0 * derivativeStep);
  let tangentAlong = surface.tangentPX * pAlong.x + surface.tangentPZ * pAlong.y + extraAlong;
  let tangentAcross = surface.tangentPX * pAcross.x + surface.tangentPZ * pAcross.y + extraAcross;
  let world = surface.world + extra;
  var output: Output;
  output.position = uniforms.viewProj * vec4<f32>(world, 1.0);
  output.world = world;
  output.normal = normalize(cross(tangentAlong, tangentAcross));
  output.fieldUv = surface.fieldUv;
  output.simulationUv = surface.simulationUv;
  output.waveHeight = surface.waveHeight + extra.y;
  let patchBreakup = breakerPatchBreakup(along, time);
  output.compression = surface.compression + smoothstep(0.62, 0.94, patchBreakup) * (1.0 - smoothstep(0.3, 4.5, abs(across))) * 0.055;
  output.breakerLip = surface.breakerLip;
  output.breakerCoord = vec2<f32>(across, along);
  output.surfaceKind = 1.0;
  return output;
}

@fragment fn waterFragment(input: Output) -> @location(0) vec4<f32> {
  let patchVisible = breakerFrontVisibility(breakerFrontPosition(uniforms.cameraTime.w)) * (1.0 - uniforms.environment.x) * ${BREAKER_SHADER_GATE};
  let patchAlong = 1.0 - smoothstep(158.0, 176.0, abs(input.breakerCoord.y));
  if (input.surfaceKind < 0.5 && patchVisible * patchAlong > 0.001 && abs(input.breakerCoord.x) < 11.72) { discard; }
  if (input.surfaceKind > 0.5 && (patchVisible <= 0.001 || abs(input.breakerCoord.x) > 11.82 || patchAlong <= 0.001)) { discard; }
  let state = simulationSample(input.simulationUv);
  // Clamp rather than discard past the terrain field. The vertex stage already
  // clamps this same lookup, so discarding here cut the water along the field
  // border while the surface it was shaded from continued -- invisible when
  // fog closed at 145 m, but a hard sawtooth edge with bare seabed behind it
  // once the open ocean reaches 1450 m. Outside the authored centre the field
  // border is flat -8.5 m seabed, so the clamped depth is the correct one.
  let displacedTerrainUv = clamp(input.world.xz / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let terrain = textureSample(terrainField, fieldSampler, displacedTerrainUv);
  // Coverage must be derived from the same displaced surface that produced
  // the raster depth. Re-evaluating height per fragment makes colour and depth
  // disagree at wet/dry intersections, exposing a checkerboard of triangles.
  let waterColumn = input.world.y - terrain.r;
  let shorelineWidth = clamp(fwidth(waterColumn), 0.006, 0.06);
  // Leave a centimetre-scale wet-sand margin in the island scene. Rendering
  // translucent water almost coplanar with terrain is visually unstable and
  // was the remaining source of dotted/checker shoreline fragments.
  let shorelineThreshold = mix(0.018, 0.28, uniforms.environment.x);
  let shorelineCoverage = smoothstep(shorelineThreshold - shorelineWidth, shorelineThreshold + shorelineWidth, waterColumn);
  if (shorelineCoverage < 0.01) { discard; }
  let depth = max(waterColumn, 0.018);
  let p = input.world.xz;
  let time = uniforms.cameraTime.w;
  let shortUv = fract(p / ${SPECTRAL_CASCADES[2].lengthScale.toFixed(1)} + vec2<f32>(0.5));
  let short0 = textureSample(shortField0, spectrumSampler, shortUv);
  let short1 = textureSample(shortField1, spectrumSampler, shortUv);
  // Capillary detail is dropped as it approaches the sampling limit, because
  // sub-pixel waves alias into crawling highlights. The threshold is a
  // quality/stability trade-off rather than a constant, so it is exposed.
  //
  // The test is screen-space sampling density, not world distance. Fading over
  // a fixed 42-118 m band looks abrupt: screen row maps to distance as 1/d, so
  // the far half of that band collapses into a few dozen pixels and the detail
  // appears to switch off. Pixels-per-wavelength is uniform in screen space, so
  // the ramp reads evenly, and it self-adjusts to field of view, resolution and
  // render scale instead of assuming the authored camera.
  let detailRange = uniforms.waves.w;
  let eyeDistance = distance(uniforms.cameraTime.xyz, input.world);
  let pixelWorldSize = eyeDistance * 2.0 * uniforms.cameraUp.w / max(uniforms.interaction.w, 1.0);
  // Representative wavelength of the capillary cascade's energy peak.
  let pixelsPerWave = ${SPECTRAL_CASCADES[2].lengthScale.toFixed(1)} / 12.0 / max(pixelWorldSize, 1e-6);
  let shortDistanceFade = smoothstep(3.0, 14.0, pixelsPerWave * detailRange);
  let shortSlope = short1.rg * shortDistanceFade;
  // Short waves become an aggregate slope distribution instead of a literal
  // high-frequency normal texture. This is the geometry-to-BRDF transition
  // used to avoid sparkling/streaking as sub-pixel waves recede.
  // Cascade 0/1 slopes are re-sampled here rather than interpolated from the
  // vertices: the clipmap doubles its cell size every ring, so a few hundred
  // metres out the grid undersamples the 64 m cascade and vertex-rate normals
  // shade as cell-sized facets with a visible seam at every ring boundary.
  // Geometry stays vertex-rate; only the shading normal is refined. The
  // pre-displacement surface parameter is recovered from the simulation UV,
  // which is affine in it and never clamped in the vertex stage.
  let surfaceParam = (input.simulationUv - vec2<f32>(0.5)) * uniforms.simulation.z + uniforms.simulation.xy;
  let paramFieldUv = clamp(surfaceParam / uniforms.terrain.x + vec2<f32>(0.5), vec2<f32>(0.0), vec2<f32>(1.0));
  let paramDepth = uniforms.sunWater.w - textureSample(terrainField, fieldSampler, paramFieldUv).r;
  let paramAttenuation = smoothstep(0.14, 2.7, paramDepth);
  let longUvF = fract(surfaceParam / uniforms.atmosphere.z + vec2<f32>(0.5));
  let mediumUvF = fract(surfaceParam / uniforms.atmosphere.w + vec2<f32>(0.5));
  let long0F = textureSample(longField0, spectrumSampler, longUvF) * uniforms.waves.x;
  let long1F = textureSample(longField1, spectrumSampler, longUvF) * uniforms.waves.x;
  let medium0F = textureSample(mediumField0, spectrumSampler, mediumUvF) * uniforms.waves.x;
  let medium1F = textureSample(mediumField1, spectrumSampler, mediumUvF) * uniforms.waves.x;
  // The same screen-space sampling-rate fade the capillary cascade gets, per
  // cascade: near the horizon one pixel spans many medium wavelengths, and
  // per-fragment slopes alias into a glittering noise band there. Each faded
  // slope joins the recovered-variance path below, so the distant-roughness
  // control keeps deciding whether the energy returns as BRDF roughness.
  let mediumPixelsPerWave = uniforms.atmosphere.w / 8.0 / max(pixelWorldSize, 1e-6);
  let swellSmoothing = uniforms.atmosphere.y;
  let mediumFadeF = select(smoothstep(3.0, 14.0, mediumPixelsPerWave * detailRange / max(swellSmoothing, 0.001)), 1.0, swellSmoothing <= 0.0);
  let longPixelsPerWave = uniforms.atmosphere.z / 5.0 / max(pixelWorldSize, 1e-6);
  let longFadeF = select(smoothstep(3.0, 14.0, longPixelsPerWave * detailRange / max(swellSmoothing, 0.001)), 1.0, swellSmoothing <= 0.0);
  let crossDerivativeF = long0F.a * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} * longFadeF + medium0F.a * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)} * mediumFadeF;
  let spectralSlopeF = long1F.rg * (1.0 + 0.28 * long0F.b) * longFadeF + medium1F.rg * (1.0 + 0.64 * medium0F.b) * mediumFadeF;
  let horizontalDerivativeF = long1F.ba * ${SPECTRAL_CASCADES[0].choppiness.toFixed(2)} * longFadeF + medium1F.ba * ${SPECTRAL_CASCADES[1].choppiness.toFixed(2)} * mediumFadeF;
  let simTexelF = uniforms.simulation.w;
  let simLeftF = simulationSample(input.simulationUv - vec2<f32>(simTexelF, 0.0)).r;
  let simRightF = simulationSample(input.simulationUv + vec2<f32>(simTexelF, 0.0)).r;
  let simBackF = simulationSample(input.simulationUv - vec2<f32>(0.0, simTexelF)).r;
  let simFrontF = simulationSample(input.simulationUv + vec2<f32>(0.0, simTexelF)).r;
  let simulationDerivativeF = vec2<f32>(simRightF - simLeftF, simFrontF - simBackF) / max(uniforms.simulation.z * simTexelF * 2.0, 0.001);
  let simulationEdgeF = min(min(input.simulationUv.x, 1.0 - input.simulationUv.x), min(input.simulationUv.y, 1.0 - input.simulationUv.y));
  let simulationCoverageF = step(0.0, simulationEdgeF) * smoothstep(0.008, 0.055, simulationEdgeF);
  let nearshoreOwnershipF = simulationCoverageF * (1.0 - smoothstep(3.8, 5.55, paramDepth));
  let blendedSlopeF = mix(spectralSlopeF * paramAttenuation, simulationDerivativeF, nearshoreOwnershipF);
  let tangentXF = vec3<f32>(1.0 + horizontalDerivativeF.x * paramAttenuation, blendedSlopeF.x, crossDerivativeF * paramAttenuation);
  let tangentZF = vec3<f32>(crossDerivativeF * paramAttenuation, blendedSlopeF.y, 1.0 + horizontalDerivativeF.y * paramAttenuation);
  // The breaker patch carries bespoke crest normals from its vertex stage and
  // keeps them; everything else takes the refined per-fragment normal.
  var baseNormal = normalize(cross(tangentZF, tangentXF));
  if (input.surfaceKind > 0.5) { baseNormal = normalize(input.normal); }
  var N = normalize(baseNormal + vec3<f32>(-shortSlope.x, 0.0, -shortSlope.y) * 0.42);
  // Fading the capillary slope out of the normal is what stops sub-pixel waves
  // from sparkling, but on its own it also drains the roughness that those
  // waves represent, so the far surface collapses toward a mirror. Feed the
  // discarded slope back in as an aggregate statistic instead: the normal stays
  // smooth while the BRDF keeps the energy. At 0 this is the original
  // behaviour; at 1 the full variance is retained.
  // Variance is the square of slope, and it is what the Cox-Munk distribution
  // in oceanSunGlitter consumes.
  let fadedSlope = length(short1.rg) * (1.0 - shortDistanceFade)
    + length(medium1F.rg) * (1.0 - mediumFadeF)
    + length(long1F.rg) * (1.0 - longFadeF);
  let recoveredVariance = fadedSlope * fadedSlope * uniforms.waves.z;
  let surfaceRoughness = mix(0.035, 0.115, smoothstep(0.012, 0.30, length(shortSlope) + fadedSlope * uniforms.waves.z));
  let underwater = uniforms.terrain.w > 0.5;
  if (underwater) { N *= -1.0; }
  let V = normalize(uniforms.cameraTime.xyz - input.world);
  let L = normalize(uniforms.sunWater.xyz);
  let ndv = clamp(abs(dot(N, V)), 0.0, 1.0);
  let fresnel = dielectricFresnel(ndv);
  let terrainNormalY = sqrt(max(1.0 - terrain.g * terrain.g - terrain.b * terrain.b, 0.0001));
  let floorLight = clamp(dot(normalize(vec3<f32>(terrain.g, terrainNormalY, terrain.b)), L) * 0.56 + 0.48, 0.0, 1.0);
  let refractedOffset = N.xz * depth * mix(0.42, 1.25, 1.0 - ndv);
  let refractedP = p + refractedOffset;
  let sandVariation = valueNoise(refractedP * 0.18) * 0.055 + valueNoise(refractedP * 0.62 + vec2<f32>(7.1, -3.4)) * 0.018;
  var floorColor = (vec3<f32>(0.46, 0.37, 0.225) + vec3<f32>(sandVariation)) * mix(0.70, 1.02, floorLight);
  let screenUv = input.position.xy / max(uniforms.interaction.zw, vec2<f32>(1.0));
  let refractionUv = clamp(screenUv + vec2<f32>(N.x, -N.z) * (0.0025 + min(depth, 14.0) * 0.00072), vec2<f32>(0.001), vec2<f32>(0.999));
  var capturedLinear = floorColor;
  if (uniforms.environment.x > 0.5) {
    let sceneDimensions = textureDimensions(sceneDepthTexture);
    let sceneCoord = vec2<i32>(clamp(refractionUv * vec2<f32>(sceneDimensions), vec2<f32>(0.0), vec2<f32>(sceneDimensions - vec2<u32>(1u))));
    let capturedDepth = textureLoad(sceneDepthTexture, sceneCoord, 0);
    let capturedScene = textureSample(sceneColorTexture, sceneColorSampler, refractionUv).rgb;
    capturedLinear = pow(max(capturedScene, vec3<f32>(0.0)), vec3<f32>(2.2));
    let capturedGeometry = 1.0 - step(0.9995, capturedDepth);
    floorColor = mix(floorColor, capturedLinear, capturedGeometry * 0.88);
  }
  let opticalDepth = select(depth / max(ndv, 0.28), max(0.0, uniforms.sunWater.w - uniforms.cameraTime.y) / max(abs(dot(N, V)), 0.32), underwater);
  let absorption = vec3<f32>(0.37, 0.125, 0.054);
  let transmission = exp(-absorption * min(opticalDepth, 24.0));
  // Open water is not a cyan diffuse material.  Keep the in-scattered body
  // colour low-energy so the interface reflection supplies the bright values.
  let scatterColor = vec3<f32>(0.0035, 0.096, 0.092);
  let scatterAmount = vec3<f32>(1.0) - transmission;
  let phaseG = 0.24;
  let lightCosine = dot(-V, L);
  let phase = (1.0 - phaseG * phaseG) / pow(max(1.0 + phaseG * phaseG - 2.0 * phaseG * lightCosine, 0.04), 1.5);
  var refracted = floorColor * transmission + scatterColor * scatterAmount * (0.72 + phase * 0.060);
  let refractionSoftness = smoothstep(3.0, 15.0, opticalDepth) * 0.08;
  refracted = mix(refracted, scatterColor, refractionSoftness);
  let reflectedDirection = reflect(-V, N);
  var reflected = skyColor(reflectedDirection, time, L);
  if (REFERENCE_MODE) {
    let blurA = skyColor(normalize(reflectedDirection + vec3<f32>(0.012, 0.006, -0.009)), time, L);
    let blurB = skyColor(normalize(reflectedDirection - vec3<f32>(0.010, 0.004, -0.012)), time, L);
    reflected = reflected * 0.64 + blurA * 0.18 + blurB * 0.18;
  }
  // Sub-resolution slope scatters the mirror direction into a cone. Broadening
  // the glitter lobe alone barely shows, because away from the sun's reflection
  // the far surface is dominated by this sky term -- and a single tap makes it
  // a perfect mirror no matter how rough the water statistically is. Cost is
  // only paid where the control is engaged; at 0 the whole branch is skipped.
  let reflectionSpread = sqrt(recoveredVariance) * 1.9;
  if (reflectionSpread > 0.002) {
    let spreadT = normalize(cross(reflectedDirection, vec3<f32>(0.0, 1.0, 0.0)) + vec3<f32>(1e-4, 0.0, 1e-4));
    let spreadB = cross(reflectedDirection, spreadT);
    let tapA = skyColor(normalize(reflectedDirection + spreadT * reflectionSpread), time, L);
    let tapB = skyColor(normalize(reflectedDirection - spreadT * 0.55 * reflectionSpread + spreadB * 0.84 * reflectionSpread), time, L);
    let tapC = skyColor(normalize(reflectedDirection - spreadT * 0.55 * reflectionSpread - spreadB * 0.84 * reflectionSpread), time, L);
    reflected = reflected * 0.40 + (tapA + tapB + tapC) * 0.20;
  }
  // Preserve environment contrast.  Tinting the reflection toward the water
  // body colour was the main source of the previous milky/plastic response.
  reflected *= mix(vec3<f32>(0.70, 0.77, 0.80), vec3<f32>(0.76, 0.81, 0.83), surfaceRoughness);
  let playerDistance = length(p - uniforms.player.xy);
  let nearSwimmer = 1.0 - smoothstep(0.85, 3.4, playerDistance);
  var reflectionWeight = select(fresnel, fresnel * 0.07, underwater);
  let shoreShallows = uniforms.environment.x * (1.0 - smoothstep(0.12, 1.02, depth));
  // At the waterline the captured sand is still the dominant optical path.
  // Suppress the old cyan body-colour halo and retain a thin, green-blue
  // transmission tint instead of treating centimetres of water like ocean.
  let shallowTransmission = capturedLinear * vec3<f32>(0.66, 0.62, 0.52)
    + vec3<f32>(0.004, 0.013, 0.011) * smoothstep(0.10, 0.90, depth);
  refracted = mix(refracted, shallowTransmission, shoreShallows * 0.76);
  reflectionWeight *= 1.0 - shoreShallows * 0.56;
  reflectionWeight *= mix(1.0, 0.38, nearSwimmer * uniforms.interaction.y);
  var color = mix(refracted, reflected, clamp(reflectionWeight, 0.0, 0.92));
  // A height-based colour wash made crests look like translucent resin.  The
  // spectral normal, Fresnel response and actual foam now carry that contrast.
  let velocityDirection = normalize(uniforms.player.zw + vec2<f32>(0.0001, 0.0));
  let toPlayer = p - uniforms.player.xy;
  let behind = smoothstep(-0.2, 2.8, dot(toPlayer, -velocityDirection));
  let wakeRibbon = exp(-pow(abs(dot(toPlayer, vec2<f32>(-velocityDirection.y, velocityDirection.x))) / 0.64, 2.0)) * (1.0 - smoothstep(0.8, 6.5, playerDistance)) * behind;
  let wake = wakeRibbon * smoothstep(0.5, 5.5, uniforms.interaction.x) * uniforms.interaction.y;
  let crestHeight = smoothstep(0.27, 0.72, input.waveHeight);
  let shortCrossDerivative = short0.a * ${SPECTRAL_CASCADES[2].choppiness.toFixed(2)};
  let shortHorizontalDerivative = short1.ba * ${SPECTRAL_CASCADES[2].choppiness.toFixed(2)};
  let shortJacobian = (1.0 + shortHorizontalDerivative.x) * (1.0 + shortHorizontalDerivative.y) - shortCrossDerivative * shortCrossDerivative;
  let surfaceCompression = input.compression + max(0.0, 1.0 - shortJacobian) * shortDistanceFade * 0.62;
  let crestPinch = smoothstep(0.16, 0.34, surfaceCompression);
  let crestVariation = valueNoise(p * 0.37 + vec2<f32>(time * 0.021, -time * 0.016)) * 0.55
    + valueNoise(p * 1.41 + vec2<f32>(-time * 0.043, time * 0.032)) * 0.30
    + valueNoise(p * 3.7 + vec2<f32>(time * 0.081, -time * 0.066)) * 0.15;
  let crestBreakup = smoothstep(0.60, 0.80, crestVariation);
  let crestDistanceFade = 1.0 - smoothstep(95.0 * detailRange, 188.0 * detailRange, distance(uniforms.cameraTime.xyz, input.world));
  let breakerBreakup = smoothstep(0.43, 0.62, crestVariation);
  let breakerFoam = smoothstep(0.24, 0.72, input.breakerLip) * breakerBreakup * crestDistanceFade;
  let whitecap = max(crestHeight * pow(crestPinch, 4.0) * crestBreakup, breakerFoam * 0.78) * crestDistanceFade;
  let persistentBreakup = 0.48
    + valueNoise(p * 0.72 + vec2<f32>(time * 0.012, -time * 0.009)) * 0.34
    + valueNoise(p * 2.45 + vec2<f32>(-time * 0.024, time * 0.018)) * 0.18;
  let foamBreakup = smoothstep(0.46, 0.78, persistentBreakup);
  let persistentFoam = state.a * 0.58 * foamBreakup * (1.0 - uniforms.environment.x);
  // The coastal wash is selected by the conservative nearshore state rather
  // than by a decorative noise strip. Momentum makes active run-up brighter;
  // the depth window lets it naturally retreat with the simulated waterline.
  let nearshoreSpeed = length(state.gb) / max(depth, 0.08);
  let swashDepth = smoothstep(0.035, 0.11, depth) * (1.0 - smoothstep(0.24, 0.62, depth));
  let activeSwash = smoothstep(0.018, 0.24, state.a) * clamp(0.46 + nearshoreSpeed * 0.12, 0.46, 0.88);
  let shoreStateFoam = uniforms.environment.x * activeSwash * swashDepth * foamBreakup * 0.62;
  var foam = max(max(max(persistentFoam, shoreStateFoam), wake * 0.16), whitecap);
  let visibleFoam = mix(smoothstep(0.16, 0.66, foam), smoothstep(0.055, 0.40, foam), uniforms.environment.x);
  foam = select(visibleFoam, 0.0, underwater);
  let foamCoverage = max(clamp(foam * 0.21, 0.0, 0.145), breakerFoam * 0.22);
  color = mix(color, vec3<f32>(0.80, 0.88, 0.84), clamp(foamCoverage, 0.0, 0.22));
  let sunGlitter = oceanSunGlitter(N, V, L, recoveredVariance);
  color += vec3<f32>(1.0, 0.91, 0.70) * min(sunGlitter * 0.070, 0.34) * select(1.0, 0.08, underwater);
  if (underwater) {
    let viewDepth = min(distance(uniforms.cameraTime.xyz, input.world), 22.0);
    color = mix(color, vec3<f32>(0.012, 0.205, 0.190), 0.42 + viewDepth / 22.0 * 0.12);
  }
  // The water pass carries no aerial term of its own: unfogged water reading
  // to the horizon is the authored look, and the island scene depends on it.
  // The 10x open ocean is the exception -- there the surface reaches far enough
  // that it needs the same fade as the terrain to settle into the horizon.
  // dryLand is 0 here: this surface is water, never exposed shore.
  // Underwater is excluded too: that camera is clamped to a ~19 m orbit and
  // already got its murk from the viewDepth blend just above.
  if (!underwater) {
    color = tethysAerialColor(color, input.world, uniforms.cameraTime.xyz, uniforms.environment.w, false, 0.0);
  }
  return vec4<f32>(linearToSrgb(aces(color)), shorelineCoverage);
}
`;

function deterministicRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function wrapAngle(value: number) {
  let angle = value;
  while (angle > Math.PI) angle -= Math.PI * 2;
  while (angle < -Math.PI) angle += Math.PI * 2;
  return angle;
}

function spectrumNormalisationFactor(spread: number) {
  const s2 = spread * spread;
  const s3 = s2 * spread;
  const s4 = s3 * spread;
  return spread < 5
    ? -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * spread + 0.163
    : -4.8e-8 * s4 + 1.07e-5 * s3 - 9.53e-4 * s2 + 5.9e-2 * spread + 0.393;
}

type SpectralCascadeConfig = Omit<(typeof SPECTRAL_CASCADES)[number], "lengthScale"> & { lengthScale: number };

function buildSpectralOceanData(size: number, config: SpectralCascadeConfig) {
  const { lengthScale, cutoffLow, cutoffHigh, amplitudeScale, secondaryScale, seed } = config;
  const gravity = 9.81;
  const depth = 54;
  const windSpeed = 11.5;
  const fetch = 120_000;
  const windAngle = -0.48;
  const peakEnhancement = 3.3;
  const swell = 0.38;
  const deltaK = Math.PI * 2 / lengthScale;
  const alpha = 0.076 * Math.pow(gravity * fetch / (windSpeed * windSpeed), -0.22);
  const peakOmega = 22 * Math.pow(windSpeed * fetch / (gravity * gravity), -0.33);
  const initialK = new Float32Array(size * size * 2);
  const waveData = new Float32Array(size * size * 4);
  const random = deterministicRandom(seed);
  const gaussian = () => {
    const u = Math.max(random(), 1e-7);
    const v = random();
    const radius = Math.sqrt(-2 * Math.log(u));
    const angle = Math.PI * 2 * v;
    return [radius * Math.cos(angle), radius * Math.sin(angle)] as const;
  };
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const nx = x - size / 2;
      const nz = y - size / 2;
      const kx = nx * deltaK;
      const kz = nz * deltaK;
      const kLength = Math.hypot(kx, kz);
      const pixel = y * size + x;
      const waveOffset = pixel * 4;
      if (kLength < cutoffLow || kLength > cutoffHigh) {
        waveData.set([0, 1, 0, 0], waveOffset);
        continue;
      }
      const kh = Math.min(kLength * depth, 20);
      const tanhKh = Math.tanh(kh);
      const omega = Math.sqrt(gravity * kLength * tanhKh);
      const sechSquared = 1 - tanhKh * tanhKh;
      const frequencyDerivative = gravity * (depth * kLength * sechSquared + tanhKh) / Math.max(omega * 2, 1e-5);
      const omegaH = omega * Math.sqrt(depth / gravity);
      const tma = omegaH <= 1 ? 0.5 * omegaH * omegaH : omegaH < 2 ? 1 - 0.5 * (2 - omegaH) * (2 - omegaH) : 1;
      const sigma = omega <= peakOmega ? 0.07 : 0.09;
      const peakDistance = (omega - peakOmega) / Math.max(sigma * peakOmega, 1e-5);
      const peakShape = Math.exp(-0.5 * peakDistance * peakDistance);
      const peakRatio = peakOmega / omega;
      const jonswap = tma * alpha * gravity * gravity / Math.pow(omega, 5)
        * Math.exp(-1.25 * Math.pow(peakRatio, 4)) * Math.pow(peakEnhancement, peakShape);
      const theta = wrapAngle(Math.atan2(kz, kx) - windAngle);
      const omegaRatio = omega / peakOmega;
      const spreadPower = ((omega > peakOmega ? 9.77 * Math.pow(omegaRatio, -2.5) : 6.97 * Math.pow(omegaRatio, 5))
        + 16 * Math.tanh(Math.min(omegaRatio, 20)) * swell * swell) * 0.58;
      const focusedDirection = spectrumNormalisationFactor(spreadPower) * Math.pow(Math.abs(Math.cos(theta * 0.5)), 2 * spreadPower);
      const broadDirection = 2 / Math.PI * Math.pow(Math.max(Math.cos(theta), 0), 2);
      const direction = focusedDirection * 0.68 + broadDirection * 0.32;
      const shortWaveFade = Math.exp(-0.00016 * kLength * kLength);
      let spectralDensity = jonswap * direction * shortWaveFade;
      if (secondaryScale > 0) {
        const swellWindSpeed = 8.4;
        const swellFetch = 310_000;
        const swellPeakOmega = 22 * Math.pow(swellWindSpeed * swellFetch / (gravity * gravity), -0.33);
        const swellAlpha = 0.076 * Math.pow(gravity * swellFetch / (swellWindSpeed * swellWindSpeed), -0.22);
        const swellSigma = omega <= swellPeakOmega ? 0.07 : 0.09;
        const swellPeakDistance = (omega - swellPeakOmega) / Math.max(swellSigma * swellPeakOmega, 1e-5);
        const swellPeakShape = Math.exp(-0.5 * swellPeakDistance * swellPeakDistance);
        const swellPeakRatio = swellPeakOmega / omega;
        const swellSpectrum = tma * swellAlpha * gravity * gravity / Math.pow(omega, 5)
          * Math.exp(-1.25 * Math.pow(swellPeakRatio, 4)) * Math.pow(2.6, swellPeakShape);
        const swellTheta = wrapAngle(Math.atan2(kz, kx) - (windAngle + 0.82));
        const swellRatio = omega / swellPeakOmega;
        const swellSpread = ((omega > swellPeakOmega ? 9.77 * Math.pow(swellRatio, -2.5) : 6.97 * Math.pow(swellRatio, 5)) + 9.0) * 0.72;
        const swellDirection = spectrumNormalisationFactor(swellSpread) * Math.pow(Math.abs(Math.cos(swellTheta * 0.5)), 2 * swellSpread);
        spectralDensity += swellSpectrum * swellDirection * shortWaveFade * secondaryScale;
      }
      const amplitude = Math.sqrt(Math.max(0, 2 * spectralDensity * Math.abs(frequencyDerivative) / kLength * deltaK * deltaK)) * amplitudeScale;
      const noise = gaussian();
      initialK[pixel * 2] = noise[0] * amplitude;
      initialK[pixel * 2 + 1] = noise[1] * amplitude;
      waveData.set([kx, 1 / kLength, kz, omega], waveOffset);
    }
  }
  const initialSpectrum = new Float32Array(size * size * 4);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const pixel = y * size + x;
      const mirror = ((size - y) % size) * size + ((size - x) % size);
      initialSpectrum.set([
        initialK[pixel * 2], initialK[pixel * 2 + 1],
        initialK[mirror * 2], -initialK[mirror * 2 + 1],
      ], pixel * 4);
    }
  }
  const twiddle = new Float32Array(SPECTRAL_LOG_SIZE * size * 4);
  for (let stage = 0; stage < SPECTRAL_LOG_SIZE; stage += 1) {
    const block = size >> (stage + 1);
    for (let output = 0; output < size / 2; output += 1) {
      const first = (2 * block * Math.floor(output / block) + output % block) % size;
      const angle = -2 * Math.PI / size * Math.floor(output / block) * block;
      const cosine = Math.cos(angle);
      const sine = Math.sin(angle);
      const base = (stage * size + output) * 4;
      const opposite = (stage * size + output + size / 2) * 4;
      twiddle.set([cosine, sine, first, first + block], base);
      twiddle.set([-cosine, -sine, first, first + block], opposite);
    }
  }
  return { initialSpectrum, waveData, twiddle };
}

function percentile(values: readonly number[], fraction: number) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

function mean(values: readonly number[]) {
  return values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function normalize(vector: Vec3): Vec3 {
  const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function lookAt(eye: Vec3, target: Vec3): Float32Array {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross([0, 1, 0], z));
  const y = cross(z, x);
  return new Float32Array([
    x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ]);
}

function perspective(fovRadians: number, aspect: number, near: number, far: number): Float32Array {
  const f = 1 / Math.tan(fovRadians / 2);
  return new Float32Array([
    f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, far / (near - far), -1, 0, 0, (near * far) / (near - far), 0,
  ]);
}

function multiply(left: Float32Array, right: Float32Array): Float32Array {
  const output = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      output[column * 4 + row] = left[row] * right[column * 4] + left[4 + row] * right[column * 4 + 1] + left[8 + row] * right[column * 4 + 2] + left[12 + row] * right[column * 4 + 3];
    }
  }
  return output;
}

async function assertShaderModule(device: GPUDevice, label: string, code: string) {
  const shaderModule = device.createShaderModule({ label, code });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  if (errors.length > 0) {
    throw new Error(`${label}: ${errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join(" | ")}`);
  }
  return shaderModule;
}

export class WebGpuWaterEngine {
  private readonly canvas: HTMLCanvasElement;
  private options: WaterLabOptions;
  private adapter: GPUAdapter | null = null;
  private device: GPUDevice | null = null;
  private context: GPUCanvasContext | null = null;
  private format: GPUTextureFormat = "bgra8unorm";
  private depthTexture: GPUTexture | null = null;
  private sceneColorTexture: GPUTexture | null = null;
  private terrainTexture: GPUTexture | null = null;
  private waterTextures: [GPUTexture, GPUTexture] | null = null;
  private breakerEventTextures: [GPUTexture, GPUTexture] | null = null;
  private worldUniformBuffer: GPUBuffer | null = null;
  private impulseParamBuffer: GPUBuffer | null = null;
  private calmParamBuffer: GPUBuffer | null = null;
  private fieldSampler: GPUSampler | null = null;
  private spectrumSampler: GPUSampler | null = null;
  private ship: ShipRenderer | null = null;
  private shipError: string | null = null;
  private terrainComputePipeline: GPUComputePipeline | null = null;
  private simulationPipeline: GPUComputePipeline | null = null;
  private breakerEventPipeline: GPUComputePipeline | null = null;
  private spectrumEvolutionPipeline: GPUComputePipeline | null = null;
  private spectralIfftPipeline: GPUComputePipeline | null = null;
  private skyPipeline: GPURenderPipeline | null = null;
  private terrainPipeline: GPURenderPipeline | null = null;
  private sceneBlitPipeline: GPURenderPipeline | null = null;
  private optimizedWaterPipeline: GPURenderPipeline | null = null;
  private referenceWaterPipeline: GPURenderPipeline | null = null;
  private optimizedBreakerPatchPipeline: GPURenderPipeline | null = null;
  private referenceBreakerPatchPipeline: GPURenderPipeline | null = null;
  private spectralInitialTextures: GPUTexture[] = [];
  private spectralWaveDataTextures: GPUTexture[] = [];
  private spectralTwiddleTexture: GPUTexture | null = null;
  private spectralFields: SpectralFieldPingPong[] = [];
  private spectrumEvolutionBindGroups: GPUBindGroup[] = [];
  private spectralIfftBindGroups: GPUBindGroup[][] = [];
  private spectralIfftParamBuffers: GPUBuffer[] = [];
  private terrainComputeBindGroup: GPUBindGroup | null = null;
  private skyBindGroup: GPUBindGroup | null = null;
  private terrainBindGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private sceneBlitBindGroup: GPUBindGroup | null = null;
  private optimizedWaterSceneBindGroup: GPUBindGroup | null = null;
  private referenceWaterSceneBindGroup: GPUBindGroup | null = null;
  private optimizedBreakerSceneBindGroup: GPUBindGroup | null = null;
  private referenceBreakerSceneBindGroup: GPUBindGroup | null = null;
  private simulationImpulseGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private simulationCalmGroups: [GPUBindGroup, GPUBindGroup] | null = null;
  private breakerEventBindGroups: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]] | null = null;
  private optimizedWaterBindGroups: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]] | null = null;
  private referenceWaterBindGroups: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]] | null = null;
  private optimizedBreakerPatchBindGroups: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]] | null = null;
  private referenceBreakerPatchBindGroups: [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]] | null = null;
  private terrainPrepared = false;
  private activeSimulationIndex = 0;
  private activeBreakerEventIndex = 0;
  private querySet: GPUQuerySet | null = null;
  private queryResolve: GPUBuffer | null = null;
  private queryReadback: GPUBuffer | null = null;
  private queryPending = false;
  private gpuSimulationTimes: number[] = [];
  private gpuRenderTimes: number[] = [];
  private frameTimes: number[] = [];
  private submitTimes: number[] = [];
  private animationFrame = 0;
  private frameIndex = 0;
  private startTime = performance.now();
  private lastFrameTime = 0;
  private elapsedSeconds = 0;
  // Start near the wind-opposed sun azimuth so the physically generated
  // Cox-Munk glitter path is visible in the default review shot.
  private yaw = 0.56;
  private pitch = 0.07;
  private radius = 58;
  private pointer: { id: number; x: number; y: number; moved: boolean } | null = null;
  private resizeObserver: ResizeObserver | null = null;
  private ready = false;
  private disposed = false;
  private adapterLabel = "正在请求 WebGPU 适配器…";
  private error: string | null = null;
  private disturbanceCount = 0;
  private lastWakeAt = -10;

  constructor(canvas: HTMLCanvasElement, options: WaterLabOptions) {
    this.canvas = canvas;
    this.options = { ...options };
    if (options.scene === "shore" && !Number.isFinite(options.cameraYaw)) this.yaw = Math.PI;
    if (options.scene === "shore" && !Number.isFinite(options.cameraPitch)) this.pitch = 0.22;
    if (Number.isFinite(options.cameraYaw)) this.yaw = options.cameraYaw!;
    if (Number.isFinite(options.cameraPitch)) this.pitch = Math.max(-0.24, Math.min(1.08, options.cameraPitch!));
  }

  async init() {
    if (!navigator.gpu) throw new Error("当前浏览器不支持 WebGPU。请使用较新版本的 Chromium 内核浏览器，并确保已启用硬件加速。");
    this.adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    // React dev StrictMode mounts, disposes and remounts the component, so a
    // disposed engine's init can still be in flight while its replacement owns
    // the same canvas. Bail after every await: the stale engine must never
    // configure the shared context, attach observers or start a render loop --
    // racing the live engine for the context is what stalled dev rendering.
    if (this.disposed) return;
    if (!this.adapter) throw new Error("未找到可用的 WebGPU 适配器。");
    const requiredFeatures: GPUFeatureName[] = this.adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
    this.device = await this.adapter.requestDevice({ requiredFeatures });
    if (this.disposed) {
      this.device.destroy();
      this.device = null;
      return;
    }
    this.device.lost.then((info) => { if (!this.disposed) this.fail(`WebGPU 设备已丢失：${info.message || info.reason}`); });
    this.device.addEventListener("uncapturederror", (event) => this.fail(event.error.message));
    const info = this.adapter.info;
    this.adapterLabel = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(" · ") || "WebGPU 适配器";
    this.context = this.canvas.getContext("webgpu");
    if (!this.context) throw new Error("无法创建 WebGPU 画布上下文。");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
    await this.createResources();
    if (this.disposed) return;
    if (this.device.features.has("timestamp-query")) {
      this.querySet = this.device.createQuerySet({ label: "Tethys compute and render timestamps", type: "timestamp", count: 4 });
      this.queryResolve = this.device.createBuffer({ label: "Tethys timestamp resolve", size: 32, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      this.queryReadback = this.device.createBuffer({ label: "Tethys timestamp readback", size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }
    this.installInteraction();
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
    this.resize();
    // The hull is optional scenery: a missing or malformed asset must not take
    // the whole renderer down, so a failure here is surfaced and then skipped.
    try {
      this.ship = await ShipRenderer.create({
        device: this.device,
        format: this.format,
        depthFormat: DEPTH_FORMAT,
        worldUniformBuffer: this.worldUniformBuffer!,
        longField: this.spectralFields[0][0][0].createView(),
        mediumField: this.spectralFields[1][0][0].createView(),
        spectrumSampler: this.spectrumSampler!,
        cascadeScales: [this.options.longCascadeScale, this.options.mediumCascadeScale],
        modelUrl: SHIP_MODEL_URL,
        placement: SHIP_PLACEMENTS[this.options.scene],
      });
    } catch (error) {
      this.shipError = error instanceof Error ? error.message : String(error);
    }
    if (this.disposed) {
      this.ship?.dispose();
      this.ship = null;
      return;
    }
    this.ready = true;
    this.startTime = performance.now();
    this.lastFrameTime = this.startTime;
    this.animationFrame = requestAnimationFrame(this.render);
  }

  private async createResources() {
    const device = this.device;
    if (!device) return;
    const [terrainModule, simulationModule, breakerEventModule, spectrumEvolutionModule, spectralIfftModule, skyModule, terrainRenderModule, sceneBlitModule, waterModule] = await Promise.all([
      assertShaderModule(device, "Tethys terrain field compute", TERRAIN_FIELD_SHADER),
      assertShaderModule(device, "Tethys water simulation compute", WATER_SIMULATION_SHADER),
      assertShaderModule(device, "Tethys persistent breaker event compute", BREAKER_EVENT_SHADER),
      assertShaderModule(device, "Tethys spectral evolution compute", SPECTRUM_EVOLUTION_SHADER),
      assertShaderModule(device, "Tethys spectral inverse FFT compute", SPECTRAL_IFFT_SHADER),
      assertShaderModule(device, "Tethys atmosphere", SKY_SHADER),
      assertShaderModule(device, "Tethys terrain material", TERRAIN_RENDER_SHADER),
      assertShaderModule(device, "Tethys captured scene composite", SCENE_BLIT_SHADER),
      assertShaderModule(device, "Tethys water material", WATER_RENDER_SHADER),
    ]);
    if (this.disposed) return;
    this.worldUniformBuffer = device.createBuffer({ label: "Tethys world uniforms", size: WORLD_UNIFORM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.impulseParamBuffer = device.createBuffer({ label: "Tethys impulse step parameters", size: SIMULATION_PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.calmParamBuffer = device.createBuffer({ label: "Tethys calm step parameters", size: SIMULATION_PARAM_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.fieldSampler = device.createSampler({ label: "Tethys field sampler", addressModeU: "clamp-to-edge", addressModeV: "clamp-to-edge", minFilter: "linear", magFilter: "linear" });
    this.spectrumSampler = device.createSampler({ label: "Tethys periodic spectrum sampler", addressModeU: "repeat", addressModeV: "repeat", minFilter: "linear", magFilter: "linear" });
    this.terrainComputePipeline = await device.createComputePipelineAsync({ label: "Tethys static terrain precompute", layout: "auto", compute: { module: terrainModule, entryPoint: "buildTerrain" } });
    this.simulationPipeline = await device.createComputePipelineAsync({ label: "Tethys nonlinear nearshore residual simulation", layout: "auto", compute: { module: simulationModule, entryPoint: "simulate" } });
    this.breakerEventPipeline = await device.createComputePipelineAsync({ label: "Tethys persistent instability-triggered breakers", layout: "auto", compute: { module: breakerEventModule, entryPoint: "updateBreakerEvents" } });
    this.spectrumEvolutionPipeline = await device.createComputePipelineAsync({ label: "Tethys time-dependent ocean spectrum", layout: "auto", compute: { module: spectrumEvolutionModule, entryPoint: "evolveSpectrum" } });
    this.spectralIfftPipeline = await device.createComputePipelineAsync({ label: "Tethys Stockham spectral inverse FFT", layout: "auto", compute: { module: spectralIfftModule, entryPoint: "inverseFftStage" } });
    this.skyPipeline = await device.createRenderPipelineAsync({
      label: "Tethys sky",
      layout: "auto",
      vertex: { module: skyModule, entryPoint: "skyVertex" },
      fragment: { module: skyModule, entryPoint: "skyFragment", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: "always" },
    });
    this.terrainPipeline = await device.createRenderPipelineAsync({
      label: "Tethys terrain",
      layout: "auto",
      vertex: { module: terrainRenderModule, entryPoint: "terrainVertex" },
      fragment: { module: terrainRenderModule, entryPoint: "terrainFragment", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: "less" },
    });
    this.sceneBlitPipeline = await device.createRenderPipelineAsync({
      label: "Tethys captured scene composite",
      layout: "auto",
      vertex: { module: sceneBlitModule, entryPoint: "blitVertex" },
      fragment: { module: sceneBlitModule, entryPoint: "blitFragment", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: "always" },
    });
    const waterBase: Omit<GPURenderPipelineDescriptor, "label" | "fragment"> = {
      layout: "auto",
      vertex: { module: waterModule, entryPoint: "waterVertex" },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: "less" },
    };
    const waterTarget: GPUColorTargetState = {
      format: this.format,
      blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
      },
    };
    this.optimizedWaterPipeline = await device.createRenderPipelineAsync({ ...waterBase, label: "optimized Tethys water", fragment: { module: waterModule, entryPoint: "waterFragment", constants: { REFERENCE_MODE: 0 }, targets: [waterTarget] } });
    this.referenceWaterPipeline = await device.createRenderPipelineAsync({ ...waterBase, label: "reference Tethys water", fragment: { module: waterModule, entryPoint: "waterFragment", constants: { REFERENCE_MODE: 1 }, targets: [waterTarget] } });
    const breakerPatchBase: Omit<GPURenderPipelineDescriptor, "label" | "fragment"> = {
      ...waterBase,
      vertex: { module: waterModule, entryPoint: "breakerPatchVertex" },
    };
    this.optimizedBreakerPatchPipeline = await device.createRenderPipelineAsync({ ...breakerPatchBase, label: "optimized Tethys breaker patch", fragment: { module: waterModule, entryPoint: "waterFragment", constants: { REFERENCE_MODE: 0 }, targets: [waterTarget] } });
    this.referenceBreakerPatchPipeline = await device.createRenderPipelineAsync({ ...breakerPatchBase, label: "reference Tethys breaker patch", fragment: { module: waterModule, entryPoint: "waterFragment", constants: { REFERENCE_MODE: 1 }, targets: [waterTarget] } });
    // A dispose during the pipeline awaits already destroyed the buffers made
    // above; allocating the field textures now would leak them irrecoverably.
    if (this.disposed) return;
    this.allocateFields();
  }

  private allocateFields() {
    const device = this.device;
    if (!device || !this.worldUniformBuffer || !this.impulseParamBuffer || !this.calmParamBuffer || !this.fieldSampler || !this.spectrumSampler || !this.terrainComputePipeline || !this.simulationPipeline || !this.breakerEventPipeline || !this.spectrumEvolutionPipeline || !this.spectralIfftPipeline || !this.skyPipeline || !this.terrainPipeline || !this.optimizedWaterPipeline || !this.referenceWaterPipeline || !this.optimizedBreakerPatchPipeline || !this.referenceBreakerPatchPipeline) return;
    this.terrainTexture?.destroy();
    this.waterTextures?.forEach((texture) => texture.destroy());
    this.breakerEventTextures?.forEach((texture) => texture.destroy());
    this.spectralInitialTextures.forEach((texture) => texture.destroy());
    this.spectralWaveDataTextures.forEach((texture) => texture.destroy());
    this.spectralTwiddleTexture?.destroy();
    this.spectralFields.flat(2).forEach((texture) => texture.destroy());
    this.spectralIfftParamBuffers.forEach((buffer) => buffer.destroy());
    this.spectralInitialTextures = [];
    this.spectralWaveDataTextures = [];
    this.spectralFields = [];
    this.spectrumEvolutionBindGroups = [];
    this.spectralIfftParamBuffers = [];
    this.spectralIfftBindGroups = [];
    const terrainSize = TERRAIN_FIELD_RESOLUTION + 1;
    this.terrainTexture = device.createTexture({ label: "Tethys terrain height-normal-material field", size: [terrainSize, terrainSize], format: "rgba16float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING });
    const simulationSize = Math.max(64, Math.min(512, Math.floor(this.options.simulationResolution)));
    this.options.simulationResolution = simulationSize;
    this.waterTextures = [0, 1].map((index) => device.createTexture({ label: `Tethys water state ${index}`, size: [simulationSize, simulationSize], format: "rgba16float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })) as [GPUTexture, GPUTexture];
    this.breakerEventTextures = [0, 1].map((index) => device.createTexture({ label: `Tethys breaker event history ${index}`, size: [BREAKER_EVENT_RESOLUTION, 1], format: "rgba16float", usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST })) as [GPUTexture, GPUTexture];
    this.spectralTwiddleTexture = device.createTexture({ label: "Tethys Stockham FFT twiddle table", size: [SPECTRAL_RESOLUTION, SPECTRAL_LOG_SIZE], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const spectrumUpload = { bytesPerRow: SPECTRAL_RESOLUTION * 16, rowsPerImage: SPECTRAL_RESOLUTION };
    for (let pass = 0; pass < SPECTRAL_LOG_SIZE * 2; pass += 1) {
      const params = device.createBuffer({ label: `Tethys inverse FFT pass ${pass} parameters`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(params, 0, new Uint32Array([pass < SPECTRAL_LOG_SIZE ? 0 : 1, pass % SPECTRAL_LOG_SIZE, SPECTRAL_RESOLUTION, pass === SPECTRAL_LOG_SIZE * 2 - 1 ? 1 : 0]));
      this.spectralIfftParamBuffers.push(params);
    }
    for (let cascadeIndex = 0; cascadeIndex < SPECTRAL_CASCADES.length; cascadeIndex += 1) {
      const config = { ...SPECTRAL_CASCADES[cascadeIndex], lengthScale: this.cascadeScale(cascadeIndex) };
      const initialTexture = device.createTexture({ label: `Tethys cascade ${cascadeIndex} initial spectrum`, size: [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      const waveDataTexture = device.createTexture({ label: `Tethys cascade ${cascadeIndex} wave vectors and dispersion`, size: [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      const fields = [0, 1].map((ping) => [0, 1].map((field) => device.createTexture({
        label: `Tethys cascade ${cascadeIndex} field ping ${ping} channel ${field}`,
        size: [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      })) as [GPUTexture, GPUTexture]) as SpectralFieldPingPong;
      const spectralData = buildSpectralOceanData(SPECTRAL_RESOLUTION, config);
      device.queue.writeTexture({ texture: initialTexture }, spectralData.initialSpectrum, spectrumUpload, [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION]);
      device.queue.writeTexture({ texture: waveDataTexture }, spectralData.waveData, spectrumUpload, [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION]);
      if (cascadeIndex === 0) device.queue.writeTexture({ texture: this.spectralTwiddleTexture }, spectralData.twiddle, { bytesPerRow: SPECTRAL_RESOLUTION * 16, rowsPerImage: SPECTRAL_LOG_SIZE }, [SPECTRAL_RESOLUTION, SPECTRAL_LOG_SIZE]);
      this.spectralInitialTextures.push(initialTexture);
      this.spectralWaveDataTextures.push(waveDataTexture);
      this.spectralFields.push(fields);
      this.spectrumEvolutionBindGroups.push(device.createBindGroup({ label: `Tethys cascade ${cascadeIndex} spectrum evolution`, layout: this.spectrumEvolutionPipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.worldUniformBuffer } },
        { binding: 1, resource: initialTexture.createView() },
        { binding: 2, resource: waveDataTexture.createView() },
        { binding: 3, resource: fields[0][0].createView() },
        { binding: 4, resource: fields[0][1].createView() },
      ] }));
      this.spectralIfftBindGroups.push(this.spectralIfftParamBuffers.map((params, pass) => {
        const source = pass % 2;
        const destination = 1 - source;
        return device.createBindGroup({ label: `Tethys cascade ${cascadeIndex} inverse FFT pass ${pass}`, layout: this.spectralIfftPipeline!.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: params } },
          { binding: 1, resource: this.spectralTwiddleTexture!.createView() },
          { binding: 2, resource: fields[source][0].createView() },
          { binding: 3, resource: fields[source][1].createView() },
          { binding: 4, resource: fields[destination][0].createView() },
          { binding: 5, resource: fields[destination][1].createView() },
        ] });
      }));
    }
    this.terrainComputeBindGroup = device.createBindGroup({ label: "Tethys terrain compute resources", layout: this.terrainComputePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.worldUniformBuffer } }, { binding: 1, resource: this.terrainTexture.createView() }] });
    this.skyBindGroup = device.createBindGroup({ label: "Tethys sky resources", layout: this.skyPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.worldUniformBuffer } }] });
    const terrainGroup = (simulationIndex: number) => device.createBindGroup({ label: `Tethys terrain resources, water ${simulationIndex}`, layout: this.terrainPipeline!.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.worldUniformBuffer! } },
      { binding: 1, resource: this.terrainTexture!.createView() },
      { binding: 2, resource: this.fieldSampler! },
      { binding: 3, resource: this.spectralFields[1][0][0].createView() },
      { binding: 4, resource: this.spectralFields[1][0][1].createView() },
      { binding: 5, resource: this.spectralFields[2][0][0].createView() },
      { binding: 6, resource: this.spectralFields[2][0][1].createView() },
      { binding: 7, resource: this.spectrumSampler! },
      { binding: 8, resource: this.waterTextures![simulationIndex].createView() },
    ] });
    this.terrainBindGroups = [terrainGroup(0), terrainGroup(1)];
    const simulationGroup = (source: number, destination: number, params: GPUBuffer) => device.createBindGroup({ label: `Tethys simulation ${source} to ${destination}`, layout: this.simulationPipeline!.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.worldUniformBuffer! } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: this.waterTextures![source].createView() },
      { binding: 3, resource: this.waterTextures![destination].createView() },
      { binding: 4, resource: this.terrainTexture!.createView() },
      { binding: 5, resource: this.spectralFields[0][0][0].createView() },
      { binding: 6, resource: this.spectralFields[0][0][1].createView() },
      { binding: 7, resource: this.spectralFields[1][0][0].createView() },
      { binding: 8, resource: this.spectralFields[1][0][1].createView() },
      { binding: 9, resource: this.spectrumSampler! },
    ] });
    this.simulationImpulseGroups = [simulationGroup(0, 1, this.impulseParamBuffer), simulationGroup(1, 0, this.impulseParamBuffer)];
    this.simulationCalmGroups = [simulationGroup(0, 1, this.calmParamBuffer), simulationGroup(1, 0, this.calmParamBuffer)];
    const breakerEventGroup = (source: number, destination: number, simulationIndex: number) => device.createBindGroup({ label: `Tethys breaker events ${source} to ${destination}, water ${simulationIndex}`, layout: this.breakerEventPipeline!.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.worldUniformBuffer! } },
      { binding: 1, resource: this.breakerEventTextures![source].createView() },
      { binding: 2, resource: this.breakerEventTextures![destination].createView() },
      { binding: 3, resource: this.terrainTexture!.createView() },
      { binding: 4, resource: this.waterTextures![simulationIndex].createView() },
      { binding: 5, resource: this.fieldSampler! },
      { binding: 6, resource: this.spectralFields[0][0][0].createView() },
      { binding: 7, resource: this.spectralFields[0][0][1].createView() },
      { binding: 8, resource: this.spectralFields[1][0][0].createView() },
      { binding: 9, resource: this.spectralFields[1][0][1].createView() },
      { binding: 10, resource: this.spectrumSampler! },
    ] });
    this.breakerEventBindGroups = [
      [breakerEventGroup(0, 1, 0), breakerEventGroup(0, 1, 1)],
      [breakerEventGroup(1, 0, 0), breakerEventGroup(1, 0, 1)],
    ];
    const waterGroup = (pipeline: GPURenderPipeline, simulationIndex: number, breakerEventIndex: number) => device.createBindGroup({ label: `Tethys water render state ${simulationIndex}, breakers ${breakerEventIndex}`, layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.worldUniformBuffer! } },
      { binding: 1, resource: this.terrainTexture!.createView() },
      { binding: 2, resource: this.waterTextures![simulationIndex].createView() },
      { binding: 3, resource: this.fieldSampler! },
      { binding: 4, resource: this.spectralFields[0][0][0].createView() },
      { binding: 5, resource: this.spectralFields[0][0][1].createView() },
      { binding: 6, resource: this.spectralFields[1][0][0].createView() },
      { binding: 7, resource: this.spectralFields[1][0][1].createView() },
      { binding: 8, resource: this.spectralFields[2][0][0].createView() },
      { binding: 9, resource: this.spectralFields[2][0][1].createView() },
      { binding: 10, resource: this.spectrumSampler! },
      { binding: 11, resource: this.breakerEventTextures![breakerEventIndex].createView() },
    ] });
    const waterGroupMatrix = (pipeline: GPURenderPipeline): [[GPUBindGroup, GPUBindGroup], [GPUBindGroup, GPUBindGroup]] => [
      [waterGroup(pipeline, 0, 0), waterGroup(pipeline, 0, 1)],
      [waterGroup(pipeline, 1, 0), waterGroup(pipeline, 1, 1)],
    ];
    this.optimizedWaterBindGroups = waterGroupMatrix(this.optimizedWaterPipeline);
    this.referenceWaterBindGroups = waterGroupMatrix(this.referenceWaterPipeline);
    this.optimizedBreakerPatchBindGroups = waterGroupMatrix(this.optimizedBreakerPatchPipeline);
    this.referenceBreakerPatchBindGroups = waterGroupMatrix(this.referenceBreakerPatchPipeline);
    this.terrainPrepared = false;
    this.activeSimulationIndex = 0;
    this.activeBreakerEventIndex = 0;
    // The cascade textures were just destroyed and recreated; the hull samples
    // them for buoyancy and would otherwise submit views of the destroyed ones
    // on every frame after a scene switch or simulation-resolution change.
    if (this.ship && this.spectrumSampler) {
      this.ship.bindSpectralFields(
        this.spectralFields[0][0][0].createView(),
        this.spectralFields[1][0][0].createView(),
        this.spectrumSampler,
      );
    }
  }

  setMode(mode: WaterRenderMode) { this.options.mode = mode; }
  setView(view: WaterView) { this.options.view = view; }
  setScene(scene: WaterScene) {
    if (scene === this.options.scene) return;
    this.options.scene = scene;
    this.ship?.setPlacement(SHIP_PLACEMENTS[scene]);
    this.terrainPrepared = false;
    this.allocateFields();
  }
  setMeshResolution(value: number) { this.options.meshResolution = Math.max(96, Math.min(320, Math.floor(value))); }
  setSimulationResolution(value: number) {
    const next = Math.max(64, Math.min(512, Math.floor(value)));
    if (next === this.options.simulationResolution) return;
    this.options.simulationResolution = next;
    this.allocateFields();
  }
  setRenderScale(value: number) { this.options.renderScale = Math.max(0.5, Math.min(1.25, value)); this.resize(true); }
  setWaveScale(value: number) {
    this.options.waveScale = Math.max(MIN_WAVE_SCALE, Math.min(MAX_WAVE_SCALE, Number.isFinite(value) ? value : 1));
  }
  setDistantRoughness(value: number) {
    this.options.distantRoughness = Math.max(0, Math.min(MAX_DISTANT_ROUGHNESS, Number.isFinite(value) ? value : 0));
  }
  setDetailRange(value: number) {
    this.options.detailRange = Math.max(MIN_DETAIL_RANGE, Math.min(MAX_DETAIL_RANGE, Number.isFinite(value) ? value : 1));
  }
  private cascadeScale(index: number) {
    if (index === 0) return this.options.longCascadeScale;
    if (index === 1) return this.options.mediumCascadeScale;
    return SPECTRAL_CASCADES[index].lengthScale;
  }
  // Re-derives one cascade's initial spectrum for its new tile size and
  // overwrites the textures in place; the evolution/IFFT chain and all
  // pipelines are untouched because shaders read the size from atmosphere.zw.
  private uploadCascadeSpectrum(cascadeIndex: number) {
    const device = this.device;
    if (!device || this.spectralInitialTextures.length <= cascadeIndex) return;
    const config = { ...SPECTRAL_CASCADES[cascadeIndex], lengthScale: this.cascadeScale(cascadeIndex) };
    const data = buildSpectralOceanData(SPECTRAL_RESOLUTION, config);
    const upload = { bytesPerRow: SPECTRAL_RESOLUTION * 16, rowsPerImage: SPECTRAL_RESOLUTION };
    device.queue.writeTexture({ texture: this.spectralInitialTextures[cascadeIndex] }, data.initialSpectrum, upload, [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION]);
    device.queue.writeTexture({ texture: this.spectralWaveDataTextures[cascadeIndex] }, data.waveData, upload, [SPECTRAL_RESOLUTION, SPECTRAL_RESOLUTION]);
  }
  setLongCascadeScale(value: number) {
    const next = Math.max(LONG_SCALE_RANGE[0], Math.min(LONG_SCALE_RANGE[1], Number.isFinite(value) ? value : 240));
    if (next === this.options.longCascadeScale) return;
    this.options.longCascadeScale = next;
    this.uploadCascadeSpectrum(0);
    this.ship?.setCascadeScales(this.options.longCascadeScale, this.options.mediumCascadeScale);
  }
  setMediumCascadeScale(value: number) {
    const next = Math.max(MEDIUM_SCALE_RANGE[0], Math.min(MEDIUM_SCALE_RANGE[1], Number.isFinite(value) ? value : 64));
    if (next === this.options.mediumCascadeScale) return;
    this.options.mediumCascadeScale = next;
    this.uploadCascadeSpectrum(1);
    this.ship?.setCascadeScales(this.options.longCascadeScale, this.options.mediumCascadeScale);
  }
  setSwellSmoothing(value: number) {
    this.options.swellSmoothing = Math.max(0, Math.min(MAX_SWELL_SMOOTHING, Number.isFinite(value) ? value : 1));
  }
  setFogReach(value: number) {
    this.options.fogReach = Math.max(0, Math.min(MAX_FOG_REACH, Number.isFinite(value) ? value : 0));
  }

  resetMetrics() {
    this.frameTimes.length = 0;
    this.submitTimes.length = 0;
    this.gpuSimulationTimes.length = 0;
    this.gpuRenderTimes.length = 0;
  }

  private installInteraction() {
    this.canvas.addEventListener("pointerdown", this.onPointerDown);
    this.canvas.addEventListener("pointermove", this.onPointerMove);
    this.canvas.addEventListener("pointerup", this.onPointerUp);
    this.canvas.addEventListener("pointercancel", this.onPointerUp);
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false });
  }

  private removeInteraction() {
    this.canvas.removeEventListener("pointerdown", this.onPointerDown);
    this.canvas.removeEventListener("pointermove", this.onPointerMove);
    this.canvas.removeEventListener("pointerup", this.onPointerUp);
    this.canvas.removeEventListener("pointercancel", this.onPointerUp);
    this.canvas.removeEventListener("wheel", this.onWheel);
  }

  private onPointerDown = (event: PointerEvent) => {
    this.pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, moved: false };
    this.canvas.setPointerCapture(event.pointerId);
  };

  private onPointerMove = (event: PointerEvent) => {
    if (!this.pointer || event.pointerId !== this.pointer.id) return;
    const dx = event.clientX - this.pointer.x;
    const dy = event.clientY - this.pointer.y;
    if (Math.abs(dx) + Math.abs(dy) > 2) this.pointer.moved = true;
    this.pointer.x = event.clientX;
    this.pointer.y = event.clientY;
    this.yaw -= dx * 0.005;
    this.pitch = Math.max(-0.24, Math.min(1.08, this.pitch + dy * 0.004));
  };

  private onPointerUp = (event: PointerEvent) => {
    if (this.pointer?.id === event.pointerId) {
      this.pointer = null;
    }
  };

  private onWheel = (event: WheelEvent) => {
    event.preventDefault();
    const ceiling = OPEN_WATER_MAX_ORBIT;
    this.radius = Math.max(6, Math.min(ceiling, this.radius * Math.exp(event.deltaY * 0.001)));
  };

  private resize(force = false) {
    const device = this.device;
    if (!device) return;
    const maximumDpr = this.options.benchmark ? 1 : 1.5;
    const dpr = Math.min(window.devicePixelRatio || 1, maximumDpr) * this.options.renderScale;
    const width = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    if (!force && this.canvas.width === width && this.canvas.height === height) return;
    this.canvas.width = width;
    this.canvas.height = height;
    this.depthTexture?.destroy();
    this.sceneColorTexture?.destroy();
    this.depthTexture = device.createTexture({ label: "Tethys captured scene depth", size: [width, height], format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.sceneColorTexture = device.createTexture({ label: "Tethys captured scene color", size: [width, height], format: this.format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    if (!this.fieldSampler || !this.sceneBlitPipeline || !this.optimizedWaterPipeline || !this.referenceWaterPipeline || !this.optimizedBreakerPatchPipeline || !this.referenceBreakerPatchPipeline) return;
    this.sceneBlitBindGroup = device.createBindGroup({ label: "Tethys captured scene blit resources", layout: this.sceneBlitPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: this.sceneColorTexture.createView() },
      { binding: 1, resource: this.fieldSampler },
    ] });
    const sceneGroup = (pipeline: GPURenderPipeline, label: string) => device.createBindGroup({ label, layout: pipeline.getBindGroupLayout(1), entries: [
      { binding: 0, resource: this.sceneColorTexture!.createView() },
      { binding: 1, resource: this.depthTexture!.createView() },
      { binding: 2, resource: this.fieldSampler! },
    ] });
    this.optimizedWaterSceneBindGroup = sceneGroup(this.optimizedWaterPipeline, "Tethys optimized water captured scene");
    this.referenceWaterSceneBindGroup = sceneGroup(this.referenceWaterPipeline, "Tethys reference water captured scene");
    this.optimizedBreakerSceneBindGroup = sceneGroup(this.optimizedBreakerPatchPipeline, "Tethys optimized breaker captured scene");
    this.referenceBreakerSceneBindGroup = sceneGroup(this.referenceBreakerPatchPipeline, "Tethys reference breaker captured scene");
  }

  // How much further than the authored 145 m this scene can see. The island
  // scene is a close-range shoreline study and stays at 1.
  private worldScale() {
    return OPEN_WATER_VIEW_SCALE;
  }

  private frameState(timestamp: number) {
    const liveSeconds = (timestamp - this.startTime) / 1000;
    this.elapsedSeconds = this.options.fixedTime ?? liveSeconds;
    const underwater = this.options.view === "underwater";
    const shoreScene = this.options.scene === "shore";
    const target: Vec3 = [0, underwater ? -5.0 : TETHYS_WATER_LEVEL, shoreScene ? 14 : (underwater ? -18 : -22)];
    const orbitRadius = underwater ? Math.min(this.radius, 19) : this.radius;
    const horizontal = Math.cos(this.pitch) * orbitRadius;
    const verticalOrbit = Math.sin(this.pitch) * orbitRadius * (underwater ? 0.22 : 1.0);
    const defaultY = underwater ? -2.0 : 5.2;
    const eye: Vec3 = [target[0] + Math.sin(this.yaw) * horizontal, defaultY + verticalOrbit, target[2] + Math.cos(this.yaw) * horizontal];
    if (underwater) eye[1] = Math.min(TETHYS_WATER_LEVEL - 0.9, eye[1]);
    const forward = normalize([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    const right = normalize(cross(forward, [0, 1, 0]));
    const up = normalize(cross(right, forward));
    const farPlane = OPEN_WATER_FAR_PLANE;
    const projection = perspective(52 * Math.PI / 180, this.canvas.width / this.canvas.height, 0.12, farPlane);
    const view = lookAt(eye, target);
    const playerAngle = this.elapsedSeconds * 0.22;
    const playerPosition: [number, number] = [Math.sin(playerAngle) * 7.5, -18 + Math.cos(playerAngle) * 5.2];
    const playerVelocity: [number, number] = [Math.cos(playerAngle) * 1.65, -Math.sin(playerAngle) * 1.14];
    return { eye, forward, right, up, viewProjection: multiply(projection, view), playerPosition, playerVelocity, underwater };
  }

  private writeUniforms(timestamp: number) {
    const device = this.device;
    const buffer = this.worldUniformBuffer;
    if (!device || !buffer) throw new Error("特提斯 uniform 缓冲不可用。");
    const frame = this.frameState(timestamp);
    const values = new Float32Array(WORLD_UNIFORM_BYTES / 4);
    values.set(frame.viewProjection, 0);
    values.set([...frame.eye, this.elapsedSeconds], 16);
    const tanHalfFov = Math.tan(52 * Math.PI / 360);
    values.set([...frame.right, tanHalfFov * (this.canvas.width / this.canvas.height)], 20);
    values.set([...frame.up, tanHalfFov], 24);
    values.set([...frame.forward, 0], 28);
    values.set([...normalize([-0.52, 0.30, -0.80]), TETHYS_WATER_LEVEL], 32);
    values.set([TERRAIN_EXTENT, this.options.meshResolution, this.options.simulationResolution, frame.underwater ? 1 : 0], 36);
    values.set([0, -12, TETHYS_WATER_FIELD_SIZE, 1 / this.options.simulationResolution], 40);
    values.set([...frame.playerPosition, ...frame.playerVelocity], 44);
    values.set([Math.hypot(...frame.playerVelocity), 1, this.canvas.width, this.canvas.height], 48);
    const validationMesh = this.options.scene === "shore" ? 512 : this.options.meshResolution;
    values.set([this.options.scene === "shore" ? 1 : 0, validationMesh, validationMesh, this.worldScale()], 52);
    const waveScale = this.options.waveScale;
    values.set([waveScale, waveScale * waveScale, this.options.distantRoughness, this.options.detailRange], 56);
    values.set([this.options.fogReach, this.options.swellSmoothing, this.options.longCascadeScale, this.options.mediumCascadeScale], 60);
    device.queue.writeBuffer(buffer, 0, values);
    return frame;
  }

  private writeSimulationParams(frame: ReturnType<WebGpuWaterEngine["frameState"]>) {
    const device = this.device;
    if (!device || !this.impulseParamBuffer || !this.calmParamBuffer) return;
    const wakeDue = this.elapsedSeconds - this.lastWakeAt >= 0.10;
    const impulseStrength = wakeDue ? -0.012 : 0;
    if (impulseStrength !== 0) {
      this.disturbanceCount += 1;
      this.lastWakeAt = this.elapsedSeconds;
    }
    const impulseUvX = (frame.playerPosition[0] - 0) / TETHYS_WATER_FIELD_SIZE + 0.5;
    const impulseUvY = (frame.playerPosition[1] + 12) / TETHYS_WATER_FIELD_SIZE + 0.5;
    const step = this.options.mode === "reference" ? 1 / 120 : 1 / 60;
    const impulse = new Float32Array([impulseUvX, impulseUvY, impulseStrength, 0.54 / TETHYS_WATER_FIELD_SIZE, step, 0.72, 0, 0]);
    const calm = new Float32Array([impulseUvX, impulseUvY, 0, 0.54 / TETHYS_WATER_FIELD_SIZE, step, 0.72, 0, 0]);
    device.queue.writeBuffer(this.impulseParamBuffer, 0, impulse);
    device.queue.writeBuffer(this.calmParamBuffer, 0, calm);
  }

  private render = (timestamp: number) => {
    if (this.disposed || !this.device || !this.context || !this.depthTexture || !this.sceneColorTexture || !this.terrainTexture || !this.terrainComputePipeline || !this.simulationPipeline || !this.breakerEventPipeline || !this.spectrumEvolutionPipeline || !this.spectralIfftPipeline || !this.skyPipeline || !this.terrainPipeline || !this.sceneBlitPipeline || !this.optimizedWaterPipeline || !this.referenceWaterPipeline || !this.optimizedBreakerPatchPipeline || !this.referenceBreakerPatchPipeline || !this.terrainComputeBindGroup || this.spectrumEvolutionBindGroups.length !== SPECTRAL_CASCADES.length || this.spectralIfftBindGroups.length !== SPECTRAL_CASCADES.length || !this.skyBindGroup || !this.terrainBindGroups || !this.sceneBlitBindGroup || !this.optimizedWaterSceneBindGroup || !this.referenceWaterSceneBindGroup || !this.optimizedBreakerSceneBindGroup || !this.referenceBreakerSceneBindGroup || !this.simulationImpulseGroups || !this.simulationCalmGroups || !this.breakerEventBindGroups || !this.optimizedWaterBindGroups || !this.referenceWaterBindGroups || !this.optimizedBreakerPatchBindGroups || !this.referenceBreakerPatchBindGroups) return;
    const frameDelta = timestamp - this.lastFrameTime;
    this.lastFrameTime = timestamp;
    if (frameDelta > 0 && frameDelta < 1000) {
      this.frameTimes.push(frameDelta);
      if (this.frameTimes.length > FRAME_HISTORY) this.frameTimes.shift();
    }
    const frame = this.writeUniforms(timestamp);
    this.writeSimulationParams(frame);
    const submitStartedAt = performance.now();
    const encoder = this.device.createCommandEncoder({ label: "Tethys WebGPU frame" });
    if (!this.terrainPrepared) {
      const pass = encoder.beginComputePass({ label: "Tethys static terrain field" });
      pass.setPipeline(this.terrainComputePipeline);
      pass.setBindGroup(0, this.terrainComputeBindGroup);
      pass.dispatchWorkgroups(Math.ceil((TERRAIN_FIELD_RESOLUTION + 1) / 16), Math.ceil((TERRAIN_FIELD_RESOLUTION + 1) / 16));
      pass.end();
      this.terrainPrepared = true;
    }
    const measureGpu = Boolean(this.querySet && this.queryResolve && this.queryReadback && !this.queryPending && this.frameIndex % 8 === 0);
    const computePass = encoder.beginComputePass({
      label: "Tethys spectral ocean and local wake simulation",
      timestampWrites: measureGpu && this.querySet ? { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
    });
    const spectralGroups = Math.ceil(SPECTRAL_RESOLUTION / 8);
    for (let cascadeIndex = 0; cascadeIndex < SPECTRAL_CASCADES.length; cascadeIndex += 1) {
      computePass.setPipeline(this.spectrumEvolutionPipeline);
      computePass.setBindGroup(0, this.spectrumEvolutionBindGroups[cascadeIndex]);
      computePass.dispatchWorkgroups(spectralGroups, spectralGroups);
      computePass.setPipeline(this.spectralIfftPipeline);
      for (const bindGroup of this.spectralIfftBindGroups[cascadeIndex]) {
        computePass.setBindGroup(0, bindGroup);
        computePass.dispatchWorkgroups(spectralGroups, spectralGroups);
      }
    }
    computePass.setPipeline(this.simulationPipeline);
    computePass.setBindGroup(0, this.simulationImpulseGroups[this.activeSimulationIndex]);
    const groups = Math.ceil(this.options.simulationResolution / 16);
    computePass.dispatchWorkgroups(groups, groups);
    this.activeSimulationIndex = 1 - this.activeSimulationIndex;
    if (this.options.mode === "reference") {
      computePass.setBindGroup(0, this.simulationCalmGroups[this.activeSimulationIndex]);
      computePass.dispatchWorkgroups(groups, groups);
      this.activeSimulationIndex = 1 - this.activeSimulationIndex;
    }
    computePass.setPipeline(this.breakerEventPipeline);
    computePass.setBindGroup(0, this.breakerEventBindGroups[this.activeBreakerEventIndex][this.activeSimulationIndex]);
    computePass.dispatchWorkgroups(Math.ceil(BREAKER_EVENT_RESOLUTION / 64));
    this.activeBreakerEventIndex = 1 - this.activeBreakerEventIndex;
    this.ship?.updateTransform(computePass);
    computePass.end();
    const shoreScene = this.options.scene === "shore";
    const surfaceView = this.context.getCurrentTexture().createView();
    const scenePass = encoder.beginRenderPass({
      label: "Tethys captured atmosphere and terrain",
      colorAttachments: [{ view: shoreScene ? this.sceneColorTexture.createView() : surfaceView, clearValue: { r: 0.18, g: 0.45, b: 0.49, a: 1 }, loadOp: "clear", storeOp: "store" }],
      depthStencilAttachment: { view: this.depthTexture.createView(), depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
      timestampWrites: measureGpu && this.querySet ? { querySet: this.querySet, beginningOfPassWriteIndex: 2 } : undefined,
    });
    scenePass.setPipeline(this.skyPipeline);
    scenePass.setBindGroup(0, this.skyBindGroup);
    scenePass.draw(3);
    scenePass.setPipeline(this.terrainPipeline);
    scenePass.setBindGroup(0, this.terrainBindGroups[this.activeSimulationIndex]);
    const sceneMeshResolution = this.options.scene === "shore" ? 512 : this.options.meshResolution;
    scenePass.draw(sceneMeshResolution * sceneMeshResolution * 6);
    this.ship?.render(scenePass);
    scenePass.end();
    const renderPass = encoder.beginRenderPass({
      label: "Tethys captured-scene water composite",
      colorAttachments: [{ view: surfaceView, clearValue: { r: 0.18, g: 0.45, b: 0.49, a: 1 }, loadOp: shoreScene ? "clear" : "load", storeOp: "store" }],
      depthStencilAttachment: { view: this.depthTexture.createView(), depthReadOnly: true },
      timestampWrites: measureGpu && this.querySet ? { querySet: this.querySet, endOfPassWriteIndex: 3 } : undefined,
    });
    if (shoreScene) {
      renderPass.setPipeline(this.sceneBlitPipeline);
      renderPass.setBindGroup(0, this.sceneBlitBindGroup);
      renderPass.draw(3);
    }
    if (this.options.mode === "reference") {
      renderPass.setPipeline(this.referenceWaterPipeline);
      renderPass.setBindGroup(0, this.referenceWaterBindGroups[this.activeSimulationIndex][this.activeBreakerEventIndex]);
      renderPass.setBindGroup(1, this.referenceWaterSceneBindGroup);
    } else {
      renderPass.setPipeline(this.optimizedWaterPipeline);
      renderPass.setBindGroup(0, this.optimizedWaterBindGroups[this.activeSimulationIndex][this.activeBreakerEventIndex]);
      renderPass.setBindGroup(1, this.optimizedWaterSceneBindGroup);
    }
    renderPass.draw(WATER_CLIPMAP_RESOLUTION * WATER_CLIPMAP_RESOLUTION * 6, WATER_CLIPMAP_LEVELS);
    const drawBreakerPatch = BREAKER_ENABLED && this.options.scene === "open";
    if (drawBreakerPatch && this.options.mode === "reference") {
      renderPass.setPipeline(this.referenceBreakerPatchPipeline);
      renderPass.setBindGroup(0, this.referenceBreakerPatchBindGroups[this.activeSimulationIndex][this.activeBreakerEventIndex]);
      renderPass.setBindGroup(1, this.referenceBreakerSceneBindGroup);
    } else if (drawBreakerPatch) {
      renderPass.setPipeline(this.optimizedBreakerPatchPipeline);
      renderPass.setBindGroup(0, this.optimizedBreakerPatchBindGroups[this.activeSimulationIndex][this.activeBreakerEventIndex]);
      renderPass.setBindGroup(1, this.optimizedBreakerSceneBindGroup);
    }
    if (drawBreakerPatch) renderPass.draw(BREAKER_PATCH_ALONG_RESOLUTION * BREAKER_PATCH_ACROSS_RESOLUTION * 6);
    renderPass.end();
    if (measureGpu && this.querySet && this.queryResolve && this.queryReadback) {
      encoder.resolveQuerySet(this.querySet, 0, 4, this.queryResolve, 0);
      encoder.copyBufferToBuffer(this.queryResolve, 0, this.queryReadback, 0, 32);
      this.queryPending = true;
    }
    this.device.queue.submit([encoder.finish()]);
    this.submitTimes.push(performance.now() - submitStartedAt);
    if (this.submitTimes.length > FRAME_HISTORY) this.submitTimes.shift();
    if (measureGpu && this.queryReadback) {
      void this.queryReadback.mapAsync(GPUMapMode.READ).then(() => {
        if (!this.queryReadback) return;
        const values = new BigUint64Array(this.queryReadback.getMappedRange().slice(0));
        this.queryReadback.unmap();
        const simulation = Number(values[1] - values[0]) / 1_000_000;
        const render = Number(values[3] - values[2]) / 1_000_000;
        if (Number.isFinite(simulation) && simulation >= 0 && simulation < 1000) {
          this.gpuSimulationTimes.push(simulation);
          if (this.gpuSimulationTimes.length > FRAME_HISTORY) this.gpuSimulationTimes.shift();
        }
        if (Number.isFinite(render) && render >= 0 && render < 1000) {
          this.gpuRenderTimes.push(render);
          if (this.gpuRenderTimes.length > FRAME_HISTORY) this.gpuRenderTimes.shift();
        }
        this.queryPending = false;
      }).catch(() => { this.queryPending = false; });
    }
    this.frameIndex += 1;
    this.animationFrame = requestAnimationFrame(this.render);
  };

  getMetrics(): WaterLabMetrics {
    const frameMeanMs = mean(this.frameTimes);
    return {
      ready: this.ready,
      mode: this.options.mode,
      view: this.options.view,
      meshResolution: this.options.meshResolution,
      simulationResolution: this.options.simulationResolution,
      waveScale: this.options.waveScale,
      distantRoughness: this.options.distantRoughness,
      detailRange: this.options.detailRange,
      swellSmoothing: this.options.swellSmoothing,
      longCascadeScale: this.options.longCascadeScale,
      mediumCascadeScale: this.options.mediumCascadeScale,
      fogReach: this.options.fogReach,
      triangles: (this.options.scene === "shore" ? 512 * 512 * 2 : this.options.meshResolution * this.options.meshResolution * 2)
        + WATER_CLIPMAP_RESOLUTION * WATER_CLIPMAP_RESOLUTION * 2
        + (WATER_CLIPMAP_LEVELS - 1) * WATER_CLIPMAP_RESOLUTION * WATER_CLIPMAP_RESOLUTION * 1.5
        + (BREAKER_ENABLED ? BREAKER_PATCH_TRIANGLES : 0)
        + (this.ship?.triangleCount ?? 0),
      simulationBytes: waterSimulationBytes(this.options.simulationResolution),
      simulationSubsteps: this.options.mode === "reference" ? 2 : 1,
      sceneCapturePasses: this.options.scene === "shore" ? 1 : 0,
      disturbanceCount: this.disturbanceCount,
      particleCount: 0,
      frameMeanMs,
      frameP95Ms: percentile(this.frameTimes, 0.95),
      frameP99Ms: percentile(this.frameTimes, 0.99),
      frameMaxMs: this.frameTimes.length ? Math.max(...this.frameTimes) : 0,
      fps: frameMeanMs > 0 ? 1000 / frameMeanMs : 0,
      hitchFrames: this.frameTimes.filter((value) => value > 50).length,
      submitMeanMs: mean(this.submitTimes),
      gpuSimulationMeanMs: this.gpuSimulationTimes.length ? mean(this.gpuSimulationTimes) : null,
      gpuSimulationP95Ms: this.gpuSimulationTimes.length ? percentile(this.gpuSimulationTimes, 0.95) : null,
      gpuRenderMeanMs: this.gpuRenderTimes.length ? mean(this.gpuRenderTimes) : null,
      gpuRenderP95Ms: this.gpuRenderTimes.length ? percentile(this.gpuRenderTimes, 0.95) : null,
      gpuTimestampSamples: Math.min(this.gpuSimulationTimes.length, this.gpuRenderTimes.length),
      adapter: this.shipError ? `${this.adapterLabel} · 船体加载失败: ${this.shipError}` : this.adapterLabel,
      error: this.error,
    };
  }

  private fail(message: string) {
    // Preserve the first validation failure. Later invalid bind-group/command
    // buffer errors are usually consequences and otherwise hide the cause.
    if (!this.error) this.error = message;
    console.error(message);
  }

  dispose() {
    this.disposed = true;
    cancelAnimationFrame(this.animationFrame);
    this.resizeObserver?.disconnect();
    this.removeInteraction();
    this.sceneColorTexture?.destroy();
    this.depthTexture?.destroy();
    this.terrainTexture?.destroy();
    this.waterTextures?.forEach((texture) => texture.destroy());
    this.spectralInitialTextures.forEach((texture) => texture.destroy());
    this.spectralWaveDataTextures.forEach((texture) => texture.destroy());
    this.spectralTwiddleTexture?.destroy();
    this.spectralFields.flat(2).forEach((texture) => texture.destroy());
    this.spectralIfftParamBuffers.forEach((buffer) => buffer.destroy());
    this.worldUniformBuffer?.destroy();
    this.impulseParamBuffer?.destroy();
    this.calmParamBuffer?.destroy();
    this.querySet?.destroy();
    this.queryResolve?.destroy();
    this.queryReadback?.destroy();
    this.ship?.dispose();
    this.ship = null;
    // Frees the adapter slot and turns any still-in-flight work from this
    // engine into no-ops; the lost handler above stays silent once disposed.
    this.device?.destroy();
  }
}

export { TETHYS_REFERENCE_SIMULATION_RESOLUTION };
