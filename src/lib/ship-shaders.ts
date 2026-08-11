import { COLOR_FUNCTIONS, TETHYS_AERIAL_WGSL, WORLD_UNIFORMS } from "@/lib/shared-wgsl";

// The hull rides the same spectral surface the water mesh is built from, so its
// motion has to be derived from those cascades rather than from an independent
// animation curve. Only the long and medium bands are sampled: a 24 m hull does
// not respond to decimetre capillary waves, and including them would jitter it.
const SHIP_WAVE_WGSL = /* wgsl */ `
struct ShipTransform {
  model: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
}

fn shipWaveHeight(p: vec2<f32>, longScale: f32, mediumScale: f32) -> f32 {
  let longUv = fract(p / longScale + vec2<f32>(0.5));
  let mediumUv = fract(p / mediumScale + vec2<f32>(0.5));
  let longHeight = textureSampleLevel(longField, spectrumSampler, longUv, 0.0).b;
  let mediumHeight = textureSampleLevel(mediumField, spectrumSampler, mediumUv, 0.0).b;
  // Same bound-harmonic correction the water surface applies, so the hull sits
  // on the crests the renderer actually draws instead of a symmetric sine.
  return longHeight + mediumHeight
    + 0.14 * (longHeight * longHeight - 0.080)
    + 0.32 * (mediumHeight * mediumHeight - 0.030);
}
`;

/**
 * Rebuilds the hull's rigid-body transform once per frame. Doing this in the
 * vertex stage instead would repeat five identical texture samples across every
 * one of the model's ~64k vertices to produce the same matrix.
 */
export const SHIP_TRANSFORM_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var longField: texture_2d<f32>;
@group(0) @binding(2) var mediumField: texture_2d<f32>;
@group(0) @binding(3) var spectrumSampler: sampler;
@group(0) @binding(4) var<uniform> placement: vec4<f32>;
@group(0) @binding(5) var<uniform> hullSpan: vec4<f32>;
${SHIP_WAVE_WGSL}
@group(0) @binding(6) var<storage, read_write> shipTransform: ShipTransform;

@compute @workgroup_size(1)
fn buildShipTransform() {
  let centre = placement.xy;
  let heading = placement.z;
  let waterLevel = uniforms.sunWater.w;
  let longScale = hullSpan.z;
  let mediumScale = hullSpan.w;

  let forward = vec2<f32>(cos(heading), sin(heading));
  let starboard = vec2<f32>(-forward.y, forward.x);
  let halfLength = hullSpan.x;
  let halfBeam = hullSpan.y;

  let hCentre = shipWaveHeight(centre, longScale, mediumScale);
  let hFore = shipWaveHeight(centre + forward * halfLength, longScale, mediumScale);
  let hAft = shipWaveHeight(centre - forward * halfLength, longScale, mediumScale);
  let hPort = shipWaveHeight(centre - starboard * halfBeam, longScale, mediumScale);
  let hStarboard = shipWaveHeight(centre + starboard * halfBeam, longScale, mediumScale);

  // Heave is the mean over the waterplane rather than the centre sample: a
  // displacement hull integrates buoyancy over its footprint, so averaging
  // keeps it from being thrown by a single crest passing under midships.
  let heave = (hCentre * 2.0 + hFore + hAft + hPort + hStarboard) / 6.0;
  // Damped so a 24 m hull leans with the swell instead of tracking every slope.
  let trim = atan2(hFore - hAft, 2.0 * halfLength) * 0.55;
  let heel = atan2(hStarboard - hPort, 2.0 * halfBeam) * 0.45;

  let cy = cos(heading);
  let sy = sin(heading);
  let cp = cos(trim);
  let sp = sin(trim);
  let cr = cos(heel);
  let sr = sin(heel);

  // Yaw about +Y, then trim about the beam axis, then heel about the keel line.
  // Column-major, matching the mat4x4 the vertex stage multiplies by.
  let m00 = cy * cp;
  let m01 = sp;
  let m02 = -sy * cp;
  let m10 = -cy * sp * cr + sy * sr;
  let m11 = cp * cr;
  let m12 = sy * sp * cr + cy * sr;
  let m20 = cy * sp * sr + sy * cr;
  let m21 = -cp * sr;
  let m22 = -sy * sp * sr + cy * cr;

  shipTransform.model = mat4x4<f32>(
    vec4<f32>(m00, m01, m02, 0.0),
    vec4<f32>(m10, m11, m12, 0.0),
    vec4<f32>(m20, m21, m22, 0.0),
    vec4<f32>(centre.x, waterLevel + heave + placement.w, centre.y, 1.0)
  );
  // The rotation is orthonormal and the scale is uniform, so the rotation block
  // transforms normals correctly without an inverse-transpose.
  shipTransform.normalMatrix = mat4x4<f32>(
    vec4<f32>(m00, m01, m02, 0.0),
    vec4<f32>(m10, m11, m12, 0.0),
    vec4<f32>(m20, m21, m22, 0.0),
    vec4<f32>(0.0, 0.0, 0.0, 1.0)
  );
}
`;

/**
 * Physically-based hull shading. The model ships no TANGENT attribute, so the
 * tangent frame is rebuilt per fragment from screen-space derivatives of
 * position and UV -- correct for a static mesh and cheaper than baking tangents.
 */
export const SHIP_RENDER_SHADER = /* wgsl */ `
${WORLD_UNIFORMS}
${COLOR_FUNCTIONS}
${TETHYS_AERIAL_WGSL}

struct ShipTransform {
  model: mat4x4<f32>,
  normalMatrix: mat4x4<f32>,
}

struct MaterialParams {
  // x: alpha cutoff, y: 0 opaque / 1 masked, z: metallic factor, w: roughness factor
  settings: vec4<f32>,
}

@group(0) @binding(0) var<uniform> uniforms: WorldUniforms;
@group(0) @binding(1) var<storage, read> shipTransform: ShipTransform;

@group(1) @binding(0) var baseColorTexture: texture_2d<f32>;
@group(1) @binding(1) var normalTexture: texture_2d<f32>;
@group(1) @binding(2) var armTexture: texture_2d<f32>;
@group(1) @binding(3) var materialSampler: sampler;
@group(1) @binding(4) var<uniform> material: MaterialParams;

struct Output {
  @builtin(position) position: vec4<f32>,
  @location(0) world: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
}

@vertex fn shipVertex(
  @location(0) position: vec3<f32>,
  @location(1) normal: vec3<f32>,
  @location(2) uv: vec2<f32>,
) -> Output {
  let world = shipTransform.model * vec4<f32>(position, 1.0);
  var output: Output;
  output.position = uniforms.viewProj * world;
  output.world = world.xyz;
  output.normal = (shipTransform.normalMatrix * vec4<f32>(normal, 0.0)).xyz;
  output.uv = uv;
  return output;
}

fn perturbNormal(geometric: vec3<f32>, world: vec3<f32>, uv: vec2<f32>, tangentNormal: vec3<f32>) -> vec3<f32> {
  let dpdxWorld = dpdx(world);
  let dpdyWorld = dpdy(world);
  let dpdxUv = dpdx(uv);
  let dpdyUv = dpdy(uv);
  let perpY = cross(dpdyWorld, geometric);
  let perpX = cross(geometric, dpdxWorld);
  let tangent = perpY * dpdxUv.x + perpX * dpdyUv.x;
  let bitangent = perpY * dpdxUv.y + perpX * dpdyUv.y;
  // Degenerate UVs would otherwise produce a zero-length basis and NaN normals.
  let scale = inverseSqrt(max(max(dot(tangent, tangent), dot(bitangent, bitangent)), 1e-12));
  return normalize(mat3x3<f32>(tangent * scale, bitangent * scale, geometric) * tangentNormal);
}

fn ggxSpecular(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
  let H = normalize(V + L);
  let ndh = max(dot(N, H), 0.0);
  let ndv = max(dot(N, V), 0.0001);
  let ndl = max(dot(N, L), 0.0);
  let a = max(roughness * roughness, 0.002);
  let a2 = a * a;
  let denominator = ndh * ndh * (a2 - 1.0) + 1.0;
  let distribution = a2 / (3.14159265 * denominator * denominator);
  let k = a * 0.5;
  let visibility = 1.0 / ((ndv * (1.0 - k) + k) * (ndl * (1.0 - k) + k));
  return distribution * visibility * 0.25;
}

@fragment fn shipFragment(input: Output) -> @location(0) vec4<f32> {
  let baseColor = textureSample(baseColorTexture, materialSampler, input.uv);
  if (material.settings.y > 0.5 && baseColor.a < material.settings.x) { discard; }

  let arm = textureSample(armTexture, materialSampler, input.uv);
  let occlusion = arm.r;
  let roughness = clamp(arm.g * material.settings.w, 0.045, 1.0);
  let metallic = clamp(arm.b * material.settings.z, 0.0, 1.0);

  let tangentNormal = normalize(textureSample(normalTexture, materialSampler, input.uv).xyz * 2.0 - 1.0);
  let geometric = normalize(input.normal);
  // Rigging is sub-pixel thin at any real viewing distance, so its screen-space
  // derivatives span unrelated surfaces and the reconstructed tangent frame goes
  // wild -- that showed as scattered black fragments along the shrouds. Fade the
  // perturbation out as the UV footprint explodes and keep the geometric normal.
  let uvFootprint = max(length(dpdx(input.uv)), length(dpdy(input.uv)));
  let tangentTrust = 1.0 - smoothstep(0.012, 0.055, uvFootprint);
  var N = normalize(mix(geometric, perturbNormal(geometric, input.world, input.uv, tangentNormal), tangentTrust));
  let V = normalize(uniforms.cameraTime.xyz - input.world);
  // Every material on this asset is double-sided; flip inward-facing shading
  // normals so backfaces are lit rather than black.
  if (dot(N, V) < 0.0) { N = -N; }

  let L = normalize(uniforms.sunWater.xyz);
  let ndl = max(dot(N, L), 0.0);

  let albedo = baseColor.rgb;
  let diffuseColor = albedo * (1.0 - metallic);
  let f0 = mix(vec3<f32>(0.04), albedo, metallic);

  let sunColor = vec3<f32>(1.0, 0.955, 0.88) * 5.6;
  var color = diffuseColor * sunColor * ndl / 3.14159265;
  let fresnel = f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - max(dot(N, normalize(V + L)), 0.0), 5.0);
  color += fresnel * sunColor * ggxSpecular(N, V, L, roughness) * ndl;

  // Hemispheric ambient: sky above, water-reflected bounce below, matching the
  // palette the ocean shader resolves to so the hull sits in the same light.
  let skyAmbient = vec3<f32>(0.62, 0.74, 0.88);
  let waterBounce = vec3<f32>(0.24, 0.46, 0.47);
  let ambient = mix(waterBounce, skyAmbient, N.y * 0.5 + 0.5);
  // Partial AO application: the maps are baked for interior occlusion and go
  // very dark in the rigging, which reads as dirt under open sky.
  color += diffuseColor * ambient * mix(1.0, occlusion, 0.72) * 1.25;

  color = tethysAerialColor(color, input.world, uniforms.cameraTime.xyz, uniforms.environment.w, uniforms.terrain.w > 0.5, 0.0);
  return vec4<f32>(linearToSrgb(aces(color)), 1.0);
}
`;
