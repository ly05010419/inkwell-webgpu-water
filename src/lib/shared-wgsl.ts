// WGSL shared by the water engine and the hull renderer. It lives apart from
// both so the hull's shaders can reuse the world uniform layout, tonemapping and
// aerial perspective without importing the engine, which would otherwise form an
// import cycle and leave these template literals undefined at module init.

export const WORLD_UNIFORMS = /* wgsl */ `
struct WorldUniforms {
  viewProj: mat4x4<f32>,
  cameraTime: vec4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  cameraForward: vec4<f32>,
  sunWater: vec4<f32>,
  terrain: vec4<f32>,
  simulation: vec4<f32>,
  player: vec4<f32>,
  interaction: vec4<f32>,
  environment: vec4<f32>,
  // x: swell amplitude multiplier, y: its square (for the bound-harmonic mean
  // terms, which are quadratic in wave height). z: how much of the faded-out
  // capillary slope is returned to BRDF roughness. w: multiplier on the
  // distance over which surface detail survives.
  waves: vec4<f32>,
  // x: multiplier on where the open ocean's radial fog closes; 0 disables it.
  atmosphere: vec4<f32>,
}
`;

export const COLOR_FUNCTIONS = /* wgsl */ `
fn linearToSrgb(value: vec3<f32>) -> vec3<f32> {
  return mix(value * 12.92, 1.055 * pow(max(value, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055, step(vec3<f32>(0.0031308), value));
}

fn aces(color: vec3<f32>) -> vec3<f32> {
  return clamp((color * (2.51 * color + vec3<f32>(0.03))) / (color * (2.43 * color + vec3<f32>(0.59)) + vec3<f32>(0.14)), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn cloudHash3(pInput: vec3<f32>) -> f32 {
  var p = fract(pInput * 0.1031);
  p += vec3<f32>(dot(p, p.yzx + vec3<f32>(33.33)));
  return fract((p.x + p.y) * p.z);
}

fn cloudNoise3(p: vec3<f32>) -> f32 {
  let cell = floor(p);
  var local = fract(p);
  local = local * local * (vec3<f32>(3.0) - 2.0 * local);
  let n000 = cloudHash3(cell + vec3<f32>(0.0, 0.0, 0.0));
  let n100 = cloudHash3(cell + vec3<f32>(1.0, 0.0, 0.0));
  let n010 = cloudHash3(cell + vec3<f32>(0.0, 1.0, 0.0));
  let n110 = cloudHash3(cell + vec3<f32>(1.0, 1.0, 0.0));
  let n001 = cloudHash3(cell + vec3<f32>(0.0, 0.0, 1.0));
  let n101 = cloudHash3(cell + vec3<f32>(1.0, 0.0, 1.0));
  let n011 = cloudHash3(cell + vec3<f32>(0.0, 1.0, 1.0));
  let n111 = cloudHash3(cell + vec3<f32>(1.0, 1.0, 1.0));
  let nearPlane = mix(mix(n000, n100, local.x), mix(n010, n110, local.x), local.y);
  let farPlane = mix(mix(n001, n101, local.x), mix(n011, n111, local.x), local.y);
  return mix(nearPlane, farPlane, local.z);
}

fn skyColor(direction: vec3<f32>, time: f32, sunDirection: vec3<f32>) -> vec3<f32> {
  let elevation = direction.y;
  let upper = smoothstep(-0.035, 0.34, elevation);
  // Lower-energy linear-light values leave headroom for the sun and clouds.
  // The old near-white dome flattened both the sky and its water reflection.
  var color = mix(vec3<f32>(0.34, 0.54, 0.64), vec3<f32>(0.070, 0.26, 0.43), upper);
  color = mix(vec3<f32>(0.055, 0.22, 0.31), color, smoothstep(-0.34, 0.055, elevation));
  let sunDot = max(dot(direction, sunDirection), 0.0);
  let horizon = exp(-abs(elevation) * 12.0);
  color = mix(color, vec3<f32>(0.72, 0.55, 0.31), pow(sunDot, 4.0) * horizon * 0.12);
  color += vec3<f32>(0.28, 0.42, 0.62) * pow(sunDot, 20.0) * 0.08;
  let drift = vec3<f32>(time * 0.0040, -time * 0.0014, time * 0.0023);
  let cloudPoint = direction * 10.5 + drift;
  let cloudField = cloudNoise3(cloudPoint) * 0.54
    + cloudNoise3(cloudPoint * 2.03 + vec3<f32>(7.1, -3.4, 5.8)) * 0.29
    + cloudNoise3(cloudPoint * 4.07 + vec3<f32>(-2.7, 9.3, 1.9)) * 0.17;
  let envelope = smoothstep(-0.045, 0.018, elevation) * (1.0 - smoothstep(0.30, 0.43, elevation));
  let cloudBody = smoothstep(0.535, 0.68, cloudField) * envelope;
  let cloudCore = smoothstep(0.63, 0.78, cloudField) * envelope;
  let cloudEdge = (smoothstep(0.51, 0.59, cloudField) - smoothstep(0.67, 0.76, cloudField)) * envelope;
  let cloudShade = mix(vec3<f32>(0.42, 0.50, 0.55), vec3<f32>(0.82, 0.76, 0.63), pow(sunDot, 0.35));
  color = mix(color, cloudShade, cloudBody * 0.42);
  color -= vec3<f32>(0.08, 0.10, 0.12) * cloudCore * (1.0 - sunDot) * 0.28;
  color += vec3<f32>(0.60, 0.49, 0.30) * cloudEdge * sunDot * 0.055;
  return color;
}
`;

export const TETHYS_AERIAL_WGSL = /* wgsl */ `
fn tethysAerialColor(color: vec3<f32>, world: vec3<f32>, cameraPos: vec3<f32>, worldScale: f32, underwater: bool, dryLand: f32) -> vec3<f32> {
  // worldScale arrives in environment.w: 1 for the island scene, which is then
  // bit-for-bit unchanged. Underwater density is a property of the medium, not
  // of how much world surrounds it, so it stays unscaled.
  let distanceToEye = distance(cameraPos, world);
  let density = select(0.00155 / worldScale, 0.0075, underwater);
  var fog = 1.0 - exp(-max(distanceToEye - select(20.0 * worldScale, 2.0, underwater), 0.0) * density);
  let radial = length(world.xz);
  // The open ocean's radial wall is adjustable and off by default. It never
  // modelled atmosphere -- it concealed the far edge of a finite water mesh --
  // so turning it off buys view distance at the cost of exposing that edge when
  // the camera is pulled back. The island scene keeps its own unconditionally,
  // where the ring bounds an authored world rather than hiding an artifact.
  let fogReach = uniforms.atmosphere.x;
  let oceanRadialFog = select(
    smoothstep(116.0 * worldScale * fogReach, 145.0 * worldScale * fogReach, radial),
    0.0,
    fogReach <= 0.0);
  let islandRadialFog = smoothstep(166.0 * worldScale, 205.0 * worldScale, radial) * 0.58;
  fog = clamp(fog + mix(oceanRadialFog, islandRadialFog, dryLand), 0.0, mix(0.99, 0.72, dryLand));
  let waterAerial = select(vec3<f32>(0.42, 0.66, 0.71), vec3<f32>(0.012, 0.205, 0.185), underwater);
  let aerial = mix(waterAerial, vec3<f32>(0.24, 0.39, 0.43), dryLand);
  return mix(color, aerial, fog);
}
`;
