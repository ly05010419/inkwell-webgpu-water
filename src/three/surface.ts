import { DoubleSide, Matrix3, Vector3, Vector4 } from "three";
import { MeshPhysicalNodeMaterial } from "three/webgpu";
import {
  and,
  atan,
  color,
  cos,
  cross,
  distance,
  dot,
  float,
  max,
  min,
  mix,
  normalize,
  positionLocal,
  select,
  sin,
  smoothstep,
  texture,
  uniform,
  uniformTexture,
  vec2,
  vec3,
} from "three/tsl";
import type * as THREE from "three";

import { createSpectralWaveSampleNodes } from "./wave-sampling";
import type { ThreeWaterWavesImpl } from "./spectral-waves";
import type { WaterHeightField } from "./types";

export type ThreeWaterMaterialState = {
  material: MeshPhysicalNodeMaterial;
  timeNode: { value: number };
  waveScaleNode: { value: number };
  detailRangeNode: { value: number };
  pixelWorldScaleNode: { value: number };
  cameraPositionNode: { value: Vector3 };
  roughnessNode: { value: number };
  patchTextureNode: ReturnType<typeof uniformTexture>;
  patchBoundsNode: { value: Vector4 };
  patchEnabledNode: { value: number };
  atmosphereNode: { value: number };
  dayLightNode: { value: number };
  sunDirectionNode: { value: Vector3 };
  envRotationNode: { value: Matrix3 };
  swellSmoothingNode: { value: number };
  windowNormalNode: { value: Vector3 };
  windowEastNode: { value: Vector3 };
  windowSouthNode: { value: Vector3 };
  windowArcNode: { value: number };
  patch: WaterHeightField | null;
};

export function createWaterMaterial(
  heightField: WaterHeightField,
  environment: THREE.Texture | null | undefined,
  waves: ThreeWaterWavesImpl,
  waveScale: number,
  detailRange: number,
  distantRoughness: number,
  radius: number,
  projection: { worldSize: number; metersPerRadianLon: number; metersPerRadianLat: number; lonOrigin?: number; latOrigin?: number },
): ThreeWaterMaterialState {
  const timeNode = uniform(0, "float");
  const waveScaleNode = waves.getWaveScaleNode();
  waveScaleNode.value = waveScale;
  const detailRangeNode = uniform(detailRange, "float");
  const pixelWorldScaleNode = uniform(1 / Math.max(radius, 1), "float");
  const cameraPositionNode = uniform(new Vector3(0, 0, radius * 2));
  const roughnessNode = uniform(distantRoughness, "float");
  const patchTextureNode = uniformTexture(heightField.texture);
  const patchBoundsNode = uniform(new Vector4());
  const patchEnabledNode = uniform(0, "float");
  const atmosphereNode = uniform(1, "float");
  const dayLightNode = uniform(1, "float");
  const sunDirectionNode = uniform(new Vector3(0.4, 0.8, 0.2));
  const envRotationNode = uniform(new Matrix3());
  const swellSmoothingNode = uniform(1, "float");
  const windowNormalNode = uniform(new Vector3(0, 0, 1));
  const windowEastNode = uniform(new Vector3(1, 0, 0));
  const windowSouthNode = uniform(new Vector3(0, -1, 0));
  const windowArcNode = uniform(0.75, "float");

  // The fixed geometry stores polar coordinates, while the node graph maps
  // them to the camera-facing spherical window every frame. This preserves
  // globe curvature without rewriting vertices on the CPU.
  const theta = positionLocal.x.mul(windowArcNode);
  const azimuth = positionLocal.y.mul(Math.PI * 2);
  const tangent = windowEastNode.mul(cos(azimuth)).add(windowSouthNode.mul(sin(azimuth)));
  const surfaceNormal = normalize(windowNormalNode.mul(cos(theta)).add(tangent.mul(sin(theta))));
  const spherePosition = surfaceNormal.mul(radius);

  // +X is east and +Z is the prime meridian/south-facing planar axis. The
  // analytic tangent frame avoids using the camera-local east/south vectors for
  // projection, which would skew height fields whenever the camera turns.
  const eastRaw = vec3(surfaceNormal.z, 0, surfaceNormal.x.negate());
  const eastLength = eastRaw.length();
  const east = select(eastLength.greaterThan(1e-4), eastRaw.div(max(eastLength, 1e-4)), windowEastNode);
  const south = normalize(cross(east, surfaceNormal));
  const longitude = atan(surfaceNormal.x, surfaceNormal.z);
  const latitude = surfaceNormal.y.asin();
  const planarX = longitude.sub(projection.lonOrigin ?? 0).mul(projection.metersPerRadianLon);
  const planarZ = float(projection.latOrigin ?? 0).sub(latitude).mul(projection.metersPerRadianLat);
  const globalUv = vec2(planarX.div(projection.worldSize).add(0.5), planarZ.div(projection.worldSize).add(0.5));

  // Bounds are supplied in geographic degrees by the host. The controller
  // converts them to radians before updating this node. V is flipped to match
  // the host's raster convention, and the outer 15% fades back to the global
  // height field rather than producing a hard LOD seam.
  const patchUv = vec2(
    longitude.sub(patchBoundsNode.x).div(max(patchBoundsNode.z.sub(patchBoundsNode.x), 1e-5)),
    patchBoundsNode.w.sub(latitude).div(max(patchBoundsNode.w.sub(patchBoundsNode.y), 1e-5)),
  );
  const patchInside = and(
    longitude.greaterThanEqual(patchBoundsNode.x),
    longitude.lessThanEqual(patchBoundsNode.z),
    latitude.greaterThanEqual(patchBoundsNode.y),
    latitude.lessThanEqual(patchBoundsNode.w),
  );
  const patchU = longitude.sub(patchBoundsNode.x).div(max(patchBoundsNode.z.sub(patchBoundsNode.x), 1e-5));
  const patchV = latitude.sub(patchBoundsNode.y).div(max(patchBoundsNode.w.sub(patchBoundsNode.y), 1e-5));
  const edgeDistance = min(min(patchU, patchV), min(patchU.oneMinus(), patchV.oneMinus()));
  const edgeFade = smoothstep(0, 0.15, edgeDistance);
  const patchMask = select(patchInside, patchEnabledNode.mul(edgeFade), 0);
  const globalHeightNode = texture(heightField.texture, globalUv).r;
  const patchHeightNode = texture(patchTextureNode, patchUv).r;
  const heightNode = mix(globalHeightNode, patchHeightNode, patchMask);
  // Height fields encode seabed altitude/depth in metres. They affect shallow
  // attenuation and shading only; adding them to the sphere position would
  // incorrectly drop deep-ocean vertices thousands of metres below the water.
  const depthNode = max(0, heightNode.negate());
  const shallowAttenuation = smoothstep(0.14, 2.7, depthNode);

  const waveSample = createSpectralWaveSampleNodes(
    waves,
    planarX,
    planarZ,
    (field, uv) => texture(uniformTexture(field), uv),
  );
  const surfaceHeight = waveSample.height.mul(shallowAttenuation);
  const displacement = waveSample.displacement.mul(shallowAttenuation);
  const worldPosition = spherePosition
    .add(surfaceNormal.mul(surfaceHeight))
    .add(east.mul(displacement.x))
    .add(south.mul(displacement.y));

  // DetailRange and swellSmoothing are screen-space quality controls. They
  // fade unresolved slopes/normals and return their energy to roughness; they
  // never scale seabed height or the actual wave displacement.
  const viewDistance = distance(cameraPositionNode, worldPosition);
  const pixelWorldSize = viewDistance.mul(pixelWorldScaleNode);
  const fade = (wavelength: number) => select(
    swellSmoothingNode.lessThanEqual(0),
    1,
    smoothstep(3, 14, float(wavelength).div(pixelWorldSize).mul(detailRangeNode).div(max(swellSmoothingNode, 0.001))),
  );
  const longFade = fade(240);
  const mediumFade = fade(64);
  const shortFade = fade(12);
  const swellFade = longFade.add(mediumFade).mul(0.5);
  const smoothedSlope = waveSample.slope.mul(swellFade).mul(shallowAttenuation);
  const shortSlope = waveSample.shortSlope.mul(shortFade);
  const crossDerivative = waveSample.crossDerivative
    .mul(shallowAttenuation)
    .mul(longFade.add(mediumFade).mul(0.5));
  const horizontalDerivative = waveSample.horizontalDerivative
    .mul(shallowAttenuation)
    .mul(longFade.add(mediumFade).mul(0.5));
  const tangentEast = east.mul(1).add(east.mul(horizontalDerivative.x)).add(surfaceNormal.mul(smoothedSlope.x)).add(south.mul(crossDerivative));
  const tangentSouth = east.mul(crossDerivative).add(surfaceNormal.mul(smoothedSlope.y)).add(south.mul(1).add(south.mul(horizontalDerivative.y)));
  const displacedNormal = normalize(cross(tangentSouth, tangentEast));
  const shadedNormal = normalize(displacedNormal
    .add(east.mul(shortSlope.x.negate()).add(south.mul(shortSlope.y.negate())).mul(0.42)));
  const jacobian = max(0, waveSample.jacobian.oneMinus()).mul(shallowAttenuation);
  const foam = smoothstep(0.055, 0.40, jacobian).mul(
    float(1).sub(smoothstep(95 * detailRange, 188 * detailRange, viewDistance)),
  );
  const nearshorePulse = sin(planarX.mul(0.08).add(planarZ.mul(0.052)).add(timeNode.mul(0.8))).mul(0.5).add(0.5);
  const nearshoreFoam = float(1).sub(smoothstep(0.035, 0.62, depthNode))
    .mul(nearshorePulse)
    .mul(0.10)
    .mul(float(1).sub(smoothstep(95 * detailRange, 188 * detailRange, viewDistance)));
  const foamAmount = max(foam, nearshoreFoam);
  // Use the analytic sphere normal for the broad body colour. The displaced
  // normal remains attached to the material's lighting path, while keeping
  // this low-frequency palette free of per-cell normal banding on the polar
  // window mesh.
  const sunlight = max(dot(surfaceNormal, sunDirectionNode), 0).mul(0.25).add(0.75);
  const atmosphere = atmosphereNode.mul(0.45).add(0.55);
  const daylight = dayLightNode.mul(0.35).add(0.65);
  const depthMix = smoothstep(0.25, 24, depthNode);
  const shallowColor = color(0x148e88);
  const deepColor = color(0x04254d);
  const waterColor = mix(shallowColor, deepColor, depthMix).mul(sunlight).mul(atmosphere).mul(daylight);
  const foamColor = color(0xd7f4ec).mul(foamAmount.mul(0.70));
  // A sun-facing glint keeps the broad Fresnel response in the PBR material,
  // while this explicit term remains stable on the polar window mesh.
  const glint = color(0xa4e9df).mul(max(sunlight.sub(0.75), 0).mul(0.42).mul(daylight));

  const material = new MeshPhysicalNodeMaterial({
    color: 0x0a8298,
    roughness: 0.18,
    metalness: 0,
    side: DoubleSide,
    transparent: false,
  });
  material.name = "Three Tethys globe water";
  material.positionNode = worldPosition;
  material.normalNode = shadedNormal;
  material.colorNode = waterColor.add(foamColor).add(glint);
  // The adapter owns a sun-direction scalar rather than requiring callers to
  // add a Three.js light just to make the water visible. A restrained emissive
  // contribution preserves that authored daylight/atmosphere palette while
  // still allowing host lights and environment reflections to add highlights.
  material.emissiveNode = waterColor.add(foamColor).add(glint).mul(0.55);
  material.roughnessNode = roughnessNode.mul(0.05).add(0.035).add(shortSlope.length().mul(0.05)).add(
    float(1).sub(shortFade).mul(0.035),
  );
  material.metalnessNode = float(0);
  if (environment) material.envMap = environment;

  // Keep the waves object attached to the material for integrations that need
  // to inspect the shared cascade state without creating a second simulation.
  (material.userData as { waves?: ThreeWaterWavesImpl }).waves = waves;
  return {
    material,
    timeNode,
    waveScaleNode,
    detailRangeNode,
    pixelWorldScaleNode,
    cameraPositionNode,
    roughnessNode,
    patchTextureNode,
    patchBoundsNode,
    patchEnabledNode,
    atmosphereNode,
    dayLightNode,
    sunDirectionNode,
    envRotationNode,
    swellSmoothingNode,
    windowNormalNode,
    windowEastNode,
    windowSouthNode,
    windowArcNode,
    patch: null,
  };
}
