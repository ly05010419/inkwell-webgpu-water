import { MeshPhysicalNodeMaterial } from "three/webgpu";
import { Matrix3, Vector3, Vector4 } from "three";
import {
  color,
  and,
  atan,
  cos,
  dot,
  float,
  max,
  mix,
  positionLocal,
  min,
  normalize,
  texture,
  uniformTexture,
  uniform,
  vec2,
  vec3,
  select,
  sin,
  smoothstep,
} from "three/tsl";
import type * as THREE from "three";

import type { ThreeWaterWavesImpl } from "./spectral-waves";
import type { WaterHeightField } from "./types";

export type ThreeWaterMaterialState = {
  material: MeshPhysicalNodeMaterial;
  timeNode: { value: number };
  waveScaleNode: { value: number };
  detailRangeNode: { value: number };
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
  const timeNode = uniform(0);
  const waveScaleNode = waves.getWaveScaleNode();
  waveScaleNode.value = waveScale;
  const detailRangeNode = uniform(detailRange);
  const roughnessNode = uniform(distantRoughness);
  const patchTextureNode = uniformTexture(heightField.texture);
  const patchBoundsNode = uniform(new Vector4());
  const patchEnabledNode = uniform(0);
  const atmosphereNode = uniform(1);
  const dayLightNode = uniform(1);
  const sunDirectionNode = uniform(new Vector3(0.4, 0.8, 0.2));
  const envRotationNode = uniform(new Matrix3());
  const swellSmoothingNode = uniform(1);
  const windowNormalNode = uniform(new Vector3(0, 0, 1));
  const windowEastNode = uniform(new Vector3(1, 0, 0));
  const windowSouthNode = uniform(new Vector3(0, -1, 0));
  const windowArcNode = uniform(0.75);
  const theta = positionLocal.x.mul(windowArcNode);
  const azimuth = positionLocal.y.mul(Math.PI * 2);
  const tangent = windowEastNode.mul(cos(azimuth)).add(windowSouthNode.mul(sin(azimuth)));
  const surfaceNormal = normalize(windowNormalNode.mul(cos(theta)).add(tangent.mul(sin(theta))));
  const surfacePosition = surfaceNormal;
  const longitude = atan(surfacePosition.x, surfacePosition.z);
  const latitude = surfacePosition.y.asin();
  const planarX = longitude.sub(projection.lonOrigin ?? 0).mul(projection.metersPerRadianLon);
  const planarZ = float(projection.latOrigin ?? 0).sub(latitude).mul(projection.metersPerRadianLat);
  const globalUv = vec2(planarX.div(projection.worldSize).add(0.5), planarZ.div(projection.worldSize).add(0.5));
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
  const heightNode = globalHeightNode.mul(patchMask.oneMinus()).add(patchHeightNode.mul(patchMask));
  const waveUv = vec2(planarX.div(240).add(0.5), planarZ.div(240).add(0.5));
  const longField = texture(uniformTexture(waves.getCascadeTexture(0)), waveUv);
  const mediumField = texture(uniformTexture(waves.getCascadeTexture(1)), waveUv.mul(240 / 64));
  const shortField = texture(uniformTexture(waves.getCascadeTexture(2)), waveUv.mul(240 / 12));
  const longHeight = longField.b.mul(waveScaleNode);
  const mediumHeight = mediumField.b.mul(waveScaleNode);
  const waveHeight = longHeight.add(mediumHeight)
    .add(longHeight.mul(longHeight).sub(waveScaleNode.mul(0.08)).mul(0.14))
    .add(mediumHeight.mul(mediumHeight).sub(waveScaleNode.mul(0.03)).mul(0.32))
    .mul(swellSmoothingNode);
  const longSlope = texture(uniformTexture(waves.getCascadeDerivativeTexture(0)), waveUv).rg.mul(waveScaleNode);
  const mediumSlope = texture(uniformTexture(waves.getCascadeDerivativeTexture(1)), waveUv.mul(240 / 64)).rg.mul(waveScaleNode);
  const shortSlope = shortField.rg.mul(waveScaleNode);
  const longDerivatives = texture(uniformTexture(waves.getCascadeDerivativeTexture(0)), waveUv);
  const mediumDerivatives = texture(uniformTexture(waves.getCascadeDerivativeTexture(1)), waveUv.mul(240 / 64));
  const crossDerivative = longDerivatives.a.add(mediumDerivatives.a).mul(waveScaleNode);
  const horizontalDerivative = longDerivatives.ba.add(mediumDerivatives.ba).mul(waveScaleNode);
  const jacobian = vec2(1).add(horizontalDerivative).x.mul(vec2(1).add(horizontalDerivative).y).sub(crossDerivative.mul(crossDerivative));
  const foam = smoothstep(0.0, 0.42, max(0, jacobian.oneMinus())).mul(0.24);
  const fieldHeight = heightNode.mul(detailRangeNode);
  const sunlight = max(dot(surfaceNormal, sunDirectionNode), 0).mul(0.25).add(0.75);
  const atmosphere = atmosphereNode.mul(0.45).add(0.55);
  const daylight = dayLightNode.mul(0.35).add(0.65);

  const material = new MeshPhysicalNodeMaterial({
    color: 0x0a8298,
    roughness: 0.18,
    metalness: 0,
    transparent: false,
  });
  material.name = "Three Tethys globe water";
  material.positionNode = surfacePosition.mul(radius).add(surfaceNormal.mul(waveHeight.add(fieldHeight)));
  material.normalNode = normalize(surfaceNormal.add(vec3(
    longSlope.x.add(mediumSlope.x).add(shortSlope.x.mul(0.4)).negate(),
    0,
    longSlope.y.add(mediumSlope.y).add(shortSlope.y.mul(0.4)).negate(),
  ).mul(0.22)));
  material.colorNode = mix(color(0x0a8298), color(0xa6e5df), foam).mul(sunlight).mul(atmosphere).mul(daylight);
  material.roughnessNode = roughnessNode.mul(0.12).add(0.12);
  if (environment) material.envMap = environment;

  // Keep the waves object attached to the material for integrations that need
  // to inspect the shared cascade state without creating a second simulation.
  (material.userData as { waves?: ThreeWaterWavesImpl }).waves = waves;
  return {
    material,
    timeNode,
    waveScaleNode,
    detailRangeNode,
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
