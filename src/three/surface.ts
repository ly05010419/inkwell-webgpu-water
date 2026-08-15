import { MeshPhysicalNodeMaterial } from "three/webgpu";
import { Matrix3, Vector3, Vector4 } from "three";
import {
  color,
  and,
  atan,
  dot,
  float,
  max,
  normalLocal,
  positionLocal,
  min,
  texture,
  uniformTexture,
  uniform,
  vec2,
  select,
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
  const surfacePosition = positionLocal.div(radius);
  const longitude = atan(surfacePosition.x, surfacePosition.z);
  const latitude = surfacePosition.y.asin();
  const planarX = longitude.sub(projection.lonOrigin ?? 0).mul(projection.metersPerRadianLon);
  const planarZ = float(projection.latOrigin ?? 0).sub(latitude).mul(projection.metersPerRadianLat);
  const globalUv = vec2(planarX.div(projection.worldSize).add(0.5), planarZ.div(projection.worldSize).add(0.5));
  const patchUv = vec2(
    longitude.sub(patchBoundsNode.x).div(patchBoundsNode.z.sub(patchBoundsNode.x)),
    patchBoundsNode.w.sub(latitude).div(patchBoundsNode.w.sub(patchBoundsNode.y)),
  );
  const patchInside = and(
    longitude.greaterThanEqual(patchBoundsNode.x),
    longitude.lessThanEqual(patchBoundsNode.z),
    latitude.greaterThanEqual(patchBoundsNode.y),
    latitude.lessThanEqual(patchBoundsNode.w),
  );
  const patchU = longitude.sub(patchBoundsNode.x).div(patchBoundsNode.z.sub(patchBoundsNode.x));
  const patchV = latitude.sub(patchBoundsNode.y).div(patchBoundsNode.w.sub(patchBoundsNode.y));
  const edgeDistance = min(min(patchU, patchV), min(patchU.oneMinus(), patchV.oneMinus()));
  const edgeFade = smoothstep(0, 0.15, edgeDistance);
  const patchMask = select(patchInside, patchEnabledNode.mul(edgeFade), 0);
  const globalHeightNode = texture(heightField.texture, globalUv).r;
  const patchHeightNode = texture(patchTextureNode, patchUv).r;
  const heightNode = globalHeightNode.mul(patchMask.oneMinus()).add(patchHeightNode.mul(patchMask));
  const waveUv = vec2(planarX.div(240).add(0.5), planarZ.div(240).add(0.5));
  const wave0 = texture(uniformTexture(waves.getCascadeTexture(0)), waveUv).b;
  const wave1 = texture(uniformTexture(waves.getCascadeTexture(1)), waveUv.mul(240 / 64)).b;
  const wave2 = texture(uniformTexture(waves.getCascadeTexture(2)), waveUv.mul(240 / 12)).b;
  const waveHeight = wave0.add(wave1.mul(0.78)).add(wave2.mul(0.32)).mul(waveScaleNode).mul(swellSmoothingNode);
  const fieldHeight = heightNode.mul(detailRangeNode);
  const sunlight = max(dot(normalLocal, sunDirectionNode), 0).mul(0.25).add(0.75);
  const atmosphere = atmosphereNode.mul(0.45).add(0.55);
  const daylight = dayLightNode.mul(0.35).add(0.65);

  const material = new MeshPhysicalNodeMaterial({
    color: 0x0a8298,
    roughness: 0.18,
    metalness: 0,
    transparent: false,
  });
  material.name = "Three Tethys globe water";
  material.positionNode = positionLocal.add(normalLocal.mul(waveHeight.add(fieldHeight)));
  material.colorNode = color(0x0a8298).mul(sunlight).mul(atmosphere).mul(daylight);
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
    patch: null,
  };
}
