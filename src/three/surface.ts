import { MeshPhysicalNodeMaterial } from "three/webgpu";
import { Vector4 } from "three";
import {
  color,
  and,
  normalLocal,
  positionLocal,
  sin,
  texture,
  uniformTexture,
  uniform,
  uv,
  vec2,
  select,
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
  patch: WaterHeightField | null;
};

export function createWaterMaterial(
  heightField: WaterHeightField,
  environment: THREE.Texture | null | undefined,
  waves: ThreeWaterWavesImpl,
  waveScale: number,
  detailRange: number,
  distantRoughness: number,
): ThreeWaterMaterialState {
  const timeNode = uniform(0);
  const waveScaleNode = uniform(waveScale);
  const detailRangeNode = uniform(detailRange);
  const roughnessNode = uniform(distantRoughness);
  const patchTextureNode = uniformTexture(heightField.texture);
  const patchBoundsNode = uniform(new Vector4());
  const patchEnabledNode = uniform(0);
  const globeUv = uv();
  const longitude = globeUv.x.mul(Math.PI * 2).sub(Math.PI);
  const latitude = globeUv.y.sub(0.5).mul(Math.PI);
  const patchUv = vec2(
    longitude.sub(patchBoundsNode.x).div(patchBoundsNode.z.sub(patchBoundsNode.x)),
    latitude.sub(patchBoundsNode.y).div(patchBoundsNode.w.sub(patchBoundsNode.y)),
  );
  const patchInside = and(
    longitude.greaterThanEqual(patchBoundsNode.x),
    longitude.lessThanEqual(patchBoundsNode.z),
    latitude.greaterThanEqual(patchBoundsNode.y),
    latitude.lessThanEqual(patchBoundsNode.w),
  );
  const patchMask = select(patchInside, patchEnabledNode, 0);
  const globalHeightNode = texture(heightField.texture, globeUv).r;
  const patchHeightNode = texture(patchTextureNode, patchUv).r;
  const heightNode = globalHeightNode.mul(patchMask.oneMinus()).add(patchHeightNode.mul(patchMask));
  const phase = positionLocal.x.mul(0.019).add(positionLocal.z.mul(0.014)).add(timeNode.mul(0.7));
  const secondaryPhase = positionLocal.x.mul(-0.041).add(positionLocal.z.mul(0.029)).sub(timeNode.mul(0.43));
  const waveHeight = sin(phase).mul(1.35).add(sin(secondaryPhase).mul(0.45)).mul(waveScaleNode);
  const fieldHeight = heightNode.sub(0.5).mul(0.18).mul(detailRangeNode);

  const material = new MeshPhysicalNodeMaterial({
    color: 0x0a8298,
    roughness: 0.18,
    metalness: 0,
    transparent: false,
  });
  material.name = "Three Tethys globe water";
  material.positionNode = positionLocal.add(normalLocal.mul(waveHeight.add(fieldHeight)));
  material.colorNode = color(0x0a8298);
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
    patch: null,
  };
}
