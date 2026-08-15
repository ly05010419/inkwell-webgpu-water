import type * as THREE from "three";
import type { StorageBufferAttribute, WebGPURenderer } from "three/webgpu";

export type WaterHeightField = {
  texture: THREE.Texture;
  channel?: "r";
  metersPerPixel?: number;
};

export type WaterPatchField = {
  texture: THREE.Texture;
  bounds: {
    /** Geographic degrees, matching the host world's LOD metadata. */
    lonMin: number;
    latMin: number;
    lonMax: number;
    latMax: number;
  };
};

export type GlobeProjection = {
  worldSize: number;
  metersPerRadianLon: number;
  metersPerRadianLat: number;
  lonOrigin?: number;
  latOrigin?: number;
};

export type ThreeGlobeWaterOptions = {
  renderer: WebGPURenderer;
  radius: number;
  heightField: WaterHeightField;
  projection: GlobeProjection;
  environment?: THREE.Texture | null;
  radialSegments?: number;
  angularSegments?: number;
  waveScale?: number;
  detailRange?: number;
  swellSmoothing?: number;
  distantRoughness?: number;
};

export type ThreeWaterCascade = {
  readonly lengthScale: number;
  readonly choppiness: number;
};

export type ThreeWaterWaves = {
  readonly cascades: readonly ThreeWaterCascade[];
  readonly waveScale: number;
  setWaveScale(value: number): void;
  update(renderer: WebGPURenderer, elapsedSeconds: number): void;
  setCascadeTiling(index: number, lengthScale: number): void;
  setCascadeScale(index: number, lengthScale: number): void;
  dispose(): void;
};

export type BuoyancyInput = {
  planarX: number;
  planarZ: number;
  forwardX: number;
  forwardZ: number;
  /** Long-wave attenuation supplied by the host shallow-water/LOD state. */
  effLong: number;
  /** Medium-wave attenuation supplied by the host shallow-water/LOD state. */
  effMedium: number;
  heaveMin: number;
};

export type ThreeWaterBuoyancySampler = {
  readonly poseBuffer: StorageBufferAttribute;
  update(renderer: WebGPURenderer, input: BuoyancyInput): void;
  dispose(): void;
};

export type ThreeGlobeWaterController = {
  readonly mesh: THREE.Mesh;
  readonly material: THREE.Material;
  readonly waves: ThreeWaterWaves;
  init(): Promise<void>;
  update(elapsedSeconds: number, camera: THREE.Camera): void;
  syncPixelScale(camera: THREE.PerspectiveCamera, renderer: WebGPURenderer): void;
  setPatch(patch: WaterPatchField | null): void;
  setAtmosphere(value: number): void;
  setDayLight(value: number): void;
  setSunDirection(direction: THREE.Vector3): void;
  setEnvRotation(rotation: THREE.Matrix3): void;
  setWaveScale(value: number): void;
  readonly detailRange: number;
  setDetailRange(value: number): void;
  readonly swellSmoothing: number;
  setSwellSmoothing(value: number): void;
  readonly distantRoughness: number;
  setDistantRoughness(value: number): void;
  readonly meshResolution: number;
  setMeshResolution(value: number): void;
  dispose(): void;
};
