import { Mesh, SphereGeometry, Vector2 } from "three";
import type { WebGPURenderer } from "three/webgpu";
import type * as THREE from "three";

import { ThreeWaterWavesImpl } from "./spectral-waves";
import { createWaterMaterial, type ThreeWaterMaterialState } from "./surface";
import type { ThreeGlobeWaterController, ThreeGlobeWaterOptions, WaterPatchField } from "./types";

export class ThreeGlobeWaterControllerImpl implements ThreeGlobeWaterController {
  readonly mesh: THREE.Mesh;
  readonly waves: ThreeWaterWavesImpl;
  private readonly renderer: WebGPURenderer;
  private readonly radius: number;
  private readonly materialState: ThreeWaterMaterialState;
  private geometry: THREE.SphereGeometry;
  private readonly angularSegments: number;
  private patch: WaterPatchField | null = null;
  private projection: ThreeGlobeWaterOptions["projection"];
  private _detailRange: number;
  private _swellSmoothing: number;
  private _distantRoughness: number;
  private _meshResolution: number;
  private disposed = false;

  constructor(options: ThreeGlobeWaterOptions) {
    this.renderer = options.renderer;
    this.radius = positive(options.radius, "radius");
    this.projection = options.projection;
    this._detailRange = clamp(options.detailRange ?? 1, 0.1, 8);
    this._swellSmoothing = clamp(options.swellSmoothing ?? 1, 0, 3);
    this._distantRoughness = clamp(options.distantRoughness ?? 0, 0, 3);
    this._meshResolution = Math.max(8, Math.floor(options.radialSegments ?? 160));
    this.angularSegments = Math.max(8, Math.floor(options.angularSegments ?? 256));
    this.waves = new ThreeWaterWavesImpl();
    this.waves.setWaveScale(options.waveScale ?? 1);
    this.materialState = createWaterMaterial(
      options.heightField,
      options.environment,
      this.waves,
      this.waves.waveScale,
      this._detailRange,
      this._distantRoughness,
    );
    this.geometry = new SphereGeometry(
      this.radius,
      this.angularSegments,
      this._meshResolution,
    );
    this.mesh = new Mesh(this.geometry, this.materialState.material);
    this.mesh.name = "Three globe water";
    this.mesh.frustumCulled = false;
    this.mesh.userData.waterProjection = this.projection;
  }

  get material(): THREE.Material {
    return this.materialState.material;
  }

  async init() {
    if (this.disposed) throw new Error("ThreeGlobeWaterController has been disposed");
    // Initialization deliberately does not touch renderer.init(), the canvas,
    // the GPU device, or the render loop. Those belong to the host application.
    this.waves.update(this.renderer, 0);
  }

  update(elapsedSeconds: number, camera: THREE.Camera) {
    if (this.disposed) return;
    this.materialState.timeNode.value = Number.isFinite(elapsedSeconds) ? elapsedSeconds : 0;
    this.waves.update(this.renderer, elapsedSeconds);
    this.mesh.userData.waterCamera = camera;
  }

  syncPixelScale(camera: THREE.PerspectiveCamera, renderer: WebGPURenderer) {
    if (this.disposed) return;
    const size = new Vector2();
    renderer.getDrawingBufferSize(size);
    const pixelScale = Math.max(0.25, Math.min(4, size.y / Math.max(camera.projectionMatrix.elements[5], 1e-6) / this.radius));
    this.materialState.detailRangeNode.value = this._detailRange * pixelScale;
  }

  setPatch(patch: WaterPatchField | null) {
    this.patch = patch;
    this.materialState.patch = patch ? { texture: patch.texture } : null;
    if (patch) {
      this.materialState.patchTextureNode.value = patch.texture;
      this.materialState.patchBoundsNode.value.set(
        patch.bounds.lonMin,
        patch.bounds.latMin,
        patch.bounds.lonMax,
        patch.bounds.latMax,
      );
      this.materialState.patchEnabledNode.value = 1;
    } else {
      this.materialState.patchEnabledNode.value = 0;
    }
    this.mesh.userData.waterPatch = patch;
  }

  setAtmosphere(value: number) {
    this.mesh.userData.atmosphere = value;
  }

  setDayLight(value: number) {
    this.mesh.userData.dayLight = value;
  }

  setSunDirection(direction: THREE.Vector3) {
    this.mesh.userData.sunDirection = direction.clone();
  }

  setEnvRotation(rotation: THREE.Matrix3) {
    this.mesh.userData.envRotation = rotation.clone();
  }

  setWaveScale(value: number) {
    this.waves.setWaveScale(value);
    this.materialState.waveScaleNode.value = this.waves.waveScale;
  }

  get detailRange() {
    return this._detailRange;
  }

  setDetailRange(value: number) {
    this._detailRange = clamp(value, 0.1, 8);
    this.materialState.detailRangeNode.value = this._detailRange;
  }

  get swellSmoothing() {
    return this._swellSmoothing;
  }

  setSwellSmoothing(value: number) {
    this._swellSmoothing = clamp(value, 0, 3);
    this.mesh.userData.swellSmoothing = this._swellSmoothing;
  }

  get distantRoughness() {
    return this._distantRoughness;
  }

  setDistantRoughness(value: number) {
    this._distantRoughness = clamp(value, 0, 3);
    this.materialState.roughnessNode.value = this._distantRoughness;
  }

  get meshResolution() {
    return this._meshResolution;
  }

  setMeshResolution(value: number) {
    this._meshResolution = Math.max(8, Math.floor(value));
    this.geometry.dispose();
    this.geometry = new SphereGeometry(
      this.radius,
      this.angularSegments,
      this._meshResolution,
    );
    this.mesh.geometry = this.geometry;
    this.mesh.userData.meshResolution = this._meshResolution;
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.geometry.dispose();
    this.materialState.material.dispose();
    this.waves.dispose();
    // External height/patch/environment textures and the host renderer are
    // intentionally never disposed here.
  }
}

export function createThreeGlobeWater(options: ThreeGlobeWaterOptions): ThreeGlobeWaterController {
  return new ThreeGlobeWaterControllerImpl(options);
}

function positive(value: number, label: string) {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${label} must be a positive finite number`);
  return value;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}
