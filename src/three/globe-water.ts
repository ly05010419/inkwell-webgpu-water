import { BufferGeometry, Float32BufferAttribute, Matrix4, Mesh, Vector2, Vector3 } from "three";
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
  private geometry: THREE.BufferGeometry;
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
    this._distantRoughness = clamp(options.distantRoughness ?? 1, 0, 3);
    this._meshResolution = Math.max(8, Math.floor(options.radialSegments ?? 256));
    this.angularSegments = Math.max(8, Math.floor(options.angularSegments ?? 1024));
    this.waves = new ThreeWaterWavesImpl();
    this.waves.setWaveScale(options.waveScale ?? 1);
    this.materialState = createWaterMaterial(
      options.heightField,
      options.environment,
      this.waves,
      this.waves.waveScale,
      this._detailRange,
      this._distantRoughness,
      this.radius,
      this.projection,
    );
    this.geometry = createSphericalWindowGeometry(this.radius, this._meshResolution, this.angularSegments);
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
  }

  update(elapsedSeconds: number, camera: THREE.Camera) {
    if (this.disposed) return;
    this.materialState.timeNode.value = Number.isFinite(elapsedSeconds) ? elapsedSeconds : 0;
    this.mesh.userData.waterCamera = camera;
    const cameraPosition = new Vector3().setFromMatrixPosition(camera.matrixWorld);
    this.materialState.cameraPositionNode.value.copy(cameraPosition);
    this.setWindowFrame(cameraPosition);
  }

  syncPixelScale(camera: THREE.PerspectiveCamera, renderer: WebGPURenderer) {
    if (this.disposed) return;
    const size = new Vector2();
    renderer.getDrawingBufferSize(size);
    const focal = Math.max(camera.projectionMatrix.elements[5], 1e-6);
    // Raw Tethys uses pixelWorldSize = distance * 2 / focal / viewportHeight.
    // Keep that scale in a separate uniform: detailRange remains the host's
    // logical distance multiplier and never leaks into seabed height.
    this.materialState.pixelWorldScaleNode.value = 2 / focal / Math.max(size.y, 1);
    this.materialState.detailRangeNode.value = this._detailRange;
  }

  setPatch(patch: WaterPatchField | null) {
    this.patch = patch;
    this.materialState.patch = patch ? { texture: patch.texture } : null;
    if (patch) {
      this.materialState.patchTextureNode.value = patch.texture;
      const degreesToRadians = Math.PI / 180;
      this.materialState.patchBoundsNode.value.set(
        patch.bounds.lonMin * degreesToRadians,
        patch.bounds.latMin * degreesToRadians,
        patch.bounds.lonMax * degreesToRadians,
        patch.bounds.latMax * degreesToRadians,
      );
      this.materialState.patchEnabledNode.value = 1;
    } else {
      this.materialState.patchEnabledNode.value = 0;
    }
    this.mesh.userData.waterPatch = patch;
  }

  setAtmosphere(value: number) {
    this.materialState.atmosphereNode.value = clamp(value, 0, 1);
  }

  setDayLight(value: number) {
    this.materialState.dayLightNode.value = clamp(value, 0, 1);
  }

  setSunDirection(direction: THREE.Vector3) {
    this.materialState.sunDirectionNode.value.copy(direction).normalize();
  }

  setEnvRotation(rotation: THREE.Matrix3) {
    this.materialState.envRotationNode.value.copy(rotation);
    if ("envMapRotation" in this.materialState.material) {
      const matrix = new Matrix4().setFromMatrix3(rotation);
      (this.materialState.material as unknown as THREE.MeshPhysicalMaterial).envMapRotation.setFromRotationMatrix(matrix);
    }
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
    this.materialState.swellSmoothingNode.value = this._swellSmoothing;
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
    this.geometry = createSphericalWindowGeometry(this.radius, this._meshResolution, this.angularSegments);
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

  private setWindowFrame(cameraPosition: THREE.Vector3) {
    const normal = cameraPosition.clone().normalize();
    const north = new Vector3(0, 1, 0);
    const east = north.clone().cross(normal);
    if (east.lengthSq() < 1e-8) east.set(1, 0, 0);
    east.normalize();
    const south = east.clone().cross(normal).normalize();
    const distance = Math.max(cameraPosition.length(), this.radius * 1.001);
    const horizon = Math.acos(Math.min(0.999, this.radius / distance));
    const arc = Math.min(Math.PI * 0.98, Math.max(0.22, horizon * 1.22));
    this.materialState.windowNormalNode.value.copy(normal);
    this.materialState.windowEastNode.value.copy(east);
    this.materialState.windowSouthNode.value.copy(south);
    this.materialState.windowArcNode.value = arc;
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

function createSphericalWindowGeometry(radius: number, radialSegments: number, angularSegments: number) {
  const geometry = new BufferGeometry();
  const positions = new Float32Array((radialSegments + 1) * (angularSegments + 1) * 3);
  const normals = new Float32Array(positions.length);
  const uvs = new Float32Array((radialSegments + 1) * (angularSegments + 1) * 2);
  const indices: number[] = [];
  for (let row = 0; row < radialSegments; row += 1) {
    for (let column = 0; column < angularSegments; column += 1) {
      const a = row * (angularSegments + 1) + column;
      const b = a + 1;
      const c = a + angularSegments + 1;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  geometry.setIndex(indices);
  for (let row = 0; row <= radialSegments; row += 1) {
    for (let column = 0; column <= angularSegments; column += 1) {
      const index = row * (angularSegments + 1) + column;
      positions[index * 3] = row / radialSegments;
      positions[index * 3 + 1] = column / angularSegments;
      positions[index * 3 + 2] = 0;
      normals[index * 3 + 2] = 1;
      uvs[index * 2] = row / radialSegments;
      uvs[index * 2 + 1] = column / angularSegments;
    }
  }
  // Float32BufferAttribute copies the supplied array in Three.js r184. Fill
  // the typed arrays before creating attributes, otherwise every vertex stays
  // at the origin and the adapter appears to render a blank scene.
  geometry.setAttribute("position", new Float32BufferAttribute(positions, 3));
  geometry.setAttribute("normal", new Float32BufferAttribute(normals, 3));
  geometry.setAttribute("uv", new Float32BufferAttribute(uvs, 2));
  geometry.computeBoundingSphere();
  return geometry;
}
