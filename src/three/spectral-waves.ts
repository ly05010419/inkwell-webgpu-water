import { DataTexture, FloatType, RGBAFormat } from "three";
import type { WebGPURenderer } from "three/webgpu";

import type { ThreeWaterCascade, ThreeWaterWaves } from "./types";

const FIELD_RESOLUTION = 64;
const FIELD_SIZE = FIELD_RESOLUTION * FIELD_RESOLUTION;

// Keep these values aligned with the Raw WebGPU renderer. The Three adapter
// owns no device; these fields are CPU-side spectral snapshots that are
// uploaded by Three's existing renderer when their textures are consumed.
const CASCADE_CONFIG = [
  { lengthScale: 240, choppiness: 1.18, amplitude: 0.75, seed: 0.51 },
  { lengthScale: 64, choppiness: 1.05, amplitude: 0.52, seed: 0.72 },
  { lengthScale: 12, choppiness: 0.40, amplitude: 0.18, seed: 0.19 },
] as const;

const DIRECTIONS = [
  [0.91, 0.41, -0.52, 0.85],
  [0.87, -0.48, -0.78, -0.62],
  [0.63, 0.78, -0.96, 0.28],
] as const;

class MutableCascade implements ThreeWaterCascade {
  constructor(private length = 1, private readonly waveChoppiness = 1) {}

  get lengthScale() {
    return this.length;
  }

  get choppiness() {
    return this.waveChoppiness;
  }

  setLengthScale(value: number) {
    this.length = clamp(value, 1, 10000);
  }
}

export type WaveSample = {
  height: number;
  slopeX: number;
  slopeZ: number;
};

export class ThreeWaterWavesImpl implements ThreeWaterWaves {
  readonly cascades: readonly MutableCascade[];
  private readonly fields: Float32Array[];
  private readonly textures: DataTexture[];
  private _waveScale = 1;
  private elapsedSeconds = 0;

  constructor() {
    this.cascades = CASCADE_CONFIG.map((config) => new MutableCascade(config.lengthScale, config.choppiness));
    this.fields = CASCADE_CONFIG.map(() => new Float32Array(FIELD_SIZE * 4));
    this.textures = this.fields.map((data) => {
      const texture = new DataTexture(data, FIELD_RESOLUTION, FIELD_RESOLUTION, RGBAFormat, FloatType);
      texture.name = "Three water spectral cascade";
      texture.needsUpdate = true;
      return texture;
    });
    this.rebuildFields();
  }

  get waveScale() {
    return this._waveScale;
  }

  setWaveScale(value: number) {
    this._waveScale = clamp(value, 0, 4);
  }

  setCascadeTiling(index: number, lengthScale: number) {
    this.cascades[index]?.setLengthScale(lengthScale);
    this.rebuildFields();
  }

  setCascadeScale(index: number, lengthScale: number) {
    this.setCascadeTiling(index, lengthScale);
  }

  update(_renderer: WebGPURenderer, elapsedSeconds: number) {
    this.elapsedSeconds = Number.isFinite(elapsedSeconds) ? elapsedSeconds : 0;
    this.rebuildFields();
  }

  sample(planarX: number, planarZ: number): WaveSample {
    let height = 0;
    let slopeX = 0;
    let slopeZ = 0;

    for (let cascadeIndex = 0; cascadeIndex < CASCADE_CONFIG.length; cascadeIndex += 1) {
      const config = CASCADE_CONFIG[cascadeIndex];
      const length = this.cascades[cascadeIndex].lengthScale;
      const base = config.seed * 13.7;
      const phase = this.elapsedSeconds * (0.52 + cascadeIndex * 0.17);
      const direction = DIRECTIONS[cascadeIndex];
      const k = (Math.PI * 2) / length;
      const a = config.amplitude * this._waveScale * (cascadeIndex === 2 ? 0.65 : 1);
      const p0 = planarX * direction[0] + planarZ * direction[1];
      const p1 = planarX * direction[2] + planarZ * direction[3];
      const p2 = planarX * (direction[0] - direction[2]) + planarZ * (direction[1] + direction[3]);
      const s0 = Math.sin(p0 * k + phase + base);
      const s1 = Math.sin(p1 * k * 1.73 - phase * 1.21 - base * 0.7);
      const s2 = Math.sin(p2 * k * 0.61 + phase * 0.47 + base * 1.3);
      height += a * (s0 * 0.57 + s1 * 0.29 + s2 * 0.14);
      slopeX += a * k * (direction[0] * 0.57 * Math.cos(p0 * k + phase + base)
        + direction[2] * 1.73 * 0.29 * Math.cos(p1 * k * 1.73 - phase * 1.21 - base * 0.7)
        + (direction[0] - direction[2]) * 0.61 * 0.14 * Math.cos(p2 * k * 0.61 + phase * 0.47 + base * 1.3));
      slopeZ += a * k * (direction[1] * 0.57 * Math.cos(p0 * k + phase + base)
        + direction[3] * 1.73 * 0.29 * Math.cos(p1 * k * 1.73 - phase * 1.21 - base * 0.7)
        + (direction[1] + direction[3]) * 0.61 * 0.14 * Math.cos(p2 * k * 0.61 + phase * 0.47 + base * 1.3));
    }

    return { height, slopeX, slopeZ };
  }

  getCascadeTexture(index: number) {
    return this.textures[index] ?? null;
  }

  dispose() {
    for (const texture of this.textures) texture.dispose();
  }

  private rebuildFields() {
    for (let cascadeIndex = 0; cascadeIndex < this.fields.length; cascadeIndex += 1) {
      const field = this.fields[cascadeIndex];
      const config = CASCADE_CONFIG[cascadeIndex];
      const length = this.cascades[cascadeIndex].lengthScale;
      const phase = this.elapsedSeconds * (0.52 + cascadeIndex * 0.17);
      const direction = DIRECTIONS[cascadeIndex];
      const k = (Math.PI * 2) / length;
      for (let z = 0; z < FIELD_RESOLUTION; z += 1) {
        for (let x = 0; x < FIELD_RESOLUTION; x += 1) {
          const planarX = (x / FIELD_RESOLUTION - 0.5) * length;
          const planarZ = (z / FIELD_RESOLUTION - 0.5) * length;
          const p = planarX * direction[0] + planarZ * direction[1];
          const q = planarX * direction[2] + planarZ * direction[3];
          const height = config.amplitude * this._waveScale
            * (Math.sin(p * k + phase) * 0.7 + Math.sin(q * k * 1.7 - phase) * 0.3);
          const offset = (z * FIELD_RESOLUTION + x) * 4;
          field[offset] = height;
          field[offset + 1] = Math.cos(p * k + phase);
          field[offset + 2] = Math.sin(q * k * 1.7 - phase);
          field[offset + 3] = 1;
        }
      }
      this.textures[cascadeIndex].needsUpdate = true;
    }
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}
