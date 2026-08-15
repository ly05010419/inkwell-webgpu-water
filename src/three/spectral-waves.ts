import { DataTexture, FloatType, HalfFloatType, RGBAFormat, RepeatWrapping } from "three";
import { StorageTexture } from "three/webgpu";
import {
  Fn,
  compute,
  cos,
  float,
  instanceIndex,
  int,
  ivec2,
  round,
  select,
  sin,
  textureLoad,
  textureStore,
  uniform,
  vec2,
  vec4,
} from "three/tsl";
import type Node from "three/src/nodes/core/Node.js";
import type UniformNode from "three/src/nodes/core/UniformNode.js";
import type { WebGPURenderer } from "three/webgpu";

import type { ThreeWaterCascade, ThreeWaterWaves } from "./types";

const FIELD_RESOLUTION = 64;
const FIELD_SIZE = FIELD_RESOLUTION * FIELD_RESOLUTION;
const GPU_RESOLUTION = 128;
const GPU_LOG_SIZE = 7;
const GPU_COUNT = GPU_RESOLUTION * GPU_RESOLUTION;

// Keep these values aligned with the Raw WebGPU renderer. The Three adapter
// owns no device: Three's existing WebGPU backend allocates and dispatches the
// storage textures through renderer.compute().
const CASCADE_CONFIG = [
  { lengthScale: 240, choppiness: 1.18, cutoffLow: 0.024, cutoffHigh: 0.36, amplitudeScale: 0.45, secondaryScale: 0.22, seed: 0x51f15e },
  { lengthScale: 64, choppiness: 1.05, cutoffLow: 0.30, cutoffHigh: 1.42, amplitudeScale: 0.45, secondaryScale: 0.08, seed: 0x72a93b },
  { lengthScale: 12, choppiness: 0.40, cutoffLow: 1.22, cutoffHigh: 24.0, amplitudeScale: 0.82, secondaryScale: 0, seed: 0x19ce47 },
] as const;
type GpuCascadeConfig = {
  lengthScale: number;
  choppiness: number;
  cutoffLow: number;
  cutoffHigh: number;
  amplitudeScale: number;
  secondaryScale: number;
  seed: number;
};

const DIRECTIONS = [
  [0.91, 0.41, -0.52, 0.85],
  [0.87, -0.48, -0.78, -0.62],
  [0.63, 0.78, -0.96, 0.28],
] as const;

class MutableCascade implements ThreeWaterCascade {
  readonly scaleUniform: UniformNode<"float", number>;

  constructor(private length = 1, private readonly waveChoppiness = 1) {
    this.scaleUniform = uniform(length, "float");
  }

  get lengthScale() {
    return this.length;
  }

  get choppiness() {
    return this.waveChoppiness;
  }

  setLengthScale(value: number) {
    this.length = clamp(value, 1, 10000);
    this.scaleUniform.value = this.length;
  }
}

type GpuCascade = {
  initial: DataTexture;
  waveData: DataTexture;
  twiddle: DataTexture;
  fields: [[StorageTexture, StorageTexture], [StorageTexture, StorageTexture]];
  computeNodes: ReturnType<typeof compute>[];
};

export type WaveSample = {
  height: number;
  slopeX: number;
  slopeZ: number;
  displacementX: number;
  displacementZ: number;
  crossDerivative: number;
  horizontalDerivativeX: number;
  horizontalDerivativeZ: number;
};

export class ThreeWaterWavesImpl implements ThreeWaterWaves {
  readonly cascades: readonly MutableCascade[];
  private readonly fields: Float32Array[];
  private readonly textures: DataTexture[];
  private readonly gpuCascades: GpuCascade[];
  private readonly timeNode = uniform(0, "float");
  private readonly waveScaleNode = uniform(1, "float");
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
    this.gpuCascades = CASCADE_CONFIG.map((config, index) => this.createGpuCascade(config, index));
    this.rebuildFields();
  }

  get waveScale() {
    return this._waveScale;
  }

  setWaveScale(value: number) {
    this._waveScale = clamp(value, 0, 4);
    this.waveScaleNode.value = this._waveScale;
  }

  getWaveScaleNode() {
    return this.waveScaleNode;
  }

  getCascadeScaleNode(index: number) {
    return this.cascades[index]?.scaleUniform ?? uniform(1, "float");
  }

  setCascadeTiling(index: number, lengthScale: number) {
    this.cascades[index]?.setLengthScale(lengthScale);
    const cascade = this.gpuCascades[index];
    const baseConfig = CASCADE_CONFIG[index];
    if (cascade && baseConfig) {
      const data = buildGpuSpectrumData(GPU_RESOLUTION, { ...baseConfig, lengthScale: this.cascades[index].lengthScale });
      cascade.initial.image.data = data.initialSpectrum;
      cascade.initial.needsUpdate = true;
      cascade.waveData.image.data = data.waveData;
      cascade.waveData.needsUpdate = true;
      cascade.twiddle.image.data = data.twiddle;
      cascade.twiddle.needsUpdate = true;
    }
    this.rebuildFields();
  }

  setCascadeScale(index: number, lengthScale: number) {
    this.setCascadeTiling(index, lengthScale);
  }

  update(_renderer: WebGPURenderer, elapsedSeconds: number) {
    this.elapsedSeconds = Number.isFinite(elapsedSeconds) ? elapsedSeconds : 0;
    this.timeNode.value = this.elapsedSeconds;
    if (_renderer && ( _renderer as unknown as { backend?: { isWebGPUBackend?: boolean } }).backend?.isWebGPUBackend) {
      _renderer.compute(this.gpuCascades.flatMap((cascade) => cascade.computeNodes));
    } else {
      this.rebuildFields();
    }
  }

  sample(planarX: number, planarZ: number): WaveSample {
    return this.sampleWeighted(planarX, planarZ, 1, 1);
  }

  sampleWeighted(planarX: number, planarZ: number, longScale: number, mediumScale: number): WaveSample {
    let height = 0;
    let slopeX = 0;
    let slopeZ = 0;
    let displacementX = 0;
    let displacementZ = 0;
    let crossDerivative = 0;
    let horizontalDerivativeX = 0;
    let horizontalDerivativeZ = 0;

    for (let cascadeIndex = 0; cascadeIndex < CASCADE_CONFIG.length; cascadeIndex += 1) {
      const config = CASCADE_CONFIG[cascadeIndex];
      const length = this.cascades[cascadeIndex].lengthScale;
      const base = config.seed * 13.7;
      const phase = this.elapsedSeconds * (0.52 + cascadeIndex * 0.17);
      const direction = DIRECTIONS[cascadeIndex];
      const k = (Math.PI * 2) / length;
      const attenuation = cascadeIndex === 0 ? longScale : cascadeIndex === 1 ? mediumScale : 1;
      const a = config.amplitudeScale * this._waveScale * attenuation * (cascadeIndex === 2 ? 0.65 : 1);
      const p0 = planarX * direction[0] + planarZ * direction[1];
      const p1 = planarX * direction[2] + planarZ * direction[3];
      const p2 = planarX * (direction[0] - direction[2]) + planarZ * (direction[1] + direction[3]);
      const s0 = Math.sin(p0 * k + phase + base);
      const s1 = Math.sin(p1 * k * 1.73 - phase * 1.21 - base * 0.7);
      const s2 = Math.sin(p2 * k * 0.61 + phase * 0.47 + base * 1.3);
      const localHeight = a * (s0 * 0.57 + s1 * 0.29 + s2 * 0.14);
      height += localHeight;
      if (cascadeIndex < 2) {
        const dx = a * config.choppiness * (direction[0] * (0.57 * Math.cos(p0 * k + phase + base))
          + direction[2] * (0.29 * 1.73 * Math.cos(p1 * k * 1.73 - phase * 1.21 - base * 0.7)));
        const dz = a * config.choppiness * (direction[1] * (0.57 * Math.cos(p0 * k + phase + base))
          + direction[3] * (0.29 * 1.73 * Math.cos(p1 * k * 1.73 - phase * 1.21 - base * 0.7)));
        displacementX += dx;
        displacementZ += dz;
        crossDerivative += a * config.choppiness * k * (direction[0] * direction[1]) * 0.25;
        horizontalDerivativeX += a * config.choppiness * k * direction[0] * 0.22;
        horizontalDerivativeZ += a * config.choppiness * k * direction[1] * 0.22;
      }
      slopeX += a * k * (direction[0] * 0.57 * Math.cos(p0 * k + phase + base)
        + direction[2] * 1.73 * 0.29 * Math.cos(p1 * k * 1.73 - phase * 1.21 - base * 0.7)
        + (direction[0] - direction[2]) * 0.61 * 0.14 * Math.cos(p2 * k * 0.61 + phase * 0.47 + base * 1.3));
      slopeZ += a * k * (direction[1] * 0.57 * Math.cos(p0 * k + phase + base)
        + direction[3] * 1.73 * 0.29 * Math.cos(p1 * k * 1.73 - phase * 1.21 - base * 0.7)
        + (direction[1] + direction[3]) * 0.61 * 0.14 * Math.cos(p2 * k * 0.61 + phase * 0.47 + base * 1.3));
    }

    const longHeight = this.cascadeApproxHeight(planarX, planarZ, 0, this.cascades[0].lengthScale, longScale);
    const mediumHeight = this.cascadeApproxHeight(planarX, planarZ, 1, this.cascades[1].lengthScale, mediumScale);
    height += 0.14 * (longHeight * longHeight - 0.080 * this._waveScale * this._waveScale)
      + 0.32 * (mediumHeight * mediumHeight - 0.030 * this._waveScale * this._waveScale);
    return {
      height,
      slopeX,
      slopeZ,
      displacementX,
      displacementZ,
      crossDerivative,
      horizontalDerivativeX,
      horizontalDerivativeZ,
    };
  }

  getCascadeTexture(index: number) {
    return this.gpuCascades[index]?.fields[0][0] ?? this.textures[index] ?? null;
  }

  getCascadeDerivativeTexture(index: number) {
    return this.gpuCascades[index]?.fields[0][1] ?? this.textures[index] ?? null;
  }

  dispose() {
    for (const texture of this.textures) texture.dispose();
    for (const cascade of this.gpuCascades) {
      cascade.initial.dispose();
      cascade.waveData.dispose();
      cascade.twiddle.dispose();
      cascade.fields.flat(2).forEach((texture) => texture.dispose());
      cascade.computeNodes.forEach((node) => node.dispose());
    }
  }

  private createGpuCascade(config: (typeof CASCADE_CONFIG)[number], index: number): GpuCascade {
    const data = buildGpuSpectrumData(GPU_RESOLUTION, config);
    const initial = floatTexture(data.initialSpectrum, GPU_RESOLUTION, GPU_RESOLUTION, `Three water cascade ${index} initial spectrum`);
    const waveData = floatTexture(data.waveData, GPU_RESOLUTION, GPU_RESOLUTION, `Three water cascade ${index} wave data`);
    const twiddle = floatTexture(data.twiddle, GPU_RESOLUTION, GPU_LOG_SIZE, "Three water Stockham twiddle table");
    const fields = [0, 1].map(() => [0, 1].map(() => storageTexture(GPU_RESOLUTION, GPU_RESOLUTION)) as [StorageTexture, StorageTexture]) as [[StorageTexture, StorageTexture], [StorageTexture, StorageTexture]];
    const computeNodes: ReturnType<typeof compute>[] = [createEvolutionNode(initial, waveData, fields[0], this.timeNode)];
    for (let pass = 0; pass < GPU_LOG_SIZE * 2; pass += 1) {
      const source = pass % 2;
      const destination = 1 - source;
      computeNodes.push(createFftNode(
        fields[source],
        fields[destination],
        twiddle,
        pass < GPU_LOG_SIZE ? 0 : 1,
        pass % GPU_LOG_SIZE,
        pass === GPU_LOG_SIZE * 2 - 1,
      ));
    }
    return { initial, waveData, twiddle, fields, computeNodes };
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
          const height = config.amplitudeScale * this._waveScale
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

  private cascadeApproxHeight(planarX: number, planarZ: number, cascadeIndex: number, lengthScale: number, attenuation: number) {
    const config = CASCADE_CONFIG[cascadeIndex];
    const direction = DIRECTIONS[cascadeIndex];
    const k = (Math.PI * 2) / lengthScale;
    const phase = this.elapsedSeconds * (0.52 + cascadeIndex * 0.17);
    const base = config.seed * 13.7;
    const p0 = planarX * direction[0] + planarZ * direction[1];
    const p1 = planarX * direction[2] + planarZ * direction[3];
    return config.amplitudeScale * this._waveScale * attenuation
      * (Math.sin(p0 * k + phase + base) * 0.57 + Math.sin(p1 * k * 1.73 - phase * 1.21 - base * 0.7) * 0.29);
  }
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, Number.isFinite(value) ? value : min));
}

function floatTexture(data: Float32Array, width: number, height: number, name: string) {
  const texture = new DataTexture(data, width, height, RGBAFormat, FloatType);
  texture.name = name;
  texture.needsUpdate = true;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  return texture;
}

function storageTexture(width: number, height: number) {
  const texture = new StorageTexture(width, height);
  texture.type = HalfFloatType;
  texture.format = RGBAFormat;
  texture.wrapS = RepeatWrapping;
  texture.wrapT = RepeatWrapping;
  return texture;
}

function createEvolutionNode(
  initial: DataTexture,
  waveData: DataTexture,
  fields: [StorageTexture, StorageTexture],
  timeNode: Node<"float">,
) {
  return Fn(() => {
    const index = instanceIndex;
    const x = index.mod(GPU_RESOLUTION);
    const y = index.div(GPU_RESOLUTION);
    const coord = ivec2(int(x), int(y));
    const initialValue = textureLoad(initial, coord);
    const wave = textureLoad(waveData, coord);
    const phase = float(wave.w).mul(timeNode);
    const exponent = vec2(cos(phase), sin(phase));
    const h = vec2(
      initialValue.x.mul(exponent.x).sub(initialValue.y.mul(exponent.y)),
      initialValue.x.mul(exponent.y).add(initialValue.y.mul(exponent.x)),
    ).add(vec2(
      initialValue.z.mul(exponent.x).add(initialValue.w.mul(exponent.y)),
      initialValue.z.mul(exponent.y).sub(initialValue.w.mul(exponent.x)),
    ));
    const ih = vec2(h.y.negate(), h.x);
    const displacementX = ih.mul(wave.x.mul(wave.y));
    const displacementY = h;
    const displacementZ = ih.mul(wave.z.mul(wave.y));
    const displacementXdx = h.mul(wave.x.mul(wave.x).mul(wave.y)).negate();
    const displacementYdx = ih.mul(wave.x);
    const displacementZdx = h.mul(wave.x.mul(wave.z).mul(wave.y)).negate();
    const displacementYdz = ih.mul(wave.z);
    const displacementZdz = h.mul(wave.z.mul(wave.z).mul(wave.y)).negate();
    const dxDz = vec2(displacementX.x.sub(displacementZ.y), displacementX.y.add(displacementZ.x));
    const dyDxz = vec2(displacementY.x.sub(displacementZdx.y), displacementY.y.add(displacementZdx.x));
    const dyxDyz = vec2(displacementYdx.x.sub(displacementYdz.y), displacementYdx.y.add(displacementYdz.x));
    const dxxDzz = vec2(displacementXdx.x.sub(displacementZdz.y), displacementXdx.y.add(displacementZdz.x));
    textureStore(fields[0], coord, vec4(dxDz, dyDxz));
    textureStore(fields[1], coord, vec4(dyxDyz, dxxDzz));
  })().compute(GPU_COUNT, [8, 8]);
}

function createFftNode(
  source: [StorageTexture, StorageTexture],
  destination: [StorageTexture, StorageTexture],
  twiddle: DataTexture,
  axis: number,
  stage: number,
  finalize: boolean,
) {
  return Fn(() => {
    const index = instanceIndex;
    const x = index.mod(GPU_RESOLUTION);
    const y = index.div(GPU_RESOLUTION);
    const transformIndex = axis === 0 ? x : y;
    const table = textureLoad(twiddle, ivec2(int(transformIndex), stage));
    const first = int(round(table.z));
    const second = int(round(table.w));
    const coord0 = axis === 0 ? ivec2(first, int(y)) : ivec2(int(x), first);
    const coord1 = axis === 0 ? ivec2(second, int(y)) : ivec2(int(x), second);
    const value0 = textureLoad(source[0], coord0);
    const value1 = textureLoad(source[0], coord1);
    const value2 = textureLoad(source[1], coord0);
    const value3 = textureLoad(source[1], coord1);
    const tw = vec2(table.x, table.y.negate());
    const mul0 = vec2(value1.x.mul(tw.x).sub(value1.y.mul(tw.y)), value1.x.mul(tw.y).add(value1.y.mul(tw.x)));
    const mul1 = vec2(value1.z.mul(tw.x).sub(value1.w.mul(tw.y)), value1.z.mul(tw.y).add(value1.w.mul(tw.x)));
    const mul2 = vec2(value3.x.mul(tw.x).sub(value3.y.mul(tw.y)), value3.x.mul(tw.y).add(value3.y.mul(tw.x)));
    const mul3 = vec2(value3.z.mul(tw.x).sub(value3.w.mul(tw.y)), value3.z.mul(tw.y).add(value3.w.mul(tw.x)));
    const output0 = vec4(value0.x.add(mul0.x), value0.y.add(mul0.y), value0.z.add(mul1.x), value0.w.add(mul1.y));
    const output1 = vec4(value2.x.add(mul2.x), value2.y.add(mul2.y), value2.z.add(mul3.x), value2.w.add(mul3.y));
    const checker = finalize ? select(x.add(y).mod(2).equal(0), 1, -1) : 1;
    const coord = ivec2(int(x), int(y));
    textureStore(destination[0], coord, output0.mul(checker));
    textureStore(destination[1], coord, output1.mul(checker));
  })().compute(GPU_COUNT, [8, 8]);
}

function buildGpuSpectrumData(size: number, config: GpuCascadeConfig) {
  const initial = new Float32Array(size * size * 4);
  const initialK = new Float32Array(size * size * 2);
  const waveData = new Float32Array(size * size * 4);
  const gravity = 9.81;
  const depth = 54;
  const windSpeed = 11.5;
  const fetch = 120_000;
  const windAngle = -0.48;
  const peakEnhancement = 3.3;
  const swell = 0.38;
  const random = seededRandom(config.seed);
  const deltaK = (Math.PI * 2) / config.lengthScale;
  const alpha = 0.076 * Math.pow(gravity * fetch / (windSpeed * windSpeed), -0.22);
  const peakOmega = 22 * Math.pow(windSpeed * fetch / (gravity * gravity), -0.33);
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const kx = (x - size / 2) * deltaK;
      const kz = (y - size / 2) * deltaK;
      const k = Math.hypot(kx, kz);
      const offset = (y * size + x) * 4;
      if (k < config.cutoffLow || k > config.cutoffHigh || k < 1e-5) {
        waveData[offset + 1] = 1;
        continue;
      }
      const kh = Math.min(k * depth, 20);
      const tanhKh = Math.tanh(kh);
      const omega = Math.sqrt(gravity * k * tanhKh);
      const sechSquared = 1 - tanhKh * tanhKh;
      const frequencyDerivative = gravity * (depth * k * sechSquared + tanhKh) / Math.max(omega * 2, 1e-5);
      const omegaH = omega * Math.sqrt(depth / gravity);
      const tma = omegaH <= 1 ? 0.5 * omegaH * omegaH : omegaH < 2 ? 1 - 0.5 * (2 - omegaH) * (2 - omegaH) : 1;
      const sigma = omega <= peakOmega ? 0.07 : 0.09;
      const peakDistance = (omega - peakOmega) / Math.max(sigma * peakOmega, 1e-5);
      const peakShape = Math.exp(-0.5 * peakDistance * peakDistance);
      const peakRatio = peakOmega / omega;
      const jonswap = tma * alpha * gravity * gravity / Math.pow(omega, 5)
        * Math.exp(-1.25 * Math.pow(peakRatio, 4)) * Math.pow(peakEnhancement, peakShape);
      const theta = wrapAngle(Math.atan2(kz, kx) - windAngle);
      const omegaRatio = omega / peakOmega;
      const spreadPower = ((omega > peakOmega ? 9.77 * Math.pow(omegaRatio, -2.5) : 6.97 * Math.pow(omegaRatio, 5))
        + 16 * Math.tanh(Math.min(omegaRatio, 20)) * swell * swell) * 0.58;
      const focusedDirection = spectrumNormalisationFactor(spreadPower) * Math.pow(Math.abs(Math.cos(theta * 0.5)), 2 * spreadPower);
      const broadDirection = 2 / Math.PI * Math.pow(Math.max(Math.cos(theta), 0), 2);
      const direction = focusedDirection * 0.68 + broadDirection * 0.32;
      const shortWaveFade = Math.exp(-0.00016 * k * k);
      let spectralDensity = jonswap * direction * shortWaveFade;
      if (config.secondaryScale > 0) {
        const swellWindSpeed = 8.4;
        const swellFetch = 310_000;
        const swellPeakOmega = 22 * Math.pow(swellWindSpeed * swellFetch / (gravity * gravity), -0.33);
        const swellAlpha = 0.076 * Math.pow(gravity * swellFetch / (swellWindSpeed * swellWindSpeed), -0.22);
        const swellSigma = omega <= swellPeakOmega ? 0.07 : 0.09;
        const swellPeakDistance = (omega - swellPeakOmega) / Math.max(swellSigma * swellPeakOmega, 1e-5);
        const swellPeakShape = Math.exp(-0.5 * swellPeakDistance * swellPeakDistance);
        const swellPeakRatio = swellPeakOmega / omega;
        const swellSpectrum = tma * swellAlpha * gravity * gravity / Math.pow(omega, 5)
          * Math.exp(-1.25 * Math.pow(swellPeakRatio, 4)) * Math.pow(2.6, swellPeakShape);
        const swellTheta = wrapAngle(Math.atan2(kz, kx) - (windAngle + 0.82));
        const swellRatio = omega / swellPeakOmega;
        const swellSpread = ((omega > swellPeakOmega ? 9.77 * Math.pow(swellRatio, -2.5) : 6.97 * Math.pow(swellRatio, 5)) + 9.0) * 0.72;
        const swellDirection = spectrumNormalisationFactor(swellSpread) * Math.pow(Math.abs(Math.cos(swellTheta * 0.5)), 2 * swellSpread);
        spectralDensity += swellSpectrum * swellDirection * shortWaveFade * config.secondaryScale;
      }
      const amplitude = Math.sqrt(Math.max(0, 2 * spectralDensity * Math.abs(frequencyDerivative) / k * deltaK * deltaK)) * config.amplitudeScale;
      const gaussian = gaussianPair(random);
      const pixel = y * size + x;
      initialK[pixel * 2] = gaussian[0] * amplitude;
      initialK[pixel * 2 + 1] = gaussian[1] * amplitude;
      waveData[offset] = kx;
      waveData[offset + 1] = 1 / k;
      waveData[offset + 2] = kz;
      waveData[offset + 3] = omega;
    }
  }
  for (let y = 0; y < size; y += 1) {
    for (let x = 0; x < size; x += 1) {
      const pixel = y * size + x;
      const mirror = ((size - y) % size) * size + ((size - x) % size);
      const offset = pixel * 4;
      initial[offset] = initialK[pixel * 2];
      initial[offset + 1] = initialK[pixel * 2 + 1];
      initial[offset + 2] = initialK[mirror * 2];
      initial[offset + 3] = -initialK[mirror * 2 + 1];
    }
  }
  const twiddle = new Float32Array(size * GPU_LOG_SIZE * 4);
  for (let stage = 0; stage < GPU_LOG_SIZE; stage += 1) {
    const block = size >> (stage + 1);
    for (let output = 0; output < size / 2; output += 1) {
      const first = (2 * block * Math.floor(output / block) + output % block) % size;
      const angle = (-2 * Math.PI * Math.floor(output / block)) / (size / block);
      const offset = (stage * size + output) * 4;
      twiddle.set([Math.cos(angle), Math.sin(angle), first, first + block], offset);
      twiddle.set([-Math.cos(angle), -Math.sin(angle), first, first + block], (stage * size + output + size / 2) * 4);
    }
  }
  return { initialSpectrum: initial, waveData, twiddle };
}

function seededRandom(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function gaussianPair(random: () => number) {
  const u = Math.max(random(), 1e-7);
  const v = random();
  const radius = Math.sqrt(-2 * Math.log(u));
  const angle = Math.PI * 2 * v;
  return [radius * Math.cos(angle), radius * Math.sin(angle)] as const;
}

function wrapAngle(value: number) {
  return ((value + Math.PI) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2) - Math.PI;
}

function spectrumNormalisationFactor(spread: number) {
  const s2 = spread * spread;
  const s3 = s2 * spread;
  const s4 = s3 * spread;
  return spread < 5
    ? -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * spread + 0.163
    : -4.8e-8 * s4 + 1.07e-5 * s3 - 9.53e-4 * s2 + 5.9e-2 * spread + 0.393;
}
