import { StorageBufferAttribute } from "three/webgpu";
import type { WebGPURenderer } from "three/webgpu";

import type { ThreeWaterBuoyancySampler, BuoyancyInput, ThreeWaterWaves } from "./types";
import { ThreeWaterWavesImpl } from "./spectral-waves";

export class ThreeWaterBuoyancySamplerImpl implements ThreeWaterBuoyancySampler {
  readonly poseBuffer = new StorageBufferAttribute(new Float32Array(4), 4);

  constructor(private readonly waves: ThreeWaterWavesImpl) {}

  update(_renderer: WebGPURenderer, input: BuoyancyInput) {
    const forwardLength = Math.hypot(input.forwardX, input.forwardZ) || 1;
    const forwardX = input.forwardX / forwardLength;
    const forwardZ = input.forwardZ / forwardLength;
    const sideX = -forwardZ;
    const sideZ = forwardX;
    const long = Math.max(0, input.effLong);
    const medium = Math.max(0, input.effMedium);
    const probes = [
      this.waves.sample(input.planarX + forwardX * long, input.planarZ + forwardZ * long).height,
      this.waves.sample(input.planarX - forwardX * long, input.planarZ - forwardZ * long).height,
      this.waves.sample(input.planarX + sideX * medium, input.planarZ + sideZ * medium).height,
      this.waves.sample(input.planarX - sideX * medium, input.planarZ - sideZ * medium).height,
    ];
    const bow = (probes[0] + probes[2]) * 0.5;
    const stern = (probes[1] + probes[3]) * 0.5;
    const port = (probes[0] + probes[1]) * 0.5;
    const starboard = (probes[2] + probes[3]) * 0.5;
    const pose = this.poseBuffer.array as Float32Array;
    pose[0] = Math.max(input.heaveMin, (probes[0] + probes[1] + probes[2] + probes[3]) * 0.25);
    pose[1] = Math.atan2(bow - stern, Math.max(long * 2, 0.001));
    pose[2] = Math.atan2(port - starboard, Math.max(medium * 2, 0.001));
    pose[3] = Math.atan2(forwardX, forwardZ);
    this.poseBuffer.needsUpdate = true;
  }

  dispose() {
    // BufferAttribute has no GPU ownership; the caller owns any object that
    // consumes poseBuffer. Clearing references is intentionally unnecessary.
  }
}

export function createBuoyancySampler(waves: ThreeWaterWaves): ThreeWaterBuoyancySampler {
  if (!(waves instanceof ThreeWaterWavesImpl)) {
    throw new TypeError("createBuoyancySampler expects waves from createThreeGlobeWater");
  }
  return new ThreeWaterBuoyancySamplerImpl(waves);
}
