import { StorageBufferAttribute } from "three/webgpu";
import type { WebGPURenderer } from "three/webgpu";
import {
  Fn,
  floor,
  fract,
  int,
  ivec2,
  max,
  storage,
  textureLoad,
  uniform,
  vec4,
} from "three/tsl";
import type Node from "three/src/nodes/core/Node.js";

import type { ThreeWaterBuoyancySampler, BuoyancyInput, ThreeWaterWaves } from "./types";
import { ThreeWaterWavesImpl } from "./spectral-waves";

export class ThreeWaterBuoyancySamplerImpl implements ThreeWaterBuoyancySampler {
  readonly poseBuffer = new StorageBufferAttribute(new Float32Array(4), 4);
  private readonly planarXNode = uniform(0, "float");
  private readonly planarZNode = uniform(0, "float");
  private readonly forwardXNode = uniform(0, "float");
  private readonly forwardZNode = uniform(1, "float");
  private readonly effLongNode = uniform(1, "float");
  private readonly effMediumNode = uniform(1, "float");
  private readonly heaveMinNode = uniform(0, "float");
  private readonly computeNode;

  constructor(private readonly waves: ThreeWaterWavesImpl) {
    const pose = storage(this.poseBuffer, "vec4", 1);
    const longField = waves.getCascadeTexture(0);
    const mediumField = waves.getCascadeTexture(1);
    this.computeNode = Fn(() => {
      const forwardLength = max(this.forwardXNode.mul(this.forwardXNode).add(this.forwardZNode.mul(this.forwardZNode)).sqrt(), 0.001);
      const forwardX = this.forwardXNode.div(forwardLength);
      const forwardZ = this.forwardZNode.div(forwardLength);
      const sideX = forwardZ.negate();
      const sideZ = forwardX;
      const longProbe = 9;
      const mediumProbe = 3.2;
      const sample = (x: Node<"float">, z: Node<"float">) => {
        const uvX = fract(x.div(240).add(0.5));
        const uvZ = fract(z.div(240).add(0.5));
        const coord = ivec2(int(floor(uvX.mul(128))), int(floor(uvZ.mul(128))));
        const longWave = textureLoad(longField, coord).b.mul(this.effLongNode).mul(this.waves.getWaveScaleNode());
        const mediumWave = textureLoad(mediumField, coord).b.mul(this.effMediumNode).mul(this.waves.getWaveScaleNode());
        const height = longWave.add(mediumWave);
        return height.add(height.mul(height).mul(0.14));
      };
      const cx = this.planarXNode;
      const cz = this.planarZNode;
      const bow = sample(cx.add(forwardX.mul(longProbe)), cz.add(forwardZ.mul(longProbe)));
      const stern = sample(cx.sub(forwardX.mul(longProbe)), cz.sub(forwardZ.mul(longProbe)));
      const port = sample(cx.add(sideX.mul(mediumProbe)), cz.add(sideZ.mul(mediumProbe)));
      const starboard = sample(cx.sub(sideX.mul(mediumProbe)), cz.sub(sideZ.mul(mediumProbe)));
      const centre = sample(cx, cz);
      const heave = max(this.heaveMinNode, centre.add(bow.add(stern).add(port).add(starboard).div(4)));
      const poseValue = vec4(
        heave,
        bow.sub(stern).div(18),
        port.sub(starboard).div(6.4),
        forwardX.atan(forwardZ),
      );
      pose.element(0).assign(poseValue);
    })().compute(1, [1]);
  }

  update(_renderer: WebGPURenderer, input: BuoyancyInput) {
    this.planarXNode.value = input.planarX;
    this.planarZNode.value = input.planarZ;
    this.forwardXNode.value = input.forwardX;
    this.forwardZNode.value = input.forwardZ;
    this.effLongNode.value = clamp(input.effLong, 0, 1);
    this.effMediumNode.value = clamp(input.effMedium, 0, 1);
    this.heaveMinNode.value = input.heaveMin;
    if (_renderer && (_renderer as unknown as { backend?: { isWebGPUBackend?: boolean } }).backend?.isWebGPUBackend) {
      _renderer.compute(this.computeNode);
      return;
    }
    const forwardLength = Math.hypot(input.forwardX, input.forwardZ) || 1;
    const forwardX = input.forwardX / forwardLength;
    const forwardZ = input.forwardZ / forwardLength;
    const sideX = -forwardZ;
    const sideZ = forwardX;
    const long = 9;
    const medium = 3.2;
    const attenuationLong = clamp(input.effLong, 0, 1);
    const attenuationMedium = clamp(input.effMedium, 0, 1);
    const sample = (x: number, z: number) => {
      const value = this.waves.sampleWeighted(x, z, attenuationLong, attenuationMedium).height;
      return value + value * value * 0.14;
    };
    const bow = sample(input.planarX + forwardX * long, input.planarZ + forwardZ * long);
    const stern = sample(input.planarX - forwardX * long, input.planarZ - forwardZ * long);
    const port = sample(input.planarX + sideX * medium, input.planarZ + sideZ * medium);
    const starboard = sample(input.planarX - sideX * medium, input.planarZ - sideZ * medium);
    const centre = sample(input.planarX, input.planarZ);
    const pose = this.poseBuffer.array as Float32Array;
    pose[0] = Math.max(input.heaveMin, (centre + bow + stern + port + starboard) * 0.2);
    pose[1] = (bow - stern) / 18;
    pose[2] = (port - starboard) / 6.4;
    pose[3] = Math.atan2(forwardX, forwardZ);
    this.poseBuffer.needsUpdate = true;
  }

  dispose() {
    this.computeNode.dispose();
    // StorageBufferAttribute has no renderer ownership; the caller owns any
    // object that consumes poseBuffer.
  }
}

function clamp(value: number, minValue: number, maxValue: number) {
  return Math.min(maxValue, Math.max(minValue, Number.isFinite(value) ? value : minValue));
}

export function createBuoyancySampler(waves: ThreeWaterWaves): ThreeWaterBuoyancySampler {
  if (!(waves instanceof ThreeWaterWavesImpl)) {
    throw new TypeError("createBuoyancySampler expects waves from createThreeGlobeWater");
  }
  return new ThreeWaterBuoyancySamplerImpl(waves);
}
