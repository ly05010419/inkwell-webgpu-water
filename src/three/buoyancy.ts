import { StorageBufferAttribute } from "three/webgpu";
import type { WebGPURenderer } from "three/webgpu";
import {
  Fn,
  floor,
  int,
  ivec2,
  max,
  storage,
  textureLoad,
  uniform,
  vec4,
  atan,
} from "three/tsl";
import type Node from "three/src/nodes/core/Node.js";

import { createSpectralWaveSampleNodes } from "./wave-sampling";
import type { ThreeWaterBuoyancySampler, BuoyancyInput, ThreeWaterWaves } from "./types";
import { ThreeWaterWavesImpl } from "./spectral-waves";

const GPU_RESOLUTION = 128;
const HALF_LENGTH = 9;
const HALF_BEAM = 3.2;

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
    const sampleField = (field: Parameters<typeof textureLoad>[0], uv: Node<"vec2">) => {
      const coord = ivec2(
        int(floor(uv.x.mul(GPU_RESOLUTION))),
        int(floor(uv.y.mul(GPU_RESOLUTION))),
      );
      return textureLoad(field, coord);
    };
    this.computeNode = Fn(() => {
      const forwardLength = max(this.forwardXNode.mul(this.forwardXNode).add(this.forwardZNode.mul(this.forwardZNode)).sqrt(), 0.001);
      const forwardX = this.forwardXNode.div(forwardLength);
      const forwardZ = this.forwardZNode.div(forwardLength);
      const starboardX = forwardZ.negate();
      const starboardZ = forwardX;
      const sample = (x: Node<"float">, z: Node<"float">) => createSpectralWaveSampleNodes(
        this.waves,
        x,
        z,
        (field, uv) => sampleField(field, uv),
        this.effLongNode,
        this.effMediumNode,
      ).height;
      const cx = this.planarXNode;
      const cz = this.planarZNode;
      const bow = sample(cx.add(forwardX.mul(HALF_LENGTH)), cz.add(forwardZ.mul(HALF_LENGTH)));
      const stern = sample(cx.sub(forwardX.mul(HALF_LENGTH)), cz.sub(forwardZ.mul(HALF_LENGTH)));
      const starboard = sample(cx.add(starboardX.mul(HALF_BEAM)), cz.add(starboardZ.mul(HALF_BEAM)));
      const port = sample(cx.sub(starboardX.mul(HALF_BEAM)), cz.sub(starboardZ.mul(HALF_BEAM)));
      const centre = sample(cx, cz);
      const heave = max(this.heaveMinNode, centre.mul(2).add(bow).add(stern).add(port).add(starboard).div(6));
      const trim = atan(bow.sub(stern), 2 * HALF_LENGTH).mul(0.55);
      const heel = atan(starboard.sub(port), 2 * HALF_BEAM).mul(0.45);
      const poseValue = vec4(
        heave,
        trim,
        heel,
        atan(forwardX, forwardZ),
      );
      pose.element(0).assign(poseValue);
    })().compute(1, [1]);
  }

  update(renderer: WebGPURenderer, input: BuoyancyInput) {
    this.planarXNode.value = input.planarX;
    this.planarZNode.value = input.planarZ;
    this.forwardXNode.value = input.forwardX;
    this.forwardZNode.value = input.forwardZ;
    this.effLongNode.value = clamp(input.effLong, 0, 1);
    this.effMediumNode.value = clamp(input.effMedium, 0, 1);
    this.heaveMinNode.value = input.heaveMin;
    if (renderer && (renderer as unknown as { backend?: { isWebGPUBackend?: boolean } }).backend?.isWebGPUBackend) {
      renderer.compute(this.computeNode);
      return;
    }

    const forwardLength = Math.hypot(input.forwardX, input.forwardZ) || 1;
    const forwardX = input.forwardX / forwardLength;
    const forwardZ = input.forwardZ / forwardLength;
    const starboardX = -forwardZ;
    const starboardZ = forwardX;
    const attenuationLong = clamp(input.effLong, 0, 1);
    const attenuationMedium = clamp(input.effMedium, 0, 1);
    const sample = (x: number, z: number) => this.waves.sampleWeighted(x, z, attenuationLong, attenuationMedium).height;
    const bow = sample(input.planarX + forwardX * HALF_LENGTH, input.planarZ + forwardZ * HALF_LENGTH);
    const stern = sample(input.planarX - forwardX * HALF_LENGTH, input.planarZ - forwardZ * HALF_LENGTH);
    const starboard = sample(input.planarX + starboardX * HALF_BEAM, input.planarZ + starboardZ * HALF_BEAM);
    const port = sample(input.planarX - starboardX * HALF_BEAM, input.planarZ - starboardZ * HALF_BEAM);
    const centre = sample(input.planarX, input.planarZ);
    const pose = this.poseBuffer.array as Float32Array;
    pose[0] = Math.max(input.heaveMin, (centre * 2 + bow + stern + port + starboard) / 6);
    pose[1] = Math.atan2(bow - stern, 2 * HALF_LENGTH) * 0.55;
    pose[2] = Math.atan2(starboard - port, 2 * HALF_BEAM) * 0.45;
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
