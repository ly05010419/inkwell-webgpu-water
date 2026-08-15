import { fract, float, max, vec2 } from "three/tsl";
import type Node from "three/src/nodes/core/Node.js";
import type { StorageTexture } from "three/webgpu";

import type { ThreeWaterWavesImpl } from "./spectral-waves";

export type SpectralWaveSampleNodes = {
  readonly longField: Node<"vec4">;
  readonly mediumField: Node<"vec4">;
  readonly shortField: Node<"vec4">;
  readonly longUv: Node<"vec2">;
  readonly mediumUv: Node<"vec2">;
  readonly shortUv: Node<"vec2">;
  readonly longHeight: Node<"float">;
  readonly mediumHeight: Node<"float">;
  readonly height: Node<"float">;
  readonly displacement: Node<"vec2">;
  readonly slope: Node<"vec2">;
  readonly shortSlope: Node<"vec2">;
  readonly crossDerivative: Node<"float">;
  readonly horizontalDerivative: Node<"vec2">;
  readonly jacobian: Node<"float">;
  readonly compression: Node<"float">;
};

type SampleField = (field: StorageTexture, uv: Node<"vec2">, index: number) => Node<"vec4">;

/**
 * Builds the one spectral sampling formula used by both the globe material
 * and the GPU buoyancy compute node. `sampleField` only chooses filtered
 * texture sampling versus integer textureLoad; all scale, harmonic,
 * choppiness and derivative math remains shared.
 */
export function createSpectralWaveSampleNodes(
  waves: ThreeWaterWavesImpl,
  planarX: Node<"float">,
  planarZ: Node<"float">,
  sampleField: SampleField,
  longAttenuation: Node<"float"> | number = 1,
  mediumAttenuation: Node<"float"> | number = 1,
): SpectralWaveSampleNodes {
  const longScale = waves.cascades[0].scaleUniform;
  const mediumScale = waves.cascades[1].scaleUniform;
  const shortScale = waves.cascades[2].scaleUniform;
  const longUv = fract(vec2(planarX.div(longScale), planarZ.div(longScale)).add(0.5));
  const mediumUv = fract(vec2(planarX.div(mediumScale), planarZ.div(mediumScale)).add(0.5));
  const shortUv = fract(vec2(planarX.div(shortScale), planarZ.div(shortScale)).add(0.5));
  const longField = sampleField(waves.getCascadeTexture(0), longUv, 0);
  const mediumField = sampleField(waves.getCascadeTexture(1), mediumUv, 1);
  const shortField = sampleField(waves.getCascadeTexture(2), shortUv, 2);
  const waveScale = waves.getWaveScaleNode();
  const waveScaleSquared = waveScale.mul(waveScale);
  const longFade = typeof longAttenuation === "number" ? float(longAttenuation) : longAttenuation;
  const mediumFade = typeof mediumAttenuation === "number" ? float(mediumAttenuation) : mediumAttenuation;
  const longHeight = longField.b.mul(waveScale).mul(longFade);
  const mediumHeight = mediumField.b.mul(waveScale).mul(mediumFade);
  const height = longHeight
    .add(mediumHeight)
    .add(longHeight.mul(longHeight).sub(waveScaleSquared.mul(0.080)).mul(0.14))
    .add(mediumHeight.mul(mediumHeight).sub(waveScaleSquared.mul(0.030)).mul(0.32));
  const displacement = longField.rg.mul(waveScale).mul(longFade).mul(waves.cascades[0].choppiness)
    .add(mediumField.rg.mul(waveScale).mul(mediumFade).mul(waves.cascades[1].choppiness));
  const slopeNode = longField.rg.mul(waveScale).mul(longFade)
    .mul(float(1).add(longHeight.mul(0.28)))
    .add(mediumField.rg.mul(waveScale).mul(mediumFade)
      .mul(float(1).add(mediumHeight.mul(0.64))));
  const shortSlope = shortField.rg.mul(waveScale);
  const crossDerivative = longField.a.mul(waveScale).mul(longFade).mul(waves.cascades[0].choppiness)
    .add(mediumField.a.mul(waveScale).mul(mediumFade).mul(waves.cascades[1].choppiness));
  const horizontalDerivative = longField.ba.mul(waveScale).mul(longFade).mul(waves.cascades[0].choppiness)
    .add(mediumField.ba.mul(waveScale).mul(mediumFade).mul(waves.cascades[1].choppiness));
  const jacobian = float(1).add(horizontalDerivative.x)
    .mul(float(1).add(horizontalDerivative.y))
    .sub(crossDerivative.mul(crossDerivative));

  return {
    longField,
    mediumField,
    shortField,
    longUv,
    mediumUv,
    shortUv,
    longHeight,
    mediumHeight,
    height,
    displacement,
    slope: slopeNode,
    shortSlope,
    crossDerivative,
    horizontalDerivative,
    jacobian,
    compression: max(0, float(1).sub(jacobian)),
  };
}
