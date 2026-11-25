// Float RGB texels retain the decoded coefficients without repacking each cloud.
export function createGaussianHarmonics(gl, resources, unit, maxSize) {
  let texture, uploaded, width, height, shOffset, colorOffset;
  function clear() {
    if (texture) resources.free(texture);
    texture = uploaded = undefined;
    width = height = undefined;
  }
  function upload(start, values) {
    const stride = width * 3, rows = Math.floor(values.length / stride);
    if (rows) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, start, width, rows, gl.RGB, gl.FLOAT, values);
    if (rows * stride < values.length) {
      const count = (values.length - rows * stride) / 3;
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, start + rows, count, 1, gl.RGB, gl.FLOAT, values, rows * stride);
    }
  }
  const rebind = () => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, texture);
  };
  function bind(harmonics, live) {
    if (!harmonics) {
      clear();
      return;
    }
    if (!live) return;
    const { count, degree, dc, sh, up, linear } = harmonics;
    if (uploaded !== harmonics) {
      const terms = (degree + 1) ** 2 - 1;
      if (!Number.isInteger(degree) || degree < 0 || degree > 4 || !Number.isInteger(count) || count < 1 ||
          !(dc instanceof Float32Array) || dc.length !== count * 3 ||
          !(sh instanceof Float32Array) || sh.length !== count * terms * 3 || !['y', 'z'].includes(up) ||
          linear && (!(linear instanceof Uint8Array) || linear.length !== count))
        throw new TypeError('Invalid live Gaussian harmonic rows');
      const nextWidth = Math.min(maxSize, Math.max(1024, Math.ceil(count * (terms + 1 + Number(!!linear) / 3) / (maxSize - 2))));
      const dcRows = Math.ceil(count / nextWidth), harmonicRows = dcRows + Math.ceil(count * terms / nextWidth);
      const nextHeight = harmonicRows + (linear ? Math.ceil(count / (3*nextWidth)) : 0);
      if (nextHeight > maxSize) throw new RangeError('Gaussian harmonics exceed this device’s WebGL texture size');
      gl.activeTexture(gl.TEXTURE0 + unit);
      texture ??= resources.texture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      if (nextWidth !== width || nextHeight !== height) {
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, nextWidth, nextHeight, 0, gl.RGB, gl.FLOAT, null);
        width = nextWidth;
        height = nextHeight;
      }
      upload(0, dc);
      if (terms) upload(dcRows, sh);
      if (linear) {
        const values = new Float32Array(Math.ceil(count/3)*3);
        values.set(linear);
        upload(harmonicRows, values);
      }
      colorOffset = linear ? harmonicRows * width : -1;
      if (gl.getError() !== gl.NO_ERROR) {
        clear();
        throw new Error('Could not upload live Gaussian harmonics on this WebGL device');
      }
      shOffset = dcRows * width;
      uploaded = harmonics;
    } else rebind();
    return { u_gaussianSH: unit, u_gaussianWidth: width, u_gaussianOffset: shOffset,
      u_gaussianCount: count, u_gaussianColorOffset: colorOffset, u_gaussianZUp: up === 'z' };
  }
  return { bind, rebind };
}

// Derivatives of the Legendre polynomials through degree four. The real
// and imaginary parts of (x + iy)^m provide the angular factors without trig.
const legendreDerivatives = [
  ['z', '1.0'],
  ['(3.0*z*z-1.0)*0.5', '3.0*z', '3.0'],
  ['(5.0*z*z-3.0)*z*0.5', '(15.0*z*z-3.0)*0.5', '15.0*z', '15.0'],
  ['((35.0*z*z-30.0)*z*z+3.0)*0.125', '(35.0*z*z-15.0)*z*0.5',
    '(105.0*z*z-15.0)*0.5', '105.0*z', '105.0']
];
const angular = [
  ['1.0', '0.0'], ['x', 'y'], ['x*x-y*y', '2.0*x*y'],
  ['x*(x*x-3.0*y*y)', 'y*(3.0*x*x-y*y)'],
  ['(x*x-y*y)*(x*x-y*y)-4.0*x*x*y*y', '4.0*x*y*(x*x-y*y)']
];

export function gaussianHarmonicGLSL(degree) {
  if (!Number.isInteger(degree) || degree < 0 || degree > 4) throw new RangeError('Invalid Gaussian SH degree');
  const terms = (degree + 1) ** 2 - 1;
  const lines = [];
  for (let l = 1; l <= degree; l++) {
    for (let m = -l; m <= l; m++) {
      const a = Math.abs(m);
      let normalization = (2 * l + 1) / (4 * Math.PI) * (a ? 2 : 1);
      for (let i = l - a + 1; i <= l + a; i++) normalization /= i;
      const sign = a % 2 ? -1 : 1;
      const weight = (sign * Math.sqrt(normalization)).toPrecision(16);
      const term = l * l + l + m - 1;
      lines.push(`  color += gaussianCoefficient(first + ${term}) * (${weight} * (${legendreDerivatives[l - 1][a]}) * (${angular[a][+(m < 0)]}));`);
    }
  }
  return `
uniform highp sampler2D u_gaussianSH;
uniform int u_gaussianWidth, u_gaussianOffset, u_gaussianCount, u_gaussianColorOffset;
uniform bool u_gaussianZUp;
vec3 gaussianCoefficient(int offset) {
  return texelFetch(u_gaussianSH, ivec2(offset % u_gaussianWidth, offset / u_gaussianWidth), 0).rgb;
}
vec3 gaussianBase(int index) {
  return vec3(0.5) + 0.28209479177387814 * gaussianCoefficient(index);
}
vec3 gaussianRgb(int index, vec3 ray) {
  vec3 color = gaussianBase(index);
${degree ? `  vec3 d = normalize(u_gaussianZUp ? vec3(ray.x, -ray.z, ray.y) : ray);
  float x = d.x, y = d.y, z = d.z;
  int first = u_gaussianOffset + index * ${terms};
${lines.join('\n')}` : ''}
  color = clamp(color, 0.0, 1.0);
  if (u_gaussianColorOffset >= 0 && gaussianCoefficient(u_gaussianColorOffset + index / 3)[index % 3] > 0.5) {
    color = mix(1.055 * pow(color, vec3(1.0 / 2.4)) - 0.055, 12.92 * color, lessThanEqual(color, vec3(0.0031308)));
  }
  return color;
}
`;
}
