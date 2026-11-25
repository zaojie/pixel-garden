export const maxGaussianMaterials = 6;
const maxSwatches = 8, rowWidth = 512, materialUnit = 11;
const plans = new WeakMap(), lastAssignments = new WeakMap();
const unit = value => Number.isFinite(value) && value >= 0 && value <= 1;
const packed = color => Number.isInteger(color) && color >= 0 && color <= 0xffffff;
const rgb = color => [color >> 16, color >> 8 & 255, color & 255].map(value => value / 255);

export function visitGaussianMask(mask, count, visit) {
  if (mask?.count !== count || !(Array.isArray(mask.indices) || mask.indices instanceof Uint32Array))
    throw new TypeError('A material needs a painted mask for this imported cloud');
  for (const index of mask.indices) {
    if (!Number.isInteger(index) || index < 0 || index >= count)
      throw new RangeError('Invalid painted material splat index');
    visit(index);
  }
}

function assignments(harmonics, masks) {
  const { count, normals } = harmonics;
  const immutable = masks.every(mask => Object.isFrozen(mask) && Object.isFrozen(mask?.indices));
  const previous = immutable ? lastAssignments.get(harmonics) : undefined;
  if (previous?.masks.length === masks.length && masks.every((mask, i) => mask === previous.masks[i])) return previous;
  const rows = new Float32Array(count * 4);
  let assigned = 0;
  for (const [slot, mask] of masks.entries()) visitGaussianMask(mask, count, i => {
    const at = i * 4;
    if (rows[at + 3]) throw new RangeError('Material masks must contain unique, non-overlapping splat IDs');
    if (normals) for (let axis = 0; axis < 3; axis++) rows[at + axis] = normals[i * 3 + axis] / 127;
    rows[at + 3] = slot + 1;
    assigned++;
  });
  const result = { masks, rows, assigned };
  if (immutable) lastAssignments.set(harmonics, result);
  return result;
}

// The recipe is plain JSON. Compile it for the live imported cloud after edits.
export function compileGaussianMaterials(world, materials) {
  const harmonics = world?.gaussianHarmonics, count = harmonics?.count;
  if (!Number.isSafeInteger(count) || count < 1 || count !== world.count ||
      !Array.isArray(materials) || !materials.length || materials.length > maxGaussianMaterials)
    throw new TypeError('Gaussian materials need an imported live cloud and 1–' + maxGaussianMaterials + ' materials');
  const { normals } = harmonics;
  if (normals !== undefined && (!(normals instanceof Int8Array) || normals.length !== count * 3))
    throw new TypeError('Invalid imported material normal rows');
  const ids = new Set(), palettes = [], ramp = new Float32Array(maxGaussianMaterials * maxSwatches * 3);
  const response = new Float32Array(maxGaussianMaterials * 4), settings = new Float32Array(maxGaussianMaterials * 4);
  const masks = Array.from(materials, (entry, i) => {
    const { id, mask, ramp: colors, captured = .65, detail = .20 } = entry ?? {};
    const { ambient = .28, diffuse = .65, specular = .18, roughness = .6 } = entry?.lighting ?? {};
    if (typeof id !== 'string' || !id.trim() || ids.has(id) ||
        !Array.isArray(colors) || !colors.length || colors.length > maxSwatches || !Array.from(colors).every(packed) ||
        ![captured, detail, ambient, diffuse, specular, roughness].every(unit))
      throw new TypeError('Materials need unique IDs, 1–' + maxSwatches + ' ordered RGB swatches and response values from 0 to 1');
    ids.add(id);
    palettes.push(colors.slice());
    colors.forEach((color, j) => ramp.set(rgb(color), (i * maxSwatches + j) * 3));
    response.set([ambient, diffuse, specular, roughness], i * 4);
    settings.set([colors.length, captured, detail, 0], i * 4);
    return mask;
  });
  const { rows, assigned } = assignments(harmonics, masks);
  const handle = Object.freeze({ count, assigned, materials: Object.freeze([...ids]) });
  plans.set(handle, { harmonics, rows, assigned, ramp, response, settings, palettes, layers: ids.size });
  return handle;
}

export function gaussianMaterialPlan(world, handle) {
  if (!handle) return null;
  const plan = plans.get(handle);
  if (!plan || plan.harmonics !== world.gaussianHarmonics || plan.harmonics.count !== world.count)
    throw new TypeError('Compile Gaussian materials for this imported world');
  return plan.assigned ? plan : null;
}

function materialLight(light = {}) {
  const { direction = [-.65, .85, .55], strength = 1 } = light;
  const length = direction?.length === 3 && Math.hypot(...direction);
  if (!Number.isFinite(length) || length < 1e-6 || !unit(strength))
    throw new RangeError('Gaussian material light needs a finite direction and a strength from 0 to 1');
  return [...Array.from(direction, n => n / length), strength];
}

const rampGLSL = `
uniform vec3 u_gaussianMaterialRamp[${maxGaussianMaterials * maxSwatches}];
uniform vec4 u_gaussianMaterialSettings[${maxGaussianMaterials}];
`;

export const gaussianMaterialVertex = `
${rampGLSL}
uniform highp sampler2D u_gaussianMaterialRows;
uniform vec4 u_gaussianMaterialLight, u_gaussianMaterialResponse[${maxGaussianMaterials}];
flat out int v_gaussianMaterial;

vec3 shadeGaussianMaterial(int index, vec3 ray, vec3 source) {
  vec4 row = texelFetch(u_gaussianMaterialRows, ivec2(index % ${rowWidth}, index / ${rowWidth}), 0);
  v_gaussianMaterial = int(row.w);
  if (v_gaussianMaterial == 0) return source;
  int slot = v_gaussianMaterial - 1;
  vec4 settings = u_gaussianMaterialSettings[slot], response = u_gaussianMaterialResponse[slot];
  float normalLength = length(row.xyz);
  float confidence = clamp(normalLength, 0.0, 1.0);
  vec3 normal = row.xyz / max(normalLength, 0.0001);
  vec3 light = u_gaussianMaterialLight.xyz;
  vec3 halfVector = light - ray / max(length(ray), 0.0001);
  float diffuse = abs(dot(normal, light));
  float sheen = pow(abs(dot(normal, halfVector / max(length(halfVector), 0.0001))), mix(48.0, 4.0, response.w));
  vec3 dc = clamp(gaussianBase(index), 0.0, 1.0);
  float detail = dot(dc, vec3(.30, .55, .15)) - .5;
  float tone = clamp(response.x + (response.y * diffuse + response.z * sheen) *
    u_gaussianMaterialLight.w * confidence + settings.z * detail, 0.0, 1.0);
  float stepAt = tone * (settings.x - 1.0);
  int lo = int(stepAt), hi = min(lo + 1, int(settings.x) - 1), start = slot * ${maxSwatches};
  vec3 base = mix(u_gaussianMaterialRamp[start + lo], u_gaussianMaterialRamp[start + hi], fract(stepAt));
  return clamp(base + settings.y * (source - dc), 0.0, 1.0);
}
`;

export const gaussianMaterialFragment = `
flat in int v_gaussianMaterial;
uniform bool u_gaussianMaterialPass;
`;

// Both buffers share the accumulated alpha. Their RGB channels record the
// visible weight of each material after attenuation by foreground splats.
export const gaussianMaterialCoverage = `
  if (u_gaussianMaterialPass) {
    vec3 slot = vec3(v_gaussianMaterial);
    o = vec4(vec3(equal(slot, vec3(1, 2, 3))) * a, a);
    od = vec4(vec3(equal(slot, vec3(4, 5, 6))) * a, a);
    return;
  }
`;

const paletteFragment = `#version 300 es
precision highp float;
uniform sampler2D u_resolved, u_shaded, u_materialLow, u_materialHigh;
uniform int u_gaussianMaterialLayers;
${rampGLSL}
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  o = texelFetch(u_resolved, p, 0);
  vec4 low = texelFetch(u_materialLow, p, 0);
  vec3 high = texelFetch(u_materialHigh, p, 0).rgb;
  float weight = 0.0;
  int slot = 0;
  for (int i = 0; i < u_gaussianMaterialLayers; i++) {
    float visible = i < 3 ? low[i] : high[i-3];
    if (visible > weight) { weight = visible; slot = i; }
  }
  if (weight < 0.035 || weight < 0.45 * low.a) return;
  vec3 source = texelFetch(u_shaded, p, 0).rgb;
  int start = slot * ${maxSwatches}, count = int(u_gaussianMaterialSettings[slot].x);
  float best = 1e9;
  for (int i = 0; i < count; i++) {
    vec3 swatch = u_gaussianMaterialRamp[start + i];
    vec3 d = swatch - source;
    float error = dot(d * d, vec3(.30, .55, .15));
    if (error < best) { best = error; o.rgb = swatch; }
  }
}
`;

export function createGaussianMaterialBinding(gl, table, maxSize) {
  gl.activeTexture(gl.TEXTURE0 + materialUnit);
  const rows = table(rowWidth, materialUnit, maxSize);
  let uploaded;
  return {
    bind(plan, light) {
      if (plan.rows !== uploaded) {
        rows.upload(plan.rows);
        uploaded = plan.rows;
      } else rows.bind();
      return {
        u_gaussianMaterialRows: materialUnit, u_gaussianMaterialLight: materialLight(light),
        u_gaussianMaterialRamp: plan.ramp, u_gaussianMaterialSettings: plan.settings,
        u_gaussianMaterialResponse: plan.response
      };
    },
    rebind: rows.bind
  };
}

export function createGaussianMaterialPalette({ target, program, textureUnit, fullscreen }) {
  const source = target(), coverage = target(2), shader = program(paletteFragment);
  return {
    source, coverage,
    resize(width, height) {
      source.resize(width, height);
      coverage.resize(width, height);
    },
    finish(plan, shaded, output, width, height) {
      for (const [i, texture] of [source.textures[0], shaded, ...coverage.textures].entries()) textureUnit(i + 12, texture);
      fullscreen(shader, output, width, height, {
        u_resolved: 12, u_shaded: 13, u_materialLow: 14, u_materialHigh: 15,
        u_gaussianMaterialLayers: plan.layers, u_gaussianMaterialRamp: plan.ramp,
        u_gaussianMaterialSettings: plan.settings
      });
    }
  };
}
