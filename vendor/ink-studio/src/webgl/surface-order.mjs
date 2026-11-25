const rowWidth = 512, rowUnit = 3, previousUnit = 15;
const maxRows = 32767;
const integer = (n, lo, hi) => Number.isSafeInteger(n) && n >= lo && n <= hi;

export function surfaceOrderSettings(mode, value = {}) {
  if (mode === undefined || mode === 'center') return null;
  if (mode !== 'surface' || !value || typeof value !== 'object' || Array.isArray(value) ||
      Object.keys(value).some(key => !['region', 'layers', 'maxSplats', 'maxPixels'].includes(key)))
    throw new TypeError('Splat order needs center or surface with optional surface-order settings');
  const { region = [0, 0, 1, 1], layers = 4, maxSplats = 8192, maxPixels = 98304 } = value;
  if (!Array.isArray(region) || region.length !== 4 || !Array.from(region).every(n => Number.isFinite(n) && n >= 0 && n <= 1) ||
      !region[2] || !region[3] || region[0] + region[2] > 1 || region[1] + region[3] > 1 ||
      !integer(layers, 2, 8) || !integer(maxSplats, 2, maxRows) || !integer(maxPixels, 1, 524288))
    throw new RangeError('Surface order needs a normalized region, 2–8 layers and bounded splat and pixel budgets');
  return { region, layers, maxSplats, maxPixels };
}

export function surfaceOrderRegion(region, width, height) {
  const [x, y, w, h] = region;
  const left = Math.floor(width * x), right = Math.ceil(width * (x + w));
  const bottom = height - Math.ceil(height * (y + h)), top = height - Math.floor(height * y);
  return [left, bottom, right - left, top - bottom];
}

function normalRows(world) {
  const normals = world.renderMetadata?.normals, baked = world.surfaceData, pixel = world.pix;
  const imported = world.gaussianHarmonics?.normals;
  const rows = new Float32Array(world.count * 4);
  let found = false;
  for (let i = 0; i < world.count; i++) {
    const n = i * 3, b = i * 8, at = i * 4;
    let x = normals?.[n] ?? 0, y = normals?.[n + 1] ?? 0, z = normals?.[n + 2] ?? 0;
    if (pixel?.[b]) { x = pixel[b + 3]; y = pixel[b + 4]; z = pixel[b + 5]; }
    if (!x && !y && !z && baked?.[b]) { x = baked[b + 1]; y = baked[b + 2]; z = baked[b + 3]; }
    if (!x && !y && !z && imported) {
      x = imported[n] / 127; y = imported[n + 1] / 127; z = imported[n + 2] / 127;
      if (x * x + y * y + z * z < .0225) continue;
    }
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && (x || y || z)) {
      rows[at] = x; rows[at + 1] = y; rows[at + 2] = z;
      found = true;
    }
  }
  if (!found) return null;
  for (const i of world.back ?? []) if (i >= 0 && i < world.count) rows[i * 4 + 3] = 1;
  return rows;
}

export const surfaceOrderVertex = `
uniform bool u_surfacePeel;
uniform highp sampler2D u_surfaceRows;
flat out vec4 v_surfacePlane;
flat out vec2 v_surfaceAxial;
flat out uint v_surfaceRank;
void surfaceOrderVertex(int row, mat3 frame, mat3 view, mat3 cov, vec3 center, bool behind) {
  vec4 data = texelFetch(u_surfaceRows, ivec2(row % ${rowWidth}, row / ${rowWidth}), 0);
  vec3 normal = data.xyz;
  v_surfacePlane = vec4(0.0);
  if (dot(normal, normal) > 0.0 && abs(determinant(frame)) > 1e-8) {
    normal = normalize(view * transpose(inverse(frame)) * normal);
    v_surfacePlane = vec4(normal, dot(normal, center));
  }
  v_surfaceAxial = vec2(-center.z, 3.0 * sqrt(max(0.0, cov[0][0] + cov[1][1] + cov[2][2])));
  // Signed only here. A background row is behind every ordinary sample.
  if (data.w > 0.0 || behind) v_surfaceAxial.x = -v_surfaceAxial.x;
  v_surfaceRank = uint(gl_InstanceID) + 1u;
}`;

const decodeKey = `
uvec2 surfaceKey(vec4 rgba) {
  uvec4 bytes = uvec4(round(rgba * 255.0));
  return uvec2(bytes.r | (bytes.g << 8), bytes.b | (bytes.a << 8));
}
`;

export const surfaceOrderFragment = `
uniform bool u_surfacePeel, u_surfaceNext, u_surfaceOrtho;
uniform highp sampler2D u_surfacePrevious;
uniform vec4 u_surfaceProjection;
uniform vec3 u_surfaceLens;
uniform vec2 u_surfaceRange;
uniform float u_surfaceNear;
flat in vec4 v_surfacePlane;
flat in vec2 v_surfaceAxial;
flat in uint v_surfaceRank;
${decodeKey}
vec4 surfacePeelKey(out float axial) {
  axial = abs(v_surfaceAxial.x);
  vec2 pixel = (gl_FragCoord.xy - u_surfaceLens.xy * 0.5) / u_surfaceLens.z;
  vec3 origin = u_surfaceOrtho ? vec3(pixel, 0.0) : vec3(0.0);
  vec3 ray = u_surfaceOrtho ? vec3(0.0, 0.0, -1.0) : vec3(pixel, -1.0);
  float facing = dot(v_surfacePlane.xyz, ray);
  if (abs(facing) >= .025 * length(ray)) {
    float hit = (v_surfacePlane.w - dot(v_surfacePlane.xyz, origin)) / facing;
    if (hit > 0.0) axial += clamp(hit - axial, -v_surfaceAxial.y, v_surfaceAxial.y);
  }
  axial = max(u_surfaceNear, axial);
  float projected = dot(u_surfaceProjection.xy, vec2(axial, 1.0))
    / dot(u_surfaceProjection.zw, vec2(axial, 1.0));
  uint depth = v_surfaceAxial.x < 0.0 ? 65535u : uint(round(clamp(projected * .5 + .5, 0.0, 1.0) * 65534.0));
  uint rank = v_surfaceRank;
  if (u_surfaceNext) {
    uvec2 previous = surfaceKey(texelFetch(u_surfacePrevious, ivec2(gl_FragCoord.xy), 0));
    uint last = previous.y & 32767u;
    if (last == 0u || depth < previous.x || (depth == previous.x && rank >= last)) discard;
    // An inversion relative to the submitted order is the only reason to
    // replace center compositing. Store the marker in the spare rank bit.
    if (rank > last) rank |= 32768u;
  }
  gl_FragDepth = float(depth) / 65535.0;
  return vec4(depth & 255u, depth >> 8, rank & 255u, rank >> 8) / 255.0;
}
`;

const accumulate = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_color, u_depth, u_key;
layout(location=0) out vec4 color;
layout(location=1) out vec4 depth;
layout(location=2) out vec4 inverted;
${decodeKey}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  uvec2 key = surfaceKey(texelFetch(u_key, p, 0));
  color = texelFetch(u_color, p, 0);
  depth = texelFetch(u_depth, p, 0);
  inverted = (key.y & 32768u) != 0u ? vec4(1.0) : vec4(0.0);
}`;

const resolve = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_centerColor, u_centerDepth, u_orderedColor, u_orderedDepth, u_inverted, u_overflow;
uniform bool u_checkOverflow;
uniform ivec4 u_region;
uniform vec4 u_background;
layout(location=0) out vec4 color;
layout(location=1) out vec4 depth;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  bool inBounds = all(greaterThanEqual(p, u_region.xy)) && all(lessThan(p, u_region.xy + u_region.zw));
  if (inBounds && texelFetch(u_inverted, p, 0).r > .5 &&
      (!u_checkOverflow || texelFetch(u_overflow, p, 0).a == 0.0 && texelFetch(u_overflow, p, 0).b == 0.0)) {
    vec4 c = texelFetch(u_orderedColor, p, 0), d = texelFetch(u_orderedDepth, p, 0);
    color = c + vec4(u_background.rgb, u_background.a) * (1.0 - c.a);
    depth = d + vec4(1.0, 0.0, 0.0, 1.0) * (1.0 - d.a);
  } else {
    color = texelFetch(u_centerColor, p, 0);
    depth = texelFetch(u_centerDepth, p, 0);
  }
}
`;

export function createSurfaceOrder(gl, { target, table, maxSize, textureUnit, fullscreen, program }) {
  gl.activeTexture(gl.TEXTURE0 + rowUnit);
  const rows = table(rowWidth, rowUnit, maxSize), results = [];
  let source, hasNormals, peels, ordered, accumulation, finish;
  const zero = [0, 0, 0, 0];
  function prepare(world) {
    const next = [world.renderMetadata?.normals, world.pix, world.surfaceData, world.gaussianHarmonics?.normals, world.back, world.count];
    if (!source || next.some((value, i) => value !== source[i])) {
      const packed = normalRows(world);
      if (packed) rows.upload(packed);
      hasNormals = !!packed;
      source = next;
    }
    if (hasNormals) {
      rows.bind();
      // The center draw also links this sampler. A preceding palette draw
      // may have left its integer key texture on this unit.
      rows.bind(previousUnit);
    }
    return hasNormals;
  }

  function draw(center, camera, background, plan, index, drawSplats) {
    const { width, height } = center, { rect, layers, overflow } = plan;
    gl.activeTexture(gl.TEXTURE0 + 1);
    peels ??= [target(3, { depth: true }), target(3, { depth: true })];
    ordered ??= target(3);
    accumulation ??= program(accumulate);
    finish ??= program(resolve);
    const output = results[index] ??= target(2);
    for (const buffer of [...peels, ordered, output]) buffer.resize(width, height);
    const p = camera.projection;
    const uniforms = {
      u_surfacePeel: true, u_surfaceRows: rowUnit, u_surfacePrevious: previousUnit,
      u_surfaceProjection: [-p[10], p[14], -p[11], p[15]], u_surfaceLens: [width, height, height / (2 * camera.tangent)],
      u_surfaceRange: camera.depthRange,
      u_surfaceOrtho: camera.ortho, u_surfaceNear: camera.clipNear
    };
    let key;
    try {
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(...rect);
      gl.bindFramebuffer(gl.FRAMEBUFFER, ordered.framebuffer);
      for (let i = 0; i < 3; i++) gl.clearBufferfv(gl.COLOR, i, zero);
      for (let i = 0, n = layers + Number(overflow); i < n; i++) {
        const current = peels[i & 1], previous = peels[(i & 1) ^ 1];
        gl.bindFramebuffer(gl.FRAMEBUFFER, current.framebuffer);
        gl.viewport(0, 0, width, height);
        gl.disable(gl.BLEND);
        gl.enable(gl.DEPTH_TEST);
        gl.depthMask(true);
        gl.depthFunc(gl.LEQUAL);
        for (let j = 0; j < 3; j++) gl.clearBufferfv(gl.COLOR, j, zero);
        gl.clearBufferfv(gl.DEPTH, 0, [1]);
        rows.bind();
        textureUnit(previousUnit, previous.textures[2]);
        drawSplats({ ...uniforms, u_surfaceNext: i > 0 });
        key = current.textures[2];
        if (i === layers) break;
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE_MINUS_DST_ALPHA, gl.ONE);
        textureUnit(1, current.textures[0]);
        textureUnit(2, current.textures[1]);
        textureUnit(5, current.textures[2]);
        fullscreen(accumulation, ordered.framebuffer, width, height, { u_color: 1, u_depth: 2, u_key: 5 });
      }
    } finally {
      gl.disable(gl.SCISSOR_TEST);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.depthMask(true);
    }
    textureUnit(1, center.textures[0]);
    textureUnit(2, center.textures[1]);
    textureUnit(5, ordered.textures[0]);
    textureUnit(6, ordered.textures[1]);
    textureUnit(9, ordered.textures[2]);
    textureUnit(13, key);
    fullscreen(finish, output.framebuffer, width, height, {
      u_centerColor: 1, u_centerDepth: 2, u_orderedColor: 5, u_orderedDepth: 6,
      u_inverted: 9, u_overflow: 13, u_checkOverflow: overflow, u_region: rect, u_background: background
    });
    return output;
  }

  return { prepare, draw, uniforms: { u_surfaceRows: rowUnit, u_surfacePrevious: previousUnit } };
}
