export function splatPaintStyle(world, candidate) {
  if (!candidate || !world.gaussianHarmonics) return null;
  const { mask, protect = false, preview = false, ramp } = candidate;
  if (!mask || mask.count !== world.gaussianHarmonics.count ||
      !(Array.isArray(mask.indices) || mask.indices instanceof Uint32Array) ||
      typeof protect !== 'boolean' || typeof preview !== 'boolean' ||
      ramp != null && (!Array.isArray(ramp) || !ramp.length || ramp.length > 64 ||
        !ramp.every(color => Number.isInteger(color) && color >= 0 && color <= 0xffffff)))
    throw new TypeError('Splat paint needs a mask for the imported cloud and an optional RGB ramp');
  return mask.indices.length && (protect || preview || ramp) ? { mask, protect, preview, ramp } : null;
}

const fragment = `#version 300 es
precision highp float;
uniform sampler2D u_resolved, u_sceneColor, u_coverage, u_marked;
uniform vec3 u_ramp[64];
uniform int u_rampCount;
uniform bool u_protect, u_preview;
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec4 base = texelFetch(u_resolved, p, 0);
  float marked = texelFetch(u_marked, p, 0).r, coverage = texelFetch(u_coverage, p, 0).a;
  if (marked >= 0.035 && marked >= 0.45 * coverage) {
    if (u_rampCount > 0) {
      vec3 source = texelFetch(u_sceneColor, p, 0).rgb;
      float best = 1e9;
      for (int i = 0; i < u_rampCount; i++) {
        vec3 d = u_ramp[i] - source;
        float error = dot(d*d, vec3(0.30, 0.55, 0.15));
        if (error < best) { best = error; base.rgb = u_ramp[i]; }
      }
    }
    if (u_protect) base.a = 0.5;
    if (u_preview) base.rgb = mix(base.rgb, vec3(.16, .92, .83), .53);
  }
  o = base;
}
`;

export function createSplatPaintPass({ table, target, maxSize, program, textureUnit, fullscreen }) {
  const source = target(), mask = target(2), shader = program(fragment);
  const unit = 12, rows = table(512, unit, maxSize);
  let uploaded;
  return {
    source, mask,
    resize(width, height) {
      source.resize(width, height);
      mask.resize(width, height);
    },
    bind(snapshot) {
      if (snapshot !== uploaded) {
        const values = new Float32Array(snapshot.count);
        for (const i of snapshot.indices) {
          if (!Number.isInteger(i) || i < 0 || i >= snapshot.count) throw new RangeError('Invalid painted splat index');
          values[i] = 1;
        }
        rows.upload(values);
        uploaded = snapshot;
      } else rows.bind();
      return unit;
    },
    finish(settings, scene, output, width, height) {
      for (const [i, texture] of [source.textures[0], scene, ...mask.textures].entries()) textureUnit(i+12, texture);
      const ramp = new Float32Array(64 * 3);
      settings.ramp?.forEach((color, i) => ramp.set([color >> 16, color >> 8 & 255, color & 255].map(v => v/255), i*3));
      fullscreen(shader, output, width, height, { u_resolved:12, u_sceneColor:13, u_coverage:14, u_marked:15,
        u_rampCount: settings.ramp?.length ?? 0, u_ramp: ramp, u_protect: settings.protect, u_preview: settings.preview });
    }
  };
}
