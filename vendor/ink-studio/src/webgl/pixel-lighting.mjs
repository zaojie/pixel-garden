import { createResources } from './resources.mjs';

const vertex = `#version 300 es
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const fragment = (maxLights, maxShadowSteps) => `#version 300 es
precision highp float;
precision highp int;
uniform highp isampler2D u_surface;
uniform highp isampler2D u_normal;
uniform highp sampler2D u_palette;
uniform ivec2 u_size;
uniform ivec4 u_sources[${maxLights}];
uniform ivec2 u_settings[${maxLights}];
uniform int u_count, u_ambient, u_bands;
uniform bool u_shadows;
out vec4 outColor;

// Same rational boundary traversal and integer energy arithmetic as PixelLighting.render.
bool blocked(ivec2 p, int z, ivec3 delta) {
  ivec2 distance = abs(delta.xy), direction = sign(delta.xy);
  if (all(equal(distance, ivec2(0)))) return false;
  ivec2 next = ivec2(8);
  const int infinity = 2147483647;
  for (int step = 0; step < ${maxShadowSteps}; ++step) {
    int tx = distance.x == 0 ? infinity : next.x * distance.y;
    int ty = distance.y == 0 ? infinity : next.y * distance.x;
    int numerator = tx <= ty ? next.x : next.y;
    int denominator = tx <= ty ? distance.x : distance.y;
    if (numerator >= denominator) return false;
    if (tx <= ty) { p.x += direction.x; next.x += 16; }
    if (ty <= tx) { p.y += direction.y; next.y += 16; }
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, u_size))) return false;
    if (texelFetch(u_surface, p, 0).b > z + delta.z * numerator / denominator + 1) return true;
  }
  return false;
}

void main() {
  ivec2 p = ivec2(int(gl_FragCoord.x), u_size.y - 1 - int(gl_FragCoord.y));
  ivec3 surface = texelFetch(u_surface, p, 0).rgb;
  if (surface.r == 0) { outColor = vec4(0); return; }
  ivec3 n = texelFetch(u_normal, p, 0).rgb;
  n.y = -n.y;
  ivec3 position = ivec3(p * 16 + 8, surface.g);
  int energy = u_ambient;
  for (int i = 0; i < ${maxLights}; ++i) {
    if (i >= u_count || energy >= 256) break;
    ivec3 d = u_sources[i].xyz - position;
    int radius = u_sources[i].w, strength = u_settings[i].x;
    if (strength == 0 || d.z <= 0 || max(max(abs(d.x), abs(d.y)), d.z) >= radius) continue;
    int squared = d.x*d.x + d.y*d.y + d.z*d.z;
    int cosine = n.x*d.x + n.y*d.y + n.z*d.z;
    if (cosine <= 0 || squared >= radius * radius) continue;
    int distance = int(sqrt(float(squared)));
    // Correct a floating-point sqrt rounded across a perfect-square boundary.
    if (distance * distance > squared) --distance;
    if ((distance + 1) * (distance + 1) <= squared) ++distance;
    int facing = min(256, cosine * 4 / distance);
    int fade = (radius - distance) * 256 / radius;
    int power = strength * facing * fade / 65536;
    if (power > 0 && (!u_shadows || u_settings[i].y == 0 || !blocked(p, surface.g, d))) energy += power;
  }
  int band = (min(energy, 256) * (u_bands - 1) + 128) / 256;
  outColor = texelFetch(u_palette, ivec2(band, surface.r - 1), 0);
}`;

export function createPixelLightingPass(gl, api, sharedResources) {
  if (!gl?.texStorage2D || !api?.frame) throw new TypeError('Pixel lighting needs WebGL2 and KIT.pixelLighting');
  const resources = sharedResources ?? createResources(gl);
  let shader, framebuffer, vao, surface, normal, palette, texture;
  try {
    shader = resources.program(vertex, fragment(api.maxLights, api.maxShadowSteps));
    framebuffer = resources.allocate('Framebuffer');
    vao = resources.allocate('VertexArray');
    surface = resources.texture();
    normal = resources.texture();
    palette = resources.texture();
    texture = resources.texture();
  } catch (error) {
    if (!sharedResources) resources.dispose();
    throw error;
  }
  let uploaded, width, height, disposed = false;
  function unit(index, target) {
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, target);
  }

  function draw(spec, options) {
    if (disposed) throw new Error('The pixel lighting pass has been disposed');
    const { scene, lights, ambient, shadows } = api.frame(spec, options);
    const { w, h } = scene;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    if (width !== w || height !== h) {
      unit(0, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Pixel lighting framebuffer is incomplete');
      width = w;
      height = h;
    }
    unit(0, surface);
    if (scene !== uploaded)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16I, w, h, 0, gl.RGBA_INTEGER, gl.SHORT, scene.surfaces);
    unit(1, normal);
    if (scene !== uploaded)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8I, w, h, 0, gl.RGBA_INTEGER, gl.BYTE, scene.normals);
    unit(2, palette);
    if (scene !== uploaded)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, scene.bands, scene.rows, 0, gl.RGBA, gl.UNSIGNED_BYTE, scene.colors);
    uploaded = scene;
    gl.viewport(0, 0, w, h);
    for (const flag of [gl.BLEND, gl.CULL_FACE, gl.DEPTH_TEST, gl.SCISSOR_TEST, gl.STENCIL_TEST, gl.RASTERIZER_DISCARD])
      gl.disable(flag);
    const dither = gl.isEnabled(gl.DITHER);
    gl.disable(gl.DITHER);
    gl.colorMask(true, true, true, true);
    gl.bindVertexArray(vao);
    gl.useProgram(shader.p);
    shader.set({ u_surface: 0, u_normal: 1, u_palette: 2, u_size: [w, h],
      u_sources: lights.sources, u_settings: lights.settings, u_count: lights.count,
      u_ambient: ambient, u_bands: scene.bands, u_shadows: shadows });
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    if (dither) gl.enable(gl.DITHER);
    return { w, h, texture, framebuffer, region: [0, 0, w, h] };
  }

  return { draw, dispose() {
    if (disposed) return;
    disposed = true;
    uploaded = undefined;
    if (sharedResources) for (const value of [texture, palette, normal, surface, vao, framebuffer, shader.p])
      resources.free(value);
    else resources.dispose();
  } };
}
