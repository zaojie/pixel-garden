import { environmentMediaSource } from './environment-shaders.mjs';

// Each side retains its own palette and lighting. Front-side fragments
// mark the source coverage before compositing the world-space aperture.
export const portalComposite = (media, lit) => `#version 300 es
precision highp float;
uniform sampler2D u_source, u_destination, u_cover, u_spriteCover, u_foreground;
uniform sampler2D u_aperture;
uniform vec3 u_origin, u_right, u_up, u_normal, u_eye;
uniform vec4 u_entryPlane;
uniform mat4 u_portalView;
uniform vec3 u_settings;
uniform vec2 u_half, u_size;
uniform int u_opening, u_hasSpriteCover, u_painted;
uniform float u_fade, u_far;
out vec4 o;
${media ? environmentMediaSource(lit) : ''}
bool throughOpening(out float axialDepth) {
  vec2 p = (gl_FragCoord.xy / u_size * 2.0 - 1.0) * u_settings.xy;
  bool ortho = u_settings.z > 0.5;
  mat3 world = transpose(mat3(u_portalView));
  vec3 eye = u_eye + (ortho ? world * vec3(p, 0.0) : vec3(0));
  vec3 ray = world * vec3(ortho ? vec2(0) : p, -1.0);
  float towards = dot(ray, u_normal);
  if (towards >= -1e-7) return false;
  float t = dot(u_origin - eye, u_normal) / towards;
  if (t <= 0.0 || t >= u_far) return false;
  axialDepth = t;
  vec3 at = eye + ray * t;
  if (dot(u_entryPlane, vec4(at, 1.0)) < 0.0) return false;
  vec3 relative = at - u_origin;
  vec2 local = vec2(dot(relative, u_right), dot(relative, u_up)) / u_half;
  if (u_opening == 2 ? dot(local, local) > 1.0 : any(greaterThan(abs(local), vec2(1.0)))) return false;
  if (u_painted != 0) {
    ivec2 size = textureSize(u_aperture, 0);
    vec2 uv = vec2(local.x, -local.y) * 0.5 + 0.5;
    ivec2 cell = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
    if (texelFetch(u_aperture, cell, 0).r < 0.5) return false;
  }
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  return texelFetch(u_cover, pixel, 0).r < 0.5 &&
    (u_hasSpriteCover == 0 || texelFetch(u_spriteCover, pixel, 0).r < 0.5);
}
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  vec3 color = texelFetch(u_source, p, 0).rgb;
  float axialDepth;
  if (u_opening != 0 && throughOpening(axialDepth)) {
    vec4 foreground = texelFetch(u_foreground, p, 0);
    vec3 destination = texelFetch(u_destination, p, 0).rgb;
    ${media ? 'destination = envMedia(destination, gl_FragCoord.xy, u_size, axialDepth);' : ''}
    color = foreground.rgb + destination * (1.0 - foreground.a);
  }
  o = vec4(color * (1.0 - u_fade), 1.0);
}`;
