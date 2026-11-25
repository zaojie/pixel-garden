// The owner pass stores axial distance in its existing 24-bit attachment.
export function readHybridDepth(gl, framebuffer, width, height, range) {
  const packed = new Uint8Array(width * height * 4);
  const depth = new Float32Array(width * height);
  gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
  gl.readBuffer(gl.COLOR_ATTACHMENT2);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, packed);
  const step = (range[1] - range[0]) / 16777215;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const src = ((height - 1 - y) * width + x) * 4;
      depth[y * width + x] = packed[src + 3]
        ? range[0] + (packed[src] + packed[src + 1] * 256 + packed[src + 2] * 65536) * step
        : Infinity;
    }
  }
  return depth;
}

export function glassSurfaceOutputs(wet) {
  return `
uniform highp sampler2D u_glassBoundary;
layout(location=${wet ? 3 : 2}) out vec4 glassBack;
layout(location=${wet ? 4 : 3}) out vec4 glassThrough;
void splitOpaqueSurface(vec4 color, float projectedDepth) {
  vec4 pane = texelFetch(u_glassBoundary, ivec2(gl_FragCoord.xy), 0);
  float boundary = dot(floor(pane.rgb * 255.0 + 0.5), vec3(1, 256, 65536)) / 16777215.0;
  bool behind = pane.a > 0.0 && projectedDepth > boundary + 0.0000001;
  glassBack = behind ? color : vec4(0);
  glassThrough = vec4(vec3(behind ? 1.0 : 0.0), 1.0);
}`;
}
