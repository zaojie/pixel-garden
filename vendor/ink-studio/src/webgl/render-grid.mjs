/* One texture read per output pixel; no readback, cell buffers or new scene geometry. */
export function createRenderGrid(gl, { api, target, textureUnit, program, fullscreen }) {
  const base = target();
  const shader = program(`#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_input;
uniform vec2 u_size, u_origin, u_rotation;
uniform float u_cellSize, u_aspect, u_gap;
uniform vec3 u_gapColor;
uniform int u_kind;
out vec4 color;
${api.glsl}
void main(){
  vec3 cell=inkGridCell(vec2(gl_FragCoord.x,u_size.y-gl_FragCoord.y),u_kind,u_cellSize,u_origin,u_rotation,u_aspect);
  ivec2 p=clamp(ivec2(floor(cell.xy)),ivec2(0),ivec2(u_size)-1);
  p.y=int(u_size.y)-1-p.y;
  color=cell.z<u_gap?vec4(u_gapColor,1.0):texelFetch(u_input,p,0);
}`);
  return function render(world, frame, plan, w, h, output, present) {
    api.dimensions(plan, w, h);
    base.resize(w, h);
    present(world, frame, w, h, base.framebuffer);
    if (!output) {
      if (gl.canvas.width !== w) gl.canvas.width = w;
      if (gl.canvas.height !== h) gl.canvas.height = h;
    }
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.STENCIL_TEST);
    gl.disable(gl.SCISSOR_TEST);
    gl.disable(gl.BLEND);
    gl.colorMask(true, true, true, true);
    textureUnit(0, base.textures[0]);
    fullscreen(shader, output, w, h, {
      u_input: 0,
      u_size: [w, h],
      u_origin: plan.origin,
      u_rotation: [plan.cos, plan.sin],
      u_cellSize: plan.size,
      u_aspect: plan.aspect,
      u_gap: plan.gap,
      u_gapColor: plan.gapColor.map(c => c / 255),
      u_kind: plan.kind
    });
    gl.activeTexture(gl.TEXTURE0);
  };
}
