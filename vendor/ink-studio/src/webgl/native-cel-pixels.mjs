const vertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_pixel;
layout(location=1) in vec3 a_rgb;
uniform vec2 u_size;
flat out vec3 v_rgb;
flat out float v_depth;
void main() {
  vec2 uv = (a_pixel.xy + .5) / u_size;
  gl_Position = vec4(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 0.0, 1.0);
  gl_PointSize = 1.0;
  v_depth = a_pixel.z;
  v_rgb = a_rgb;
}`;

const fragment = `#version 300 es
precision highp float;
uniform sampler2D u_ownerDepth;
uniform vec2 u_depthRange;
flat in vec3 v_rgb;
flat in float v_depth;
out vec4 color;
void main() {
  vec4 owner = texelFetch(u_ownerDepth, ivec2(gl_FragCoord.xy), 0);
  float z = dot(round(owner.rgb * 255.0), vec3(1.0, 256.0, 65536.0)) / 16777215.0;
  if (owner.a > .5 && v_depth > mix(u_depthRange.x, u_depthRange.y, z) + .0001) discard;
  color = vec4(v_rgb, 1.0);
}`;

export function createNativeCelDraw(gl, resources, textureUnit, pixels) {
  const shader = resources.program(vertex, fragment);
  const vao = resources.allocate('VertexArray'), buffer = resources.allocate('Buffer');
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  for (let i = 0; i < 2; i++) {
    gl.enableVertexAttribArray(i);
    gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 24, i * 12);
  }
  return function draw(rows, width, height, output, depth, range, retained = []) {
    const vertices = [];
    const emit = (x, y, z, r, g, b) => vertices.push(x, y, z, r/255, g/255, b/255);
    pixels(rows, width, height, emit);
    for (const row of retained) emit(...row);
    if (!vertices.length) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, output);
    gl.viewport(0, 0, width, height);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, Float32Array.from(vertices), gl.DYNAMIC_DRAW);
    textureUnit(3, depth);
    gl.useProgram(shader.p);
    shader.set({ u_size: [width, height], u_ownerDepth: 3, u_depthRange: range });
    const dither = gl.isEnabled(gl.DITHER);
    gl.disable(gl.DITHER);
    gl.drawArrays(gl.POINTS, 0, vertices.length / 6);
    if (dither) gl.enable(gl.DITHER);
  };
}

// Gather only the native pixels an authored mark could occupy.
export function createNativeDepthProbe(gl, resources, target, textureUnit) {
  const size = 256;
  const buffer = resources.allocate('Buffer'), vao = resources.allocate('VertexArray'), output = target();
  const shader = resources.program(`#version 300 es
precision highp float;
layout(location=0) in vec2 a_sample;
uniform vec2 u_size;
flat out vec2 v_sample;
void main(){
  ivec2 at=ivec2(gl_VertexID&255,gl_VertexID>>8);
  gl_Position=vec4((vec2(at)+.5)/u_size*2.-1.,0.,1.);
  gl_PointSize=1.;
  v_sample=a_sample;
}`,`#version 300 es
precision highp float;
uniform highp sampler2D u_depth;
flat in vec2 v_sample;
out vec4 packedDepth;
void main(){
  ivec2 at=ivec2(v_sample);
  packedDepth=texelFetch(u_depth,ivec2(at.x,textureSize(u_depth,0).y-1-at.y),0);
}`);
  gl.bindVertexArray(vao);
  gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0,2,gl.FLOAT,false,8,0);
  return function probe(coordinates, texture, range) {
    const count=coordinates.size;
    if(!count)return new Map();
    const w=Math.min(size,count), h=Math.ceil(count/size);
    const source=new Float32Array(count*2), packed=new Uint8Array(w*h*4);
    let at=0;
    for(const point of coordinates.values()){ source.set(point,at);at+=2; }
    output.resize(w,h);
    gl.bindFramebuffer(gl.FRAMEBUFFER,output.framebuffer);
    gl.viewport(0,0,w,h);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER,buffer);
    gl.bufferData(gl.ARRAY_BUFFER,source,gl.STREAM_DRAW);
    textureUnit(3,texture);
    gl.useProgram(shader.p);
    shader.set({u_depth:3,u_size:[w,h]});
    const dither=gl.isEnabled(gl.DITHER);
    gl.disable(gl.DITHER);
    gl.drawArrays(gl.POINTS,0,count);
    if(dither)gl.enable(gl.DITHER);
    gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,packed);
    const samples=new Map(), step=(range[1]-range[0])/16777215;
    at=0;
    for(const key of coordinates.keys()){
      if(packed[at+3])samples.set(key,range[0]+step*(packed[at]+256*packed[at+1]+65536*packed[at+2]));
      at+=4;
    }
    return samples;
  };
}
