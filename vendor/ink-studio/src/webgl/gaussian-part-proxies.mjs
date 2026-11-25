import { glassSurfaceOutputs } from './hybrid-depth.mjs';
import { gaussianHarmonicGLSL } from './gaussian-harmonics.mjs';

const plans = new WeakMap();
const rowWidth = 512;
const rowUnit = 26, layerUnit = 27;
const roles = { replace: 1, silhouette: 2, depth: 3, normals: 4 };

const packed = color => Number.isInteger(color) && color >= 0 && color <= 0xffffff;
const rgb = color => [color >> 16, color >> 8 & 255, color & 255].map(n => n / 255);
const vector = (v, n) => v?.length === n && Array.from(v).every(Number.isFinite);
const positions = values => Array.isArray(values) && values.length >= 2 && values.length <= 8192 &&
  Array.from(values).every(point => vector(point, 3));

/**
 * Compile painted row membership and proxy geometry in the imported world's
 * coordinate space. Pass the returned handle as world.gaussianPartProxies
 * or as a frame option. Rebuild the plan after changing masks or geometry.
 */
export function compileGaussianPartProxies(world, parts) {
  if (!world?.gaussianHarmonics || !Array.isArray(parts) || !parts.length || parts.length > 8)
    throw new TypeError('Gaussian part proxies need an imported live cloud and 1–8 parts');
  const { count, data } = world;
  const membership = new Float32Array(count);
  const entries = Array.from(parts, (part, layer) => {
    const { mask, shape, mode = 'replace', color, light } = part ?? {};
    const role = roles[mode];
    if (!Number.isInteger(role) || mask?.count !== count || !mask.indices?.length ||
        !(Array.isArray(mask.indices) || mask.indices instanceof Uint32Array))
      throw new TypeError('A part proxy needs a nonempty painted mask and a valid mode');
    const selected = new Set(mask.indices);
    for (const i of mask.indices) {
      if (!Number.isInteger(i) || i < 0 || i >= count || membership[i])
        throw new RangeError('Proxy masks must have unique, non-overlapping splat IDs');
      membership[i] = role === 1 ? -1 : layer + 1 + role * 16;
    }
    const owner = mask.indices[0];
    if (color !== undefined && !packed(color) || light !== undefined &&
        (!vector(light.direction, 3) || Math.hypot(...light.direction) < 1e-6 ||
          !Number.isFinite(light.strength) || light.strength < 0 || light.strength > 1))
      throw new TypeError('Proxy color or directional lighting is invalid');
    if (mode === 'normals' && !light) throw new TypeError('A normals proxy needs a light');
    const lightUniform = light ? [...light.direction, light.strength] : [0, 1, 0, 0];
    const points = shape?.points;
    if (!positions(points)) throw new TypeError('A proxy needs finite XYZ points');
    const source = shape.samples ?? points.map(() => owner);
    if (source.length !== points.length || Array.from(source).some(i => !Number.isInteger(i) ||
        i < 0 || i >= count || !selected.has(i)))
      throw new RangeError('Each color sample must belong to its proxy mask');
    const vertices = [];
    const sample = i => {
      const row = source[i], baked = data[row * 16 + 7];
      return [row, ...(color === undefined
        ? [baked & 255, baked >> 8 & 255, baked >> 16].map(n => n / 255) : rgb(color))];
    };
    let size, minPixels = 0;
    if (shape.type === 'mesh') {
      const { triangles, normals } = shape;
      if (points.length < 3 || !(Array.isArray(triangles) || triangles instanceof Uint32Array || triangles instanceof Uint16Array) ||
          !triangles.length || triangles.length % 3 || triangles.length > 98304 ||
          Array.from(triangles).some(i => !Number.isInteger(i) || i < 0 || i >= points.length) ||
          normals != null && (!Array.isArray(normals) || normals.length !== points.length ||
            Array.from(normals).some(n => !vector(n, 3) || Math.hypot(...n) < 1e-6)))
        throw new TypeError('A mesh proxy needs triangle indices and optional vertex normals');
      for (const i of triangles) vertices.push(...points[i], ...(normals?.[i] ?? [0, 0, 0]), ...sample(i));
      size = 10;
    } else if (shape.type === 'stroke') {
      const { radii } = shape;
      minPixels = shape.minPixels ?? 1;
      if (mode === 'normals' || light || !vector(radii, points.length) ||
          Array.from(radii).some(radius => radius <= 0) ||
          !Number.isFinite(minPixels) || minPixels < 0 || minPixels > 16)
        throw new TypeError('A stroke proxy needs positive radii and a minimum width from 0 to 16 art pixels');
      for (let i = 0; i < points.length - 1; i++) {
        vertices.push(...points[i], radii[i], ...points[i + 1], radii[i + 1], ...sample(i), ...sample(i + 1));
      }
      size = 16;
    } else throw new TypeError('Use a mesh or stroke for a part proxy');
    return { vertices: Float32Array.from(vertices), size, stroke: shape.type === 'stroke', role,
      owner, minPixels, live: color === undefined, light: lightUniform };
  });
  const handle = Object.freeze({ count, parts: entries.length });
  plans.set(handle, { data, membership, entries, hasMetadata: entries.some(entry => entry.role !== 1) });
  return handle;
}

export function gaussianProxyPlan(world, value) {
  if (!value) return null;
  const plan = plans.get(value);
  if (plan?.data !== world.data) throw new TypeError('Compile Gaussian part proxies for this imported world');
  return plan;
}

export const gaussianProxyVertex = `
uniform highp sampler2D u_gaussianProxyRows;
uniform int u_gaussianProxyCount;
flat out int v_gaussianProxy;
`;
// Layer pixels store a world normal and axial depth. The all-zero clear means
// the proxy doesn't cover the fragment. Depth is packed separately from alpha.
export const gaussianProxyFragment = `
uniform highp sampler2DArray u_gaussianProxyLayers;
uniform vec2 u_proxyGrid;
uniform vec4 u_proxyLights[8];
uniform vec2 u_proxyDepthRange;
uniform mat4 u_proxyProjection;
flat in int v_gaussianProxy;
vec3 proxyNormal(vec2 encoded) {
  vec2 oct = (encoded*255.0-1.0)/127.0-1.0;
  vec3 n = vec3(oct, 1.0-abs(oct.x)-abs(oct.y));
  if (n.z < 0.0) n.xy = (1.0-abs(n.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(n.xy,vec2(0)));
  return normalize(n);
}
float proxyLighting(vec3 n, vec4 light) {
  return mix(1.0, max(0.0, dot(n, normalize(light.xyz))), light.w);
}
`;

const uniforms = `
uniform mat4 u_view, u_proj;
uniform vec3 u_eye;
uniform bool u_ortho, u_live;
uniform vec4 u_clip, u_plane;
uniform vec2 u_viewport;
uniform float u_minRadius;
`;

const appearance = `
vec3 proxyColor(vec3 pos, vec4 source) {
  #ifdef SH
  vec3 ray = u_ortho ? -vec3(u_view[0].z,u_view[1].z,u_view[2].z) : pos-u_eye;
  if (u_live) return gaussianRgb(int(source.x),ray);
  #endif
  return source.yzw;
}
`;

function shaders(stroke, degree, glassLayers, wet) {
  const sh = degree === undefined ? '' : '#define SH\n' + gaussianHarmonicGLSL(degree);
  const declarations = stroke ? `
layout(location=0) in vec4 a_start;
layout(location=1) in vec4 a_end;
layout(location=2) in vec4 a_colorA;
layout(location=3) in vec4 a_colorB;
flat out vec4 v_line, v_widthAndZ, v_axial;
flat out vec3 v_colorA,v_colorB,v_worldA,v_worldB;
` : `
layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_normal;
layout(location=2) in vec4 a_color;
out vec3 v_world,v_ink,v_normal;
out float v_z;
`;
  const project = stroke ? `
  vec3 a = a_start.xyz, b = a_end.xyz;
  vec4 av = u_view*vec4(a,1), bv = u_view*vec4(b,1);
  float da = -av.z-u_near, db = -bv.z-u_near;
  if (max(da,db) <= 0.0) { gl_Position=vec4(2,2,2,1); return; }
  float blendA = da < 0.0 ? da/(da-db) : 0.0;
  float blendB = db < 0.0 ? db/(db-da) : 0.0;
  vec4 ac = u_proj*mix(av,bv,blendA), bc = u_proj*mix(bv,av,blendB);
  vec2 ap = (ac.xy/ac.w*0.5+0.5)*u_viewport, bp = (bc.xy/bc.w*0.5+0.5)*u_viewport;
  float focal = u_proj[1][1]*u_viewport.y*0.5;
  vec2 rad = max(vec2(u_minRadius), focal*vec2(mix(a_start.w,a_end.w,blendA)/ac.w,
    mix(a_end.w,a_start.w,blendB)/bc.w));
  vec2 corner = vec2(gl_VertexID & 1,gl_VertexID >> 1);
  vec2 low = min(ap-rad.x,bp-rad.y), hi = max(ap+rad.x,bp+rad.y);
  gl_Position=vec4(mix(low,hi,corner)/u_viewport*2.0-1.0,0,1);
  v_line=vec4(ap,bp);
  v_widthAndZ=vec4(rad,ac.z/ac.w,bc.z/bc.w);
  v_axial=vec4(-mix(av,bv,blendA).z,-mix(bv,av,blendB).z,1.0/ac.w,1.0/bc.w);
  v_worldA=mix(a,b,blendA); v_worldB=mix(b,a,blendB);
  vec3 colorA=proxyColor(a,a_colorA), colorB=proxyColor(b,a_colorB);
  v_colorA=mix(colorA,colorB,blendA);v_colorB=mix(colorB,colorA,blendB);
` : `
  v_world=a_position; v_normal=a_normal; v_ink=proxyColor(a_position,a_color);
  vec4 view = u_view*vec4(a_position,1); v_z=-view.z;
  gl_Position=u_proj*view;
`;
  const sample = stroke ? `
  vec2 axis=v_line.zw-v_line.xy, from=gl_FragCoord.xy-v_line.xy;
  float span=dot(axis,axis);
  float t=span<1e-8 ? float(v_widthAndZ.w<v_widthAndZ.z) : clamp(dot(from,axis)/span,0.0,1.0);
  if (length(from-axis*t)>mix(v_widthAndZ.x,v_widthAndZ.y,t)) discard;
  float w = t*v_axial.w/max(1e-8,mix(v_axial.z,v_axial.w,t));
  vec3 world=mix(v_worldA,v_worldB,w), ink=mix(v_colorA,v_colorB,w), normal=vec3(0,1,0);
  float z=mix(v_axial.x,v_axial.y,w);
  gl_FragDepth=mix(v_widthAndZ.z,v_widthAndZ.w,t)*0.5+0.5;
` : `
  vec3 world=v_world, ink=v_ink;
  vec3 normal=dot(v_normal,v_normal)>1e-8 ? normalize(v_normal) :
    normalize(cross(dFdx(v_world),dFdy(v_world)))*(gl_FrontFacing?1.0:-1.0);
  float z=v_z;
  gl_FragDepth=gl_FragCoord.z;
`;
  return [`#version 300 es
precision highp float;precision highp int;
${sh}
${uniforms}
${declarations}
uniform float u_near;
${appearance}
void main(){${project}}`,
    `#version 300 es
precision highp float;precision highp int;
${uniforms}
${declarations.replace(/^layout.*\n/gm, '').replace(/\bout /g,'in ')}
uniform int u_pass, u_owner;
uniform vec4 u_range,u_light;
uniform vec3 u_fog;
layout(location=0) out vec4 color;
layout(location=1) out vec4 depth;
${glassLayers && !wet ? 'vec4 aux;' : 'layout(location=2) out vec4 aux;'}
${glassLayers ? glassSurfaceOutputs(wet) : ''}
vec4 encode(int value) {
  return vec4(value&255,(value>>8)&255,(value>>16)&255,255)/255.0;
}
void main(){
  ${sample}
  vec4 point=vec4(world,1);
  if (dot(point,u_plane)<0.0 || dot(u_view*point,u_clip)<0.0 || gl_FragDepth<0.0 || gl_FragDepth>1.0) discard;
  float linear=clamp((z-u_range.x)/(u_range.y-u_range.x),0.0,1.0);
  depth=vec4(linear,0,0,1); aux=vec4(0,0,0,1);
  if (u_pass==4) {
    normal/=dot(abs(normal),vec3(1));
    vec2 oct=normal.xy;
    if (normal.z<0.0) oct=(1.0-abs(oct.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(oct,vec2(0)));
    float d=round(linear*65535.0);
    color=vec4(round((oct*0.5+0.5)*254.0)+1.0,floor(d/256.0),mod(d,256.0))/255.0;
  } else if (u_pass==1) {
    color=encode(u_owner+1); aux=encode(int(round(linear*16777215.0)));
  } else if (u_pass==2) {
    color=vec4(1);
  } else if (u_pass==3) {
    color=vec4(0);
  } else {
    ink*=mix(1.0,max(0.0,dot(normal,normalize(u_light.xyz))),u_light.w);
    float haze=u_range.z<1.0?smoothstep(u_range.z,1.0,linear)*u_range.w:0.0;
    color=vec4(mix(ink,u_fog,haze),1);
    ${glassLayers ? 'splitOpaqueSurface(color,gl_FragDepth);' : ''}
  }
}`];
}

export function createGaussianProxyRenderer(gl, resources, table, maxSize) {
  const rows = table(rowWidth, rowUnit, maxSize);
  const programs = new Map();
  let uploaded, packet, layerTexture, framebuffer, layerDepth, sizes, records = [];
  function upload(plan) {
    if (uploaded === plan) { rows.bind(); return; }
    for (const { vao, buffer } of records) {
      resources.free(vao);
      resources.free(buffer);
    }
    records = plan.entries.map(entry => {
      const vao = resources.allocate('VertexArray'), buffer = resources.allocate('Buffer');
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, entry.vertices, gl.STATIC_DRAW);
      let offset = 0;
      for (const [i, width] of (entry.stroke ? [4,4,4,4] : [3,3,4]).entries()) {
        gl.enableVertexAttribArray(i);
        gl.vertexAttribPointer(i,width,gl.FLOAT,false,entry.size*4,offset*4);
        if (entry.stroke) gl.vertexAttribDivisor(i,1);
        offset += width;
      }
      return { ...entry, vao, buffer };
    });
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);
    rows.upload(plan.membership);
    uploaded = plan;
    packet = { u_gaussianProxyRows:rowUnit, u_gaussianProxyLayers:layerUnit, u_gaussianProxyCount:plan.membership.length,
      u_proxyLights:Float32Array.from([...plan.entries.flatMap(entry=>entry.light),
        ...Array(8-plan.entries.length).fill([0,1,0,0]).flat()]) };
  }
  function drawPart(entry, frame, pass) {
    const degree = frame.gaussian?.u_gaussianCount && entry.live && pass === 0 ? frame.degree : undefined;
    const glassLayers = pass === 0 && frame.glassLayers;
    const key = (entry.stroke ? 8 : 0) + (degree ?? -1) + (glassLayers ? (frame.wet ? 32 : 16) : 0);
    if (!programs.has(key)) programs.set(key,resources.program(...shaders(entry.stroke,degree,glassLayers,frame.wet)));
    const shader = programs.get(key), {camera,width,height,scale,range,fog,plane,clip} = frame;
    gl.useProgram(shader.p);
    gl.bindVertexArray(entry.vao);
    shader.set({ ...frame.gaussian, u_view:camera.view,u_proj:camera.projection,u_eye:camera.eye,u_ortho:camera.ortho,
      u_live:entry.live,u_viewport:[width,height],u_near:camera.clipNear,
      u_minRadius:entry.minPixels/Math.SQRT2*scale,
      u_range:range,u_fog:fog,u_glassBoundary:13,u_light:entry.light,u_pass:pass,u_owner:entry.owner,u_plane:plane,u_clip:clip });
    const count = entry.vertices.length / entry.size;
    if (entry.stroke) gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,count);
    else gl.drawArrays(gl.TRIANGLES,0,count);
  }
  return {
    prepare(plan, frame) {
      upload(plan);
      const {width,height,camera} = frame;
      gl.activeTexture(gl.TEXTURE0 + layerUnit);
      if (!layerTexture) {
        layerTexture = resources.allocate('Texture');
        gl.bindTexture(gl.TEXTURE_2D_ARRAY,layerTexture);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      } else gl.bindTexture(gl.TEXTURE_2D_ARRAY,layerTexture);
      const size = plan.hasMetadata ? [width,height,records.length] : [1,1,1];
      if (!sizes || size.some((n,i)=>n!==sizes[i])) {
        gl.texImage3D(gl.TEXTURE_2D_ARRAY,0,gl.RGBA8,...size,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
        sizes = size;
      }
      if (plan.hasMetadata) {
        framebuffer ??= resources.allocate('Framebuffer');
        layerDepth ??= resources.allocate('Renderbuffer');
        gl.bindRenderbuffer(gl.RENDERBUFFER,layerDepth);
        gl.renderbufferStorage(gl.RENDERBUFFER,gl.DEPTH_COMPONENT16,width,height);
        gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.RENDERBUFFER,layerDepth);
        gl.viewport(0,0,width,height);
        gl.disable(gl.BLEND);gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.depthMask(true);
        for (const [i,entry] of records.entries()) if (entry.role !== 1) {
          gl.framebufferTextureLayer(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,layerTexture,0,i);
          gl.clearBufferfv(gl.DEPTH,0,[1]);
          gl.clearBufferfv(gl.COLOR,0,[0,0,0,0]);
          drawPart(entry,frame,4);
        }
        gl.disable(gl.DEPTH_TEST);
      }
      return { ...packet, u_proxyDepthRange:camera.depthRange, u_proxyProjection:camera.projection };
    },
    draw(frame, pass) {
      for (const entry of records) if (entry.role === 1) drawPart(entry,frame,pass);
    },
    get hasGeometry() { return records.some(entry => entry.role === 1); }
  };
}
