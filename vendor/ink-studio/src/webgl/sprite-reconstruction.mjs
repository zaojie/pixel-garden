import { createResources } from './resources.mjs';
import { environmentMediaSource } from './environment-shaders.mjs';
import { waterGLSL } from './water-optics.mjs';
import { glassSurfaceOutputs } from './hybrid-depth.mjs';

const cacheLimit = 32 * 1024 * 1024;
const unit = 29;

// Directional atlas fragments enter the same depth buffer as native sprite surfaces.
export function createSpriteReconstructionPipeline(gl, api, cutaway) {
  const resources = createResources(gl),
    cache = new Map(),
    programs = new Map();
  let bytes = 0;
  const vao = resources.allocate('VertexArray');
  function texture(image) {
    let item = cache.get(image);
    if (item) {
      cache.delete(image);
      cache.set(image, item);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, item);
      return;
    }
    while (bytes + image.values.byteLength > cacheLimit && cache.size) {
      const [oldest, previous] = cache.entries().next().value;
      resources.free(previous);
      bytes -= oldest.values.byteLength;
      cache.delete(oldest);
    }
    item = resources.allocate('Texture');
    gl.bindTexture(gl.TEXTURE_2D_ARRAY, item);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    const layers = image.values.length / (image.width * image.height * 4);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGBA32F, image.width, image.height, layers, 0, gl.RGBA, gl.FLOAT, image.values);
    cache.set(image, item);
    bytes += image.values.byteLength;
  }

  function program(wet, section, env, glassLayers) {
    const key = [!!wet, wet?.airPockets, wet?.layered, wet?.atlas, !!section, !!env?.media, !!env?.lit, glassLayers].join('/');
    if (programs.has(key)) return programs.get(key);
    const shader = resources.program(
      `#version 300 es
precision highp float;
uniform mat4 u_view,u_proj;
uniform vec3 u_anchor,u_right,u_up;
uniform vec2 u_size,u_pivot;
uniform float u_units;
out vec2 v_texel;
out vec3 v_world;
void main(){
  vec2 corner=vec2(gl_VertexID & 1,gl_VertexID >> 1)*u_size;
  v_texel=corner;
  v_world=u_anchor+u_units*((corner.x-u_pivot.x)*u_right+(u_pivot.y-corner.y)*u_up);
  gl_Position=u_proj*u_view*vec4(v_world,1.0);
}`,
      `#version 300 es
precision highp float; precision highp int;
uniform highp sampler2DArray u_sprite;
uniform mat4 u_view,u_proj;
uniform mat3 u_normalAxes;
uniform vec3 u_eye,u_plane,u_back,u_fog,u_sun;
uniform vec4 u_range,u_splatPlane;
uniform vec2 u_surfaceViewport,u_artScale,u_sunGain,u_size;
uniform int u_phase,u_step;
uniform bool u_ortho,u_depth,u_normal,u_emission,u_baked,u_effect,u_index,u_graphDepth,u_portalCover,u_markPass,u_historyOwner;
uniform float u_scale,u_effectDepth,u_filterSoftness;
uniform bool u_filterMatte;
${wet ? waterGLSL(wet.airPockets, wet.layered, wet.atlas) : 'uniform vec4 u_arrivalPlane;'}
${section ? cutaway : ''}
${env?.media ? environmentMediaSource(env.lit) : ''}
${api.glsl}
in vec2 v_texel;
in vec3 v_world;
layout(location=0) out vec4 color;
layout(location=1) out vec4 depth;
${glassLayers && !wet ? 'vec4 water;' : 'layout(location=2) out vec4 water;'}
${glassLayers ? glassSurfaceOutputs(wet) : ''}
void main(){
  vec2 pixel=gl_FragCoord.xy*u_artScale;
  if(spriteRank(pixel,u_phase)<u_step)discard;
  ivec2 size=textureSize(u_sprite,0).xy;
  vec2 texel=all(equal(vec2(size),u_size))?v_texel:v_texel/u_size*vec2(size);
  ivec2 tx=clamp(ivec2(texel),ivec2(0),size-1);
  vec4 ink=texelFetch(u_sprite,ivec3(tx,u_effect?3:0),0);
  if(ink.a*64.0<=float(spriteRank(pixel+vec2(3,5),u_phase^45)))discard;
  vec4 material=texelFetch(u_sprite,ivec3(tx,1),0);
  vec3 ray=u_ortho?-u_back:normalize(v_world-u_eye);
  float facing=-dot(ray,u_plane);
  if(facing<=1.0e-7)discard;
  vec3 position=v_world+ray*(u_depth?(u_effect?u_effectDepth:material.w)*u_scale/facing:0.0);
  vec4 camera=u_view*vec4(position,1.0);
  if(dot(u_arrivalPlane,vec4(position,1.0))<0.0||dot(u_splatPlane,camera)<0.0)discard;
  vec4 projected=u_proj*camera;
  float z=0.5+0.5*projected.z/projected.w;
  if(z<0.0||z>1.0||projected.w<=0.0)discard;
  gl_FragDepth=z;
  if(u_markPass){color=vec4(0);depth=vec4(0);return;}
  float axial=-camera.z;
  ${section ? 'float margin=sectionMargin(position,axial);if(margin<0.0)discard;' : ''}
  float d=clamp((axial-u_range.x)/(u_range.y-u_range.x),0.0,1.0);
  float fog=u_range.z<1.0?smoothstep(u_range.z,1.0,d)*u_range.w:0.0;
  vec3 glow=u_emission&&!u_effect?texelFetch(u_sprite,ivec3(tx,2),0).rgb:vec3(0);
  float gain=u_normal&&!u_effect&&dot(material.xyz,material.xyz)>1.0e-8?
    u_sunGain.x+u_sunGain.y*max(0.0,dot(normalize(u_normalAxes*material.xyz),u_sun)):1.0;
  vec3 rgb=u_baked?(gain==1.0?ink.rgb:max(vec3(0),ink.rgb-glow)*gain+glow):ink.rgb*gain+glow;
  rgb=clamp(rgb,0.0,1.0);
  vec3 mark=vec3(0);
  ${wet ? 'if(!u_index)rgb=waterPaint(rgb,gl_FragCoord.xy,u_surfaceViewport,axial,mark);' : ''}
  ${env?.media ? 'if(!u_index)rgb=envMedia(rgb,gl_FragCoord.xy,u_surfaceViewport,axial);' : ''}
  rgb=mix(rgb,u_fog,fog);
  ${section ? 'rgb=mix(rgb,u_sectionEdge.rgb,sectionInk(margin));' : ''}
  color=u_filterMatte?vec4(vec3(u_filterSoftness),1):u_index?vec4(float(u_portalCover)):vec4(rgb,1);
  depth=u_historyOwner?vec4(0):vec4(d,0,0,1);
  water=vec4(mark,1);
  ${glassLayers ? 'splitOpaqueSurface(color,z);' : ''}
  if(u_index&&u_graphDepth){
    float packedDepth=round(d*16777215.0);
    water=vec4(mod(packedDepth,256.0),mod(floor(packedDepth/256.0),256.0),floor(packedDepth/65536.0),255.0)/255.0;
  }
}`
    );
    programs.set(key, shader);
    return shader;
  }

  return {
    draw(rows, frame) {
      const {
        view,
        projection,
        eye,
        back,
        ortho,
        sun,
        artScale,
        range,
        fog,
        width,
        height,
        wet,
        water,
        environment,
        section,
        index,
        graphDepth,
        anchors,
        coverageFilter,
        historyOwner,
        glassLayers,
        portalCover,
        arrivalPlane,
        clip
      } = frame;
      const shader = program(wet, section, environment, glassLayers);
      gl.useProgram(shader.p);
      gl.bindVertexArray(vao);
      gl.activeTexture(gl.TEXTURE0 + unit);
      shader.set({
        u_sprite: unit,
        u_view: view,
        u_proj: projection,
        u_eye: eye,
        u_back: back,
        u_ortho: ortho,
        u_sun: sun?.direction ?? [0, 1, 0],
        u_sunGain: sun ? [sun.ambient, sun.intensity] : [1, 0],
        u_artScale: artScale,
        u_range: range,
        u_fog: fog,
        u_surfaceViewport: [width, height],
        u_index: !!index,
        u_markPass:!!anchors,
        u_historyOwner:!!historyOwner,
        u_filterMatte:!!coverageFilter,
        u_glassBoundary: 13,
        u_graphDepth: !!graphDepth,
        u_portalCover: !!portalCover,
        u_arrivalPlane: arrivalPlane,
        u_splatPlane: clip,
        ...water,
        ...section,
        ...environment?.uniforms
      });
      for (const row of rows) {
        const { placement, at, axes, normalAxes, phase, step, cutoutLevel, reprojection } = row;
        const tile = reprojection?.view ?? row.view;
        const mip = !reprojection && placement.atlas.cutoutImage?.(tile, cutoutLevel);
        const image = reprojection?.image ?? (mip || api.tile(row));
        texture(image);
        shader.set({
          u_anchor: at,
          u_right: axes.right,
          u_up: axes.up,
          u_plane: axes.normal,
          u_size: [tile.rect.width, tile.rect.height],
          u_pivot: [tile.pivot.x, tile.pivot.y],
          u_units: tile.unitsPerPixel * placement.scale,
          u_scale: placement.scale,
          u_phase: phase,
          u_filterSoftness:coverageFilter?.at(placement.material) ?? 1,
          u_step: step,
          u_depth: placement.depth,
          u_normal: !!sun && placement.lighting,
          u_emission: placement.emission,
          u_baked: placement.atlas.emissionMode === 'baked',
          u_normalAxes: [...normalAxes.right, ...normalAxes.up, ...normalAxes.normal],
          u_effect: false,
          u_effectDepth: 0
        });
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        const effectDepth = mip ? placement.atlas.effect(tile)?.depth : image.effectDepth;
        if (effectDepth !== undefined) {
          if (mip) texture(api.tile(row));
          shader.set({ u_effect: true, u_effectDepth: effectDepth });
          gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        }
      }
      gl.activeTexture(gl.TEXTURE0);
    },
    dispose: () => resources.dispose()
  };
}
