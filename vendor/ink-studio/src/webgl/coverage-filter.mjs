/* Composite subpixel samples in RGB on the art grid, ahead of palette and tone shaders. */
import * as shared from '../coverage-filter.js';

export const coverageFilter = shared.default ?? globalThis[Symbol.for('ink-studio.coverage-filter')];
export const filterUnit = 31;

export const filterMatte = `
uniform bool u_filterMatte;
uniform highp sampler2D u_filterRows;
float filterSoftness(int row){
  return texelFetch(u_filterRows,ivec2((row>>2)&511,row>>11),0)[row&3];
}
`;

const source = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_center,u_taps,u_centerMatte,u_tapMatte;
uniform bool u_hasMatte;
out vec4 color;
void main(){
  ivec2 pixel=ivec2(gl_FragCoord.xy);
  vec4 center=texelFetch(u_center,pixel,0);
  float softness=u_hasMatte ? texelFetch(u_centerMatte,pixel,0).r : 1.;
  vec4 result=center;
  for(int i=0;i<4;i++){
    ivec2 tap=2*pixel+ivec2(i&1,i>>1);
    float weight=u_hasMatte ? min(softness,texelFetch(u_tapMatte,tap,0).r) : 1.;
    result+=.25*weight*(texelFetch(u_taps,tap,0)-center);
  }
  color=result;
}`;

export function createCoverageFilter({ target, table, maxSize, program, textureUnit, fullscreen }) {
  const taps = target(2), filtered = target(), shader = program(source);
  let centerMatte, tapMatte, sampleDepth, rows, lastData, lastMetadata, lastSettings, lastCount;
  return {
    taps, filtered,
    prepare(world, settings, width, height, curves) {
      taps.resize(width * 2,height * 2);
      filtered.resize(width,height);
      if (curves) {
        sampleDepth ??= target(3);
        sampleDepth.resize(width * 2,height * 2);
      }
      if (settings.varied) {
        centerMatte ??= target(2);
        tapMatte ??= target(2);
        centerMatte.resize(width,height);
        tapMatte.resize(width * 2,height * 2);
        rows ??= table(512,filterUnit,maxSize);
        const { materialIds, materials } = world.renderMetadata ?? {};
        if (lastMetadata !== world.renderMetadata || lastSettings !== settings || lastCount !== world.count) {
          if (lastData?.length !== world.count) lastData = new Float32Array(world.count);
          for (let i = 0; i < world.count; i++) lastData[i] = settings.at(materials?.[materialIds?.[i]]);
          rows.upload(lastData);
          lastMetadata = world.renderMetadata;
          lastSettings = settings;
          lastCount = world.count;
        }
      }
      return { centerMatte: settings.varied && centerMatte, tapMatte: settings.varied && tapMatte,
        sampleDepth: curves && sampleDepth, bindRows: () => rows?.bind() };
    },
    finish(center, mattes) {
      textureUnit(1,center);
      textureUnit(2,taps.textures[0]);
      textureUnit(5,mattes.centerMatte ? mattes.centerMatte.textures[0] : center);
      textureUnit(6,mattes.tapMatte ? mattes.tapMatte.textures[0] : center);
      fullscreen(shader,filtered.framebuffer,filtered.width,filtered.height,
        {u_center:1,u_taps:2,u_centerMatte:5,u_tapMatte:6,u_hasMatte:!!mattes.centerMatte});
    }
  };
}
