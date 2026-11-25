import * as shared from '../surface-history.js';
export const surfaceHistory=shared.default??globalThis[Symbol.for('ink-studio.surface-history')];

const shaderSource=`#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_current,u_currentOwner,u_currentDepth,u_previous,u_previousOwner,u_previousDepth;
uniform mat4 u_inverseView,u_oldView,u_oldProjection;
uniform vec2 u_depthRange,u_oldDepthRange,u_oldClip;
uniform vec4 u_camera; // tangent, aspect, orthographic, previous tangent
uniform float u_weight,u_tolerance;
uniform bool u_valid,u_hold;
layout(location=0) out vec4 color;
layout(location=1) out vec4 owner;
layout(location=2) out vec4 depth;
layout(location=3) out vec4 receipt;
bool validDepth(vec4 packed){
  float sum=packed.r+packed.g+packed.b;
  return packed.a>.5&&sum>0.&&sum<3.;
}
float unpackDepth(vec4 p,vec2 range){
  return mix(range.x,range.y,dot(round(p.rgb*255.),vec3(1,256,65536))/16777215.);
}
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy),size=textureSize(u_current,0);
  color=texelFetch(u_current,p,0);
  owner=texelFetch(u_currentOwner,p,0);
  depth=texelFetch(u_currentDepth,p,0);
  receipt=u_hold?color:vec4(0,0,0,1);
  if(!u_valid)return;
  if(owner.a<.5||!validDepth(depth)){receipt.r=2./255.;return;}
  float z=unpackDepth(depth,u_depthRange);
  vec2 ndc=(vec2(p)+.5)/vec2(size)*2.-1.;
  vec3 local=vec3(ndc*u_camera.x*vec2(u_camera.y,1.)*(u_camera.z>.5?1.:z),-z);
  vec4 old=u_oldView*(u_inverseView*vec4(local,1.)), clip=u_oldProjection*old;
  float expected=-old.z;
  if(expected<u_oldClip.x||expected>u_oldClip.y||clip.w<=0.){receipt.r=3./255.;return;}
  ivec2 q=ivec2(floor((clip.xy/clip.w*.5+.5)*vec2(size)));
  if(any(lessThan(q,ivec2(0)))||any(greaterThanEqual(q,size))){receipt.r=3./255.;return;}
  vec4 priorOwner=texelFetch(u_previousOwner,q,0);
  if(priorOwner.a<.5||any(notEqual(round(priorOwner.rgb*255.),round(owner.rgb*255.)))){receipt.r=4./255.;return;}
  vec4 past=texelFetch(u_previousDepth,q,0);
  float units=2.*u_camera.w*(u_camera.z>.5?1.:expected)/float(size.y);
  if(!validDepth(past)||abs(unpackDepth(past,u_oldDepthRange)-expected)>max(.0001,u_tolerance*units)){
    receipt.r=5./255.;return;
  }
  vec4 prior=texelFetch(u_previous,q,0);
  color=u_hold?vec4(prior.rgb,color.a):mix(color,prior,u_weight);
  if(!u_hold)receipt.r=1./255.;
}`;

export function createSurfaceHistory(gl,{target,program,textureUnit,fullscreen}) {
  const buffers=[target(4),target(4)], shader=program(shaderSource);
  let previous,index=0,bytes;
  let atlasIds=new WeakMap(), nextId=1, layoutCount;
  return {
    clear(){previous=undefined;},
    labels(count){
      if(layoutCount!==count||nextId>0xc00000){
        previous=undefined;layoutCount=count;atlasIds=new WeakMap();nextId=count+1;
      }
      return atlas=>{
        const triangles=atlas.vertices.length/15, prior=atlasIds.get(atlas);
        if(prior?.triangles===triangles)return prior.first;
        if(nextId+triangles>0xffffff)throw new RangeError('Surface history exceeds its 24-bit owner budget');
        const first=nextId;
        atlasIds.set(atlas,{first,triangles});
        nextId+=triangles;
        return first;
      };
    },
    resolve(color,owner,frame,settings,inverseView,{cut=false,capture=false,hold=false}={}){
      const {w,h,camera}=frame, reset=surfaceHistory.resets(previous,frame,settings,cut);
      const old=previous?.camera??camera;
      const buffer=buffers[1-index], prior=buffers[index];
      buffer.resize(w,h);
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      textureUnit(1,color);textureUnit(2,owner.textures[1]);textureUnit(3,owner.textures[2]);
      textureUnit(5,reset?color:prior.textures[hold?3:0]);
      textureUnit(6,reset?owner.textures[1]:prior.textures[1]);
      textureUnit(7,reset?owner.textures[2]:prior.textures[2]);
      const dither=gl.isEnabled(gl.DITHER);
      gl.disable(gl.DITHER);
      fullscreen(shader,buffer.framebuffer,w,h,{
        u_current:1,u_currentOwner:2,u_currentDepth:3,u_previous:5,u_previousOwner:6,u_previousDepth:7,
        u_inverseView:inverseView,u_oldView:old.view,u_oldProjection:old.projection,
        u_depthRange:camera.depthRange,u_oldDepthRange:old.depthRange,u_oldClip:[old.clipNear,old.clipFar],
        u_camera:[camera.tangent,camera.aspect,+camera.ortho,old.tangent],u_valid:!reset,u_hold:hold,
        u_weight:settings.weight,u_tolerance:settings.depthPixels
      });
      if(dither)gl.enable(gl.DITHER);
      previous={...frame,key:settings.key};
      index=1-index;
      if(!capture)return {texture:buffer.textures[0]};
      if(bytes?.length!==w*h*4)bytes=new Uint8Array(w*h*4);
      gl.readBuffer(gl.COLOR_ATTACHMENT3);
      gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,bytes);
      const mask=Uint8Array.from({length:w*h},(_,p)=>bytes[((h-1-Math.floor(p/w))*w+p%w)*4]);
      return {texture:buffer.textures[0],report:surfaceHistory.summary(mask,w,h,reset)};
    }
  };
}
