import { surfaceHistory } from './surface-history.mjs';

// Only matching current-frame taps can win. Owner, packed depth and the chosen
// tap travel together. The previous frame contributes evidence, never color.
const decisionSource = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_native,u_depth,u_taps,u_tapDepth,u_past,u_pastDepth,u_rows,u_locks;
uniform mat4 u_inverseView,u_oldView,u_oldProjection;
uniform vec4 u_camera; // tangent, aspect, orthographic, old tangent
uniform vec2 u_depthRange,u_oldDepthRange,u_oldClip;
uniform ivec4 u_region;
uniform int u_count;
uniform bool u_history,u_hasLocks;
uniform float u_temporal,u_spatial,u_depthPixels,u_maxMotion;
layout(location=0) out vec4 owner;
layout(location=1) out vec4 decision;
layout(location=2) out vec4 depth;

int rowOf(vec4 id) { return int(dot(round(id.rgb*255.),vec3(1,256,65536)))-1; }
bool inGrid(ivec2 p,ivec2 size) { return all(greaterThanEqual(p,ivec2(0)))&&all(lessThan(p,size)); }
bool inRegion(ivec2 p) { return all(greaterThanEqual(p,u_region.xy))&&all(lessThan(p,u_region.xy+u_region.zw)); }
bool locked(ivec2 p) { return u_hasLocks && texelFetch(u_locks,p,0).r>.5; }
vec4 rowData(int row) {
  return row>=0&&row<u_count ? texelFetch(u_rows,ivec2(row&511,row>>9),0) : vec4(0);
}
int labelOf(int row,vec4 packed) {
  return row>=0 ? int(rowData(row).x) : packed.a>.5 ? -1 : 0;
}
float depthOf(vec4 packed,vec2 range) {
  return mix(range.x,range.y,dot(round(packed.rgb*255.),vec3(1,256,65536))/16777215.);
}
bool reproject(vec2 position,float z,vec3 move,ivec2 size,out ivec2 q,out float expected,out float tolerance) {
  vec2 ndc=position/vec2(size)*2.-1.;
  vec3 local=vec3(ndc*u_camera.x*vec2(u_camera.y,1.)*(u_camera.z>.5?1.:z),-z);
  vec4 old=u_oldView*(u_inverseView*vec4(local,1.)+vec4(move,0.));
  vec4 clip=u_oldProjection*old;
  expected=-old.z;
  if (clip.w<=0. || expected<u_oldClip.x || expected>u_oldClip.y) return false;
  vec2 previous=(clip.xy/clip.w*.5+.5)*vec2(size);
  q=ivec2(floor(previous));
  tolerance=max(.0001,u_depthPixels*2.*u_camera.w*(u_camera.z>.5?1.:expected)/float(size.y));
  return inGrid(q,size)&&inRegion(q)&&!locked(q)&&length(previous-position)<=u_maxMotion;
}

void main() {
  ivec2 p=ivec2(gl_FragCoord.xy), size=textureSize(u_native,0);
  owner=texelFetch(u_native,p,0); depth=texelFetch(u_depth,p,0);
  decision=vec4(0);
  if (!inRegion(p)) return;
  int center=labelOf(rowOf(owner),depth);
  decision.ba=vec2(max(0,center))/255.;
  if (locked(p)) return;
  int labels[4], rows[4], votes[4];
  vec4 ids[4], depths[4];
  bool selected=center>0, differs=false;
  for (int i=0;i<4;i++) {
    ivec2 tap=p*2+ivec2(i&1,i>>1);
    ids[i]=texelFetch(u_taps,tap,0); depths[i]=texelFetch(u_tapDepth,tap,0);
    rows[i]=rowOf(ids[i]); labels[i]=labelOf(rows[i],depths[i]);
    selected=selected||labels[i]>0; differs=differs||labels[i]!=center;
  }
  if (!selected||!differs) return;
  for (int i=0;i<4;i++) {
    votes[i]=0;
    for (int j=0;j<4;j++) if (labels[i]==labels[j]) votes[i]++;
  }

  // A 3:1 or 4:0 result is new evidence. In particular, a newly covered
  // pixel or a newly exposed gap can take effect without waiting for history.
  for (int i=0;i<4;i++) if (votes[i]>=3) {
    if (labels[i]!=center) {
      owner=ids[i]; depth=depths[i]; decision.rg=vec2(float(i+1),4.)/255.;
      decision.a=float(max(0,labels[i]))/255.;
    }
    return;
  }

  ivec4 neighbors=ivec4(999999);
  for (int i=0;i<4;i++) {
    ivec2 q=p+ivec2(i==0?-1:i==1?1:0,i==2?-1:i==3?1:0);
    if (inGrid(q,size)) neighbors[i]=labelOf(rowOf(texelFetch(u_native,q,0)),texelFetch(u_depth,q,0));
  }
  // The nearest visible patch provides one shared motion anchor for both
  // sides. Giving every tap its own history vote can find both old owners and
  // cancel the hold at exactly the edges that need it.
  int favored=-999999;
  if (u_history && u_temporal>0.) {
    int anchor=-1;
    for (int i=0;i<4;i++) if (rows[i]>=0&&depths[i].a>.5) {
      if (anchor<0||depthOf(depths[i],u_depthRange)<depthOf(depths[anchor],u_depthRange)) anchor=i;
    }
    ivec2 q;
    float expected,tolerance;
    vec2 position=vec2(p)+.5;
    if (anchor>=0 && reproject(position,depthOf(depths[anchor],u_depthRange),rowData(rows[anchor]).yzw,size,q,expected,tolerance)) {
      vec4 oldDepth=texelFetch(u_pastDepth,q,0);
      int old=labelOf(rowOf(texelFetch(u_past,q,0)),oldDepth);
      if (old==0 && oldDepth.a<.5) favored=0;
      else for (int i=0;i<4;i++) if (rows[i]>=0 && labels[i]==old && depths[i].a>.5 && oldDepth.a>.5) {
        ivec2 projected;
        if (reproject(position,depthOf(depths[i],u_depthRange),rowData(rows[i]).yzw,size,projected,expected,tolerance) &&
            all(equal(projected,q)) && abs(depthOf(oldDepth,u_oldDepthRange)-expected)<=tolerance) favored=old;
      }
    }
  }
  int best=0,fresh=0;
  float score=-1.,freshScore=-1.;
  for (int i=0;i<4;i++) {
    float adjacent=dot(vec4(equal(neighbors,ivec4(labels[i]))),vec4(1.));
    float base=float(votes[i])+u_spatial*adjacent+(labels[i]==center?.20:0.);
    if (base>freshScore) {freshScore=base;fresh=i;}
    float cost=base+(labels[i]==favored?u_temporal:0.);
    if (cost>score) {score=cost;best=i;}
  }
  if (labels[best]==center) { decision.g=1./255.; return; }
  owner=ids[best]; depth=depths[best];
  decision.rg=vec2(float(best+1),labels[best]!=labels[fresh] ? 3. : 2.)/255.;
  decision.a=float(max(0,labels[best]))/255.;
}
`;

const applySource = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_decision,u_color,u_aux,u_tapColor,u_tapAux;
uniform bool u_high;
layout(location=0) out vec4 color;
layout(location=1) out vec4 aux;
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  ivec2 native=u_high?p/2:p;
  int tap=int(round(texelFetch(u_decision,native,0).r*255.))-1;
  if (tap<0) {
    color=texelFetch(u_color,p,0); aux=texelFetch(u_aux,p,0);
  } else {
    ivec2 q=native*2+ivec2(tap&1,tap>>1);
    color=texelFetch(u_tapColor,q,0); aux=texelFetch(u_tapAux,q,0);
  }
}
`;

export function createLivePixelBoundaries(gl,{api,kit,target,table,program,textureUnit,fullscreen,maxSize,resources}) {
  const sampleColor=target(2), sampleOwner=target(3), applied=target(2);
  let high;
  const decisions=[target(3),target(3)], judge=program(decisionSource), apply=program(applySource);
  const rows=table(512,8,maxSize), lockTexture=resources.texture();
  const status=['untouched','ambiguous','spatial','temporal','decisive'];
  let previous, index=0, packing, positionBuffers, members, metadata, settingsKey, count;
  let receipt,lockBytes;

  function prepare(world,settings) {
    if (metadata!==world.renderMetadata||settingsKey!==settings.key||count!==world.count) {
      members=api.memberships(world,settings);
      packing=new Float32Array(world.count*4);
      positionBuffers=[new Float32Array(world.count*3),new Float32Array(world.count*3)];
      metadata=world.renderMetadata; settingsKey=settings.key; count=world.count;
      previous=undefined;
    }
    return members.selected.some(Boolean);
  }

  function resolve(nativeColor,owner,world,frame,settings,region,inverseView,poseTime,{cut,locks,capture,highColor}) {
    const {w,h,camera}=frame;
    const reset=surfaceHistory.resets(previous,frame,settings,cut) ??
      (previous.output!==frame.output ? 'view' : null);
    const old=reset ? camera : previous.camera;
    const pose=kit.splatFrames.create(world,poseTime,kit.M);
    const next=1-index, positions=positionBuffers[next], prior=positionBuffers[index];
    const {data}=world;
    for (let i=0,j=0,k=0;i<count;i++,j+=4,k+=3) {
      packing[j]=members.groups[i];
      const at=i*16, kind=Math.round(data[at+11])&15;
      if (kind && kind!==1 && kind!==4 && kind!==7) positions.set(pose(i).position(),k);
      else for(let axis=0;axis<3;axis++) positions[k+axis]=data[at+axis];
      for(let axis=0;axis<3;axis++) packing[j+1+axis]=reset?0:prior[k+axis]-positions[k+axis];
    }
    rows.upload(packing);
    const hasLocks=!!locks?.size;
    if(hasLocks) {
      if (lockBytes?.length!==w*h*4) lockBytes=new Uint8Array(w*h*4);
      else lockBytes.fill(0);
      for (const [x,y] of locks.values()) lockBytes[((h-1-y)*w+x)*4]=255;
      textureUnit(10,lockTexture);
      gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,lockBytes);
    } else textureUnit(10,owner.textures[0]);
    const chosen=decisions[next], past=reset?owner:decisions[index];
    const taps=highColor??sampleColor;
    chosen.resize(w,h); applied.resize(w,h);
    if (highColor) {
      high??=target(2);
      high.resize(w*2,h*2);
    }
    textureUnit(1,owner.textures[0]); textureUnit(2,owner.textures[2]);
    textureUnit(3,sampleOwner.textures[0]); textureUnit(4,sampleOwner.textures[2]);
    textureUnit(5,past.textures[0]); textureUnit(6,past.textures[2]);
    gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    const dither=gl.isEnabled(gl.DITHER);
    gl.disable(gl.DITHER);
    fullscreen(judge,chosen.framebuffer,w,h,{
      u_native:1,u_depth:2,u_taps:3,u_tapDepth:4,u_past:5,u_pastDepth:6,u_rows:8,u_locks:10,
      u_hasLocks:hasLocks,u_count:count,u_region:region,u_history:!reset,
      u_inverseView:inverseView,u_oldView:old.view,u_oldProjection:old.projection,
      u_camera:[camera.tangent,camera.aspect,+camera.ortho,old.tangent],u_oldClip:[old.clipNear,old.clipFar],
      u_depthRange:camera.depthRange,u_oldDepthRange:old.depthRange,u_temporal:settings.temporal,
      u_spatial:settings.spatial,u_depthPixels:settings.depthPixels,u_maxMotion:settings.maxMotion
    });
    textureUnit(1,chosen.textures[1]);
    textureUnit(4,taps.textures[0]); textureUnit(5,taps.textures[1]);
    const copy=(destination,source,isHigh)=>{
      textureUnit(2,source.textures[0]); textureUnit(3,source.textures[1]);
      fullscreen(apply,destination.framebuffer,destination.width,destination.height,{
        u_decision:1,u_color:2,u_aux:3,u_tapColor:4,u_tapAux:5,u_high:isHigh
      });
    };
    copy(applied,nativeColor,false);
    if (highColor) copy(high,highColor,true);
    if (dither) gl.enable(gl.DITHER);
    previous={...frame,key:settings.key};
    index=next;
    if (capture) {
      if(receipt?.length!==w*h*4) receipt=new Uint8Array(w*h*4);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER,chosen.framebuffer);
      gl.readBuffer(gl.COLOR_ATTACHMENT1);
      gl.readPixels(0,0,w,h,gl.RGBA,gl.UNSIGNED_BYTE,receipt);
      const counts=Object.fromEntries(status.map(name=>[name,0]));
      const mask=new Uint8Array(w*h), nativeFeature=new Uint8Array(w*h), resolvedFeature=new Uint8Array(w*h);
      for (let p=0;p<mask.length;p++) {
        const at=((h-1-Math.floor(p/w))*w+p%w)*4;
        counts[status[mask[p]=receipt[at+1]]]++;
        nativeFeature[p]=receipt[at+2];
        resolvedFeature[p]=receipt[at+3];
      }
      capture({w,h,reset,counts,mask,nativeFeature,resolvedFeature});
    }
    return {geometry:highColor ? high : applied,contours:applied,owner:chosen};
  }
  return {sampleColor,sampleOwner,prepare,resolve,clear(){previous=undefined;}};
}
