import { waterGLSL } from './water-optics.mjs';
import { environmentMediaSource } from './environment-shaders.mjs';
import { surfaceEngravingFragment } from './engraving-shaders.mjs';

const rays = `
const float farRay = 1e20;
bool clipRay(vec4 plane, vec3 origin, vec3 ray, inout vec2 interval) {
  float offset = dot(plane,vec4(origin,1)), slope = dot(plane.xyz,ray);
  if (abs(slope) < 1e-8) return offset >= -1e-7;
  if (slope > 0.0) interval.x = max(interval.x,-offset/slope);
  else interval.y = min(interval.y,-offset/slope);
  return interval.x <= interval.y;
}
bool projectedCap(vec4 plane, vec3 point, vec3 origin, vec3 ray) {
  float facing = dot(plane.xyz,ray);
  return abs(facing) >= 1.0-1e-7 ||
    dot(plane,vec4(origin,1))-facing*dot(origin-point,ray) >= -1e-7;
}
float sphereEntry(vec3 center, vec3 ray, float radius, float near) {
  float at = dot(center,ray);
  vec3 offset = center-at*ray;
  float disc = radius*radius-dot(offset,offset);
  if (disc < 0.0) return farRay;
  float reach = sqrt(disc);
  return at+reach < near ? farRay : max(near,at-reach);
}
float capsuleEntry(vec3 start, vec3 ray, vec3 axis, float span, float radius, float near) {
  float shift = dot(start,axis), tilt = dot(ray,axis);
  float first = min(sphereEntry(start,ray,radius,near),sphereEntry(start+span*axis,ray,radius,near));
  if (first == near) return near;
  vec3 origin = start-shift*axis, direction = ray-tilt*axis;
  float a = dot(direction,direction), b = dot(origin,direction), c = dot(origin,origin)-radius*radius;
  float along = near*tilt-shift;
  if (along >= 0.0 && along <= span && a*near*near-2.0*b*near+c <= 0.0) return near;
  float disc = b*b-a*c;
  if (a > 1e-15 && disc >= 0.0) {
    float t = (b-sqrt(disc))/a;
    along = t*tilt-shift;
    if (t >= near && along >= 0.0 && along <= span) first = min(first,t);
  }
  return first;
}
float boxEntry(vec3 low, vec3 high, vec3 ray, float near) {
  float enter = near, leave = farRay;
  for (int k=0; k<3; k++) {
    if (abs(ray[k]) < 1e-8) {
      if (low[k] > 0.0 || high[k] < 0.0) return farRay;
    } else {
      float a = low[k]/ray[k], b = high[k]/ray[k];
      enter = max(enter,min(a,b)); leave = min(leave,max(a,b));
      if (enter > leave) return farRay;
    }
  }
  return enter;
}
float hardEntry(vec3 start, vec3 ray, vec3 along, float span, float radius, bool flattened, float near) {
  if (!flattened) return capsuleEntry(start,ray,along,span,radius,near);
  float halfWidth = 3.0*radius;
  vec3 low = start-vec3(0,halfWidth,0), high = start+vec3(0,halfWidth,0);
  float enter = boxEntry(low-vec3(0,0,radius),high+vec3(span,0,radius),ray,near);
  enter = min(enter,capsuleEntry(low,ray,along,span,radius,near));
  enter = min(enter,capsuleEntry(high,ray,along,span,radius,near));
  vec3 crossAxis = vec3(0,1,0);
  enter = min(enter,capsuleEntry(low,ray,crossAxis,2.0*halfWidth,radius,near));
  return min(enter,capsuleEntry(low+vec3(span,0,0),ray,crossAxis,2.0*halfWidth,radius,near));
}
float erfCurve(float x) {
  float t = 1.0/(1.0+.3275911*abs(x));
  float p = ((((1.061405429*t-1.453152027)*t+1.421413741)*t-.284496736)*t+.254829592)*t;
  return sign(x)*(1.0-p*exp(-x*x));
}
float primitive(float x) {
  float s = x*x;
  return (x*sqrt(max(0.0,1.0-s))*(33.0-26.0*s+8.0*s*s)+15.0*asin(x))/48.0;
}
// The result is optical mass and its representative station in world units.
vec2 integrated(vec3 start, vec3 tangent, float span, float radius, bool beta) {
  float transverse = length(tangent), reach = radius*(beta ? sqrt(6.0) : sqrt(2.0));
  if (transverse*span < 1e-4*reach) {
    float at = span*.5, q = dot(start+at*tangent,start+at*tangent)/(reach*reach);
    float profile = beta ? pow(max(0.0,1.0-q),2.5) : exp(-q);
    return vec2(profile*span/(beta ? reach*.9817477042 : radius*2.5066282746),at);
  }
  float center = -dot(start,tangent)/(transverse*transverse);
  vec3 delta = start+center*tangent;
  float q = dot(delta,delta)/(reach*reach), at = clamp(center,0.0,span);
  if (beta) {
    if (q >= 1.0) return vec2(0);
    float scale = transverse/(reach*sqrt(1.0-q));
    float lo = max(-1.0,-center*scale), hi = min(1.0,(span-center)*scale);
    return vec2(hi > lo ? pow(1.0-q,3.0)*(primitive(hi)-primitive(lo))/(transverse*.9817477042) : 0.0,at);
  }
  float lo = -center*transverse/reach, hi = (span-center)*transverse/reach;
  if (q > 12.96 || lo >= 4.0 || hi <= -4.0) return vec2(0);
  return vec2(exp(-q)*(erfCurve(hi)-erfCurve(lo))/(2.0*transverse),at);
}
const float circlePoints[7] = float[7](-.9491079123,-.7415311856,-.4058451514,0.0,.4058451514,.7415311856,.9491079123);
const float circleMass[7] = float[7](.1294849662,.2797053915,.3818300505,.4179591837,.3818300505,.2797053915,.1294849662);

uniform highp sampler2D u_curveSpans;
vec4 spanRow(int id, int field) {
  int texel = id*11+field;
  return texelFetch(u_curveSpans,ivec2(texel & 511,texel >> 9),0);
}
struct Hit { int row; float depth; float exitDepth; float alpha; float group; };
bool curveHit(int id, vec3 origin, vec3 ray, vec2 viewInterval, out Hit hit) {
  vec4 a = spanRow(id,0), b = spanRow(id,1), major = spanRow(id,2), minor = spanRow(id,3);
  vec4 capStart = spanRow(id,4), capEnd = spanRow(id,5), meta = spanRow(id,7);
  int mode = int(major.w);
  bool hard = (mode & 2) != 0, flattened = (mode & 8) != 0, beta = (mode & 1) != 0;
  if (hard) {
    if (!clipRay(capStart,origin,ray,viewInterval) || !clipRay(capEnd,origin,ray,viewInterval)) return false;
  } else if (!projectedCap(capStart,spanRow(id,8).xyz,origin,ray) || !projectedCap(capEnd,spanRow(id,9).xyz,origin,ray)) return false;
  vec3 from = a.xyz-origin, start = from, axis = b.xyz, along = axis, direction = ray;
  float axial = -ray.z, scale = 1.0;
  if ((mode & 4) != 0) {
    float blur = flattened ? .25 : 1.0;
    start = vec3(dot(axis,from),dot(major.xyz,from)/blur,dot(minor.xyz,from));
    direction = vec3(dot(axis,ray),dot(major.xyz,ray)/blur,dot(minor.xyz,ray));
    scale = length(direction);
    direction /= scale;
    along = vec3(1,0,0);
  }
  float depth, exitDepth, alpha;
  if (hard) {
    float pos = hardEntry(start,direction,along,b.w,a.w,flattened,viewInterval.x*scale)/scale;
    if (pos > viewInterval.y || pos >= farRay*.5) return false;
    float exit = -hardEntry(start,-direction,along,b.w,a.w,flattened,-farRay)/scale;
    depth = pos*axial;
    exitDepth = max(pos,min(exit,viewInterval.y))*axial;
    alpha = minor.w;
  } else {
    float offset = dot(start,direction), tilt = dot(along,direction);
    vec3 perpendicular = start-offset*direction, tangent = along-tilt*direction;
    float base = dot(from,ray), slope = dot(axis,ray);
    if (flattened) {
      float sum = 0.0, weighted = 0.0, lateral = dot(major.xyz,ray);
      for (int i=0; i<7; i++) {
        float shift = a.w*circlePoints[i], whitened = shift*4.0;
        vec3 off = vec3(0,whitened,0)-whitened*direction.y*direction;
        vec2 seg = integrated(perpendicular+off,tangent,b.w,a.w,beta);
        float mass = seg.x*circleMass[i];
        sum += mass;
        weighted += mass*(base+shift*lateral+seg.y*slope);
      }
      if (sum <= 0.0) return false;
      depth = weighted/sum;
      alpha = 1.0-exp(-meta.y*sum*(beta ? 35.0/(8.0*sqrt(6.0)) : 1.5957691216)/scale);
    } else {
      vec2 seg = integrated(perpendicular,tangent,b.w,a.w,beta);
      depth = base+seg.y*slope;
      alpha = 1.0-exp(-meta.y*seg.x/scale);
    }
    depth = max(depth,u_curveNear/axial);
    if (depth < viewInterval.x-1e-5 || depth > viewInterval.y) return false;
    depth *= axial;
    exitDepth = depth;
  }
  hit = Hit(id,depth,exitDepth,alpha,meta.x);
  return alpha >= 1e-5;
}`;

export function finiteCurveFragment({ tileSize, layers, mode, wet, layeredGlass, environment, section, spriteLOD, strandLOD, engraving, print, pbr }) {
  const owner=mode==='owner', mask=mode==='mask', anchors=mode==='anchors', color=mode==='color';
  return `#version 300 es
precision highp float;
precision highp int;
uniform vec2 u_curveViewport,u_curveInvProjection,u_curveRange;
uniform vec4 u_curveEntry,u_curveCover;
uniform mat4 u_curveView,u_curveProjection;
uniform vec3 u_curveFog;
uniform vec2 u_curveFogRange;
uniform float u_curveNear,u_curveFar,u_curveSetback;
uniform bool u_curveOrtho,u_curveHasProxy,u_filterMatte;
uniform float u_curveArtScale;
uniform highp sampler2D u_curveTiles,u_curveSolid,u_curveProxy;
uniform int u_curveTileColumns;
${owner ? 'uniform sampler2D u_curveOwner,u_curveOwnerMask;uniform bool u_historyOwner;' : ''}
${anchors ? 'uniform highp sampler2D u_curveBaseWorld,u_curveBaseRest;' : ''}
${color && wet ? waterGLSL(wet.airPockets,wet.layered,wet.atlas) : ''}
${color && environment?.media ? environmentMediaSource(environment.lit) : ''}
${section ?? ''}
${spriteLOD ?? ''}
${strandLOD ?? ''}
${engraving ? surfaceEngravingFragment(print) : ''}
layout(location=0) out vec4 o_color;
${!mask ? 'layout(location=1) out vec4 o_depth;' : ''}
${owner || color && wet ? 'layout(location=2) out vec4 o_aux;' : ''}
${color && layeredGlass ? `uniform sampler2D u_curveGlass;
layout(location=${wet ? 3 : 2}) out vec4 o_back;
layout(location=${wet ? 4 : 3}) out vec4 o_through;` : ''}
${rays}
${pbr ? pbr+`
float curveRadialEntry(vec3 start,vec3 ray,vec3 axis,float radius,float near){
  vec3 offset=start-axis*dot(start,axis), direction=ray-axis*dot(ray,axis);
  float a=dot(direction,direction),b=dot(offset,direction);
  float disc=b*b-a*(dot(offset,offset)-radius*radius);
  if(a<1e-20||disc<0.0)return farRay;
  float reach=sqrt(disc);
  return (b+reach)/a<near?farRay:max(near,(b-reach)/a);
}
vec3 curvePbrNormal(int id,vec3 origin,vec3 ray,float distance){
  vec4 a=spanRow(id,0),b=spanRow(id,1),major=spanRow(id,2),minor=spanRow(id,3);
  int mode=int(major.w);
  bool hard=(mode&2)!=0, elliptical=(mode&4)!=0, ribbon=(mode&8)!=0;
  float blur=ribbon?.25:1.0;
  vec3 axis=b.xyz,start=a.xyz-origin,direction=ray;
  if(elliptical){
    start=vec3(dot(axis,start),dot(major.xyz,start)/blur,dot(minor.xyz,start));
    direction=vec3(dot(axis,ray),dot(major.xyz,ray)/blur,dot(minor.xyz,ray));
    float scale=length(direction);
    direction/=scale;
    distance*=scale;
    axis=vec3(1,0,0);
  }
  float wing=ribbon?a.w*(1.0-blur)/blur:0.0;
  if(!hard){
    float near=u_curveNear/(-ray.z)*(elliptical?length(vec3(dot(b.xyz,ray),dot(major.xyz,ray)/blur,dot(minor.xyz,ray))):1.0);
    if(ribbon){
      float center=abs(direction.z)>1e-12?(start.z-sign(direction.z)*a.w)/direction.z:farRay;
      if(center<near||abs(direction.y*center-start.y)>wing)center=farRay;
      distance=min(center,min(curveRadialEntry(start-vec3(0,wing,0),direction,axis,a.w,near),
        curveRadialEntry(start+vec3(0,wing,0),direction,axis,a.w,near)));
    }else distance=curveRadialEntry(start,direction,axis,a.w,near);
    if(distance>=farRay*.5)return -ray;
  }
  vec3 offset=distance*direction-start;
  float along=dot(offset,axis);
  vec3 delta=offset-axis*(hard?clamp(along,0.0,b.w):along);
  if(ribbon)delta.y-=clamp(delta.y,-wing,wing);
  vec3 normal=elliptical?b.xyz*delta.x+major.xyz*delta.y/blur+minor.xyz*delta.z:delta;
  return pbrUnit(normal,-ray);
}
` : ''}
${anchors || engraving ? `
uniform highp sampler2D u_curveMarks;
vec4 curveMark(int id,int field){
  int at=id*6+field;return texelFetch(u_curveMarks,ivec2(at&511,at>>9),0);
}
void curveChart(int id,vec3 at,vec3 ray,out vec3 point,out vec3 localNormal,out vec3 normal,out float stretch){
  vec4 c=curveMark(id,0),a=curveMark(id,1),u=curveMark(id,2),p=curveMark(id,3),t=curveMark(id,4),r=curveMark(id,5);
  vec3 q=at-c.xyz,v=cross(a.xyz,u.xyz),s=cross(t.xyz,r.xyz);
  float along=dot(q,a.xyz),ratio=p.w/max(1e-8,c.w);
  point=p.xyz+t.xyz*(along*ratio)+a.w*(r.xyz*dot(q,u.xyz)+s*dot(q,v));
  vec3 radial=q-a.xyz*clamp(along,-c.w,c.w);
  if(dot(radial,radial)<1e-12)radial=-ray;
  radial=normalize(radial);
  localNormal=normalize(t.xyz*dot(radial,a.xyz)+r.xyz*dot(radial,u.xyz)+s*dot(radial,v));
  normal=transpose(mat3(u_curveView))*radial;
  stretch=max(ratio,a.w)/max(.3,abs(dot(radial,ray)));
}` : ''}
vec4 tileRow(int i) { return texelFetch(u_curveTiles,ivec2(i & 511,i >> 9),0); }
float decode(vec3 rgb) { return dot(round(rgb*255.0),vec3(1,256,65536)); }
vec4 encode(float v) { return vec4(mod(v,256.0),mod(floor(v/256.0),256.0),floor(v/65536.0),255.0)/255.0; }
float projectedDepth(float axial) {
  vec4 p = u_curveProjection*vec4(0,0,-axial,1);
  return p.z/p.w*.5+.5;
}
void main() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  vec4 base = texelFetch(u_curveSolid,pixel,0);
  vec2 tile = tileRow((pixel.y/${tileSize})*u_curveTileColumns + pixel.x/${tileSize}).xy;
  ${owner ? 'o_color=texelFetch(u_curveOwner,pixel,0); o_depth=texelFetch(u_curveOwnerMask,pixel,0); o_aux=base;' : ''}
  ${anchors ? 'o_color=texelFetch(u_curveBaseWorld,pixel,0);o_depth=texelFetch(u_curveBaseRest,pixel,0);' : ''}
  if (tile.y <= 0.0) ${owner || anchors ? 'return;' : 'discard;'}
  vec2 film = (gl_FragCoord.xy*2.0/u_curveViewport-1.0)*u_curveInvProjection;
  vec3 origin = u_curveOrtho ? vec3(film,0) : vec3(0);
  vec3 ray = u_curveOrtho ? vec3(0,0,-1) : normalize(vec3(film,-1));
  float opaque = base.a > 0.0 ? mix(u_curveRange.x,u_curveRange.y,decode(base.rgb)/16777215.0) : u_curveFar;
  vec2 interval = vec2(u_curveNear, min(u_curveFar,opaque+1e-4))/(-ray.z);
  if (!clipRay(u_curveEntry,origin,ray,interval) || !clipRay(u_curveCover,origin,ray,interval))
    ${owner || anchors ? 'return;' : 'discard;'}
  float proxyDepth = u_curveHasProxy ? texelFetch(u_curveProxy,pixel,0).r : 1.0;
  ${owner || mask || anchors ? `
  float nearest = opaque+1e-4; int row = -1;
  ` : `
  Hit hits[${layers}];
  int count = 0;
  ${layeredGlass ? 'vec4 glass = texelFetch(u_curveGlass,pixel,0); float pane = decode(glass.rgb)/16777215.0;' : ''}
  `}
  int start = int(tile.x);
  for (int j=0; j<int(tile.y); j++) {
    int k = start+j;
    int id = int(tileRow(k >> 2)[k & 3]);
    ${spriteLOD ? 'if (!spriteGeometry(int(spanRow(id,7).z),gl_FragCoord.xy/u_curveArtScale)) continue;' : ''}
    ${strandLOD ? 'int strandCode=int(spanRow(id,7).w); if (strandCode!=0 && !strandVisible(strandCode,gl_FragCoord.xy/u_curveArtScale)) continue;' : ''}
    Hit hit;
    if (!curveHit(id,origin,ray,interval,hit)) continue;
    float projected = projectedDepth(hit.depth);
    if (proxyDepth < 1.0 && projected > proxyDepth+2e-7) continue;
    ${section ? `
    vec3 at = origin+ray*(hit.depth/(-ray.z));
    vec3 world = transpose(mat3(u_curveView))*(at-u_curveView[3].xyz);
    if (sectionMargin(world, hit.depth-u_curveSetback) < 0.0) continue;
    ` : ''}
    ${owner || mask || anchors ? `
    if (hit.group > 0.0 && hit.alpha >= 1.0 && hit.depth <= nearest) {
      nearest = hit.depth; row = id;
    }` : `
    // Join overlapping hard intervals, but keep distinct crossings of a coiled strand.
    float front = hit.depth, exit = hit.exitDepth;
    if (hit.group > 0.0) {
      for (int i=0; i<count;) {
        Hit previous = hits[i];
        if (previous.group != hit.group || front > previous.exitDepth+1e-6 || previous.depth > exit+1e-6) {
          i++;
          continue;
        }
        if (previous.depth < hit.depth) hit = previous;
        front = min(front,previous.depth);
        exit = max(exit,previous.exitDepth);
        for (int j=i; j<count-1; ++j) hits[j] = hits[j+1];
        count--;
      }
      hit.depth = front; hit.exitDepth = exit;
    }
    // Keep the nearest layers in depth order. Far fragments beyond this budget
    // cannot hide an in-budget crossing.
    int at = count;
    while (at > 0 && hit.depth <= hits[at-1].depth) at--;
    if (at >= ${layers}) continue;
    count = min(count+1,${layers});
    for (int i=count-1; i>at; --i) hits[i] = hits[i-1];
    hits[at] = hit;
    `}
  }
  ${owner ? `
  if (row >= 0) {
    o_color = encode(spanRow(row,6).a);
    o_depth = u_historyOwner ? o_color : vec4(0);
    o_aux = encode(round(clamp((nearest-u_curveRange.x)/(u_curveRange.y-u_curveRange.x),0.0,1.0)*16777215.0));
  }` : anchors ? `
  if(row>=0){
    vec3 at=origin+ray*(nearest/(-ray.z)), rest, rn,n;float footprint;
    curveChart(row,at,ray,rest,rn,n,footprint);
    o_color=vec4(transpose(mat3(u_curveView))*(at-u_curveView[3].xyz),curveMark(row,2).w);
    o_depth=vec4(rest,1);
  }` : mask ? `
  if (row < 0) discard;
  o_color=vec4(1);` : `
  if (count == 0) discard;
  o_color=vec4(0); o_depth=vec4(0);
  ${wet ? 'o_aux=vec4(0);' : ''}
  ${layeredGlass ? 'o_back=vec4(0); o_through=vec4(0);' : ''}
  for (int i=0; i<count; i++) {
    Hit hit = hits[i];
    vec4 pigment = spanRow(hit.row,6);
    float normalized = clamp((hit.depth-u_curveRange.x)/(u_curveRange.y-u_curveRange.x),0.0,1.0);
    vec3 rgb = pigment.rgb;
    ${pbr ? `
    vec4 controls=spanRow(hit.row,10);
    if(abs(controls.w)>=1.0){
      float travel=hit.depth/(-ray.z);
      mat3 toWorld=transpose(mat3(u_curveView));
      vec3 at=toWorld*(origin+travel*ray-u_curveView[3].xyz);
      vec3 normal=toWorld*curvePbrNormal(hit.row,origin,ray,travel);
      PbrTap taps[4];
      if(pbrSampleSpecular)for(int j=0;j<4;j++){
        taps[j].coverage=0.0;
        vec2 delta=pbrTapOffset(gl_FragCoord.xy,u_curveArtScale,j);
        vec2 tapFilm=film+delta*2.0/u_curveViewport*u_curveInvProjection;
        vec3 tapOrigin=u_curveOrtho?vec3(tapFilm,0):vec3(0);
        vec3 tapRay=u_curveOrtho?vec3(0,0,-1):normalize(vec3(tapFilm,-1));
        vec2 limits=vec2(u_curveNear,min(u_curveFar,opaque+1e-4))/(-tapRay.z);
        Hit tap;
        if(!clipRay(u_curveEntry,tapOrigin,tapRay,limits)||!clipRay(u_curveCover,tapOrigin,tapRay,limits)||
            !curveHit(hit.row,tapOrigin,tapRay,limits,tap))continue;
        if(proxyDepth<1.0&&projectedDepth(tap.depth)>proxyDepth+2e-7)continue;
        float distance=tap.depth/(-tapRay.z);
        vec3 point=toWorld*(tapOrigin+tapRay*distance-u_curveView[3].xyz);
        ${section ? 'if(sectionMargin(point,tap.depth-u_curveSetback)<0.0)continue;' : ''}
        vec3 tapNormal=toWorld*curvePbrNormal(hit.row,tapOrigin,tapRay,distance);
        taps[j]=PbrTap(point,tapNormal,tapNormal,-toWorld*tapRay,min(1.0,tap.alpha/max(1e-5,hit.alpha)));
      }
      rgb=pbrShade(int(controls.w),rgb,normal,normal,toWorld*spanRow(hit.row,1).xyz,-toWorld*ray,at,pbrRestPoint(int(spanRow(hit.row,6).w)-1,at),
        fract(abs(controls.w))*2.0,0.0,0.0,pbrLinear(controls.rgb),taps);
    }` : ''}
    ${engraving ? `
    int slot=int(curveMark(hit.row,4).w);
    if(slot>0){
      vec3 at=origin+ray*(hit.depth/(-ray.z)),rest,rn,n;float stretch;
      curveChart(hit.row,at,ray,rest,rn,n,stretch);
      float footprint=2.*u_curveInvProjection.y/u_curveViewport.y*(u_curveOrtho?1.:hit.depth)*stretch;
      rgb=surfaceEngraving(slot,rgb,rest,rn,curveMark(hit.row,4).xyz,n,footprint,u_curveArtScale,0.,vec3(0));
    }` : ''}
    if (u_curveFogRange.x < 1.0) rgb = mix(rgb,u_curveFog,smoothstep(u_curveFogRange.x,1.0,normalized)*u_curveFogRange.y);
    ${wet ? 'vec3 mark; rgb = waterPaint(rgb,gl_FragCoord.xy,u_curveViewport,hit.depth,mark);' : ''}
    ${environment?.media ? 'rgb = envMedia(rgb,gl_FragCoord.xy,u_curveViewport,hit.depth);' : ''}
    if(u_filterMatte)rgb=vec3(spanRow(hit.row,9).w);
    vec4 ink = vec4(rgb*hit.alpha,hit.alpha);
    float left = 1.0-o_color.a;
    o_color += ink*left;
    ${wet ? 'o_aux += vec4(mark*hit.alpha,hit.alpha)*left;' : ''}
    ${layeredGlass ? `
    if (glass.a > 0.0 && projectedDepth(hit.depth) > pane+1e-7) o_back += ink*(1.0-o_back.a);
    else if (glass.a > 0.0) o_through.a += hit.alpha*(1.0-o_through.a);
    ` : ''}
    if (hit.alpha >= 1.0) {
      if (hit.group > 0.0) o_depth=vec4(normalized,0,0,1);
      break;
    }
  }
  `}
}
`;
}
