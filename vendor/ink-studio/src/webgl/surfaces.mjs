import { environmentMediaSource } from './environment-shaders.mjs';
import { emitterGLSL } from './light-emitters.mjs';
import { waterGLSL } from './water-optics.mjs';
import { glassSurfaceOutputs } from './hybrid-depth.mjs';
import { createResources } from './resources.mjs';
import { surfaceEngravingFragment } from './engraving-shaders.mjs';
import { simulationMasks,simulationMaterialSource } from './simulation-material-layers.mjs';

const uvSlip = Symbol.for('ink-studio.spriteSurface.uvSlip');
const eyeFromView = view => [0, 1, 2].map(
  k => -(view[k * 4] * view[12] + view[k * 4 + 1] * view[13] + view[k * 4 + 2] * view[14])
);

// Registered pigment, normal/roughness, emission, and optional directional capture.
const semanticSource=`
uniform highp sampler2DArray u_semanticAtlas;
uniform bool u_semanticOn,u_semanticLevels,u_semanticNormals,u_semanticOrtho;
uniform bool u_texelMaterial,u_texelSun;
uniform int u_semanticMax;
uniform vec4 u_semanticRect;
uniform vec3 u_semanticLight,u_semanticEye,u_semanticBack;
uniform float u_semanticScale,u_semanticStrength,u_semanticSpecular,u_semanticAmbient;
uniform float u_texelSunIntensity;
uniform bool u_scanMaterial,u_scanDaylight,u_scanCaptured;
uniform float u_scanAppearance;
uniform vec4 u_scanSourceView;
uniform vec2 u_scanAngles;
vec3 semanticLinear(vec3 c){
  return mix(pow((c+.055)/1.055,vec3(2.4)),c/12.92,lessThanEqual(c,vec3(.04045)));
}
vec3 semanticDisplay(vec3 c){
  c=clamp(c,0.,1.);
  return mix(1.055*pow(c,vec3(1./2.4))-.055,12.92*c,lessThanEqual(c,vec3(.0031308)));
}
ivec2 semanticTexel(vec2 uv,int level){
  ivec2 size=textureSize(u_semanticAtlas,level).xy;
  return clamp(ivec2(floor(.5+uv*vec2(size-1))),ivec2(0),size-1);
}
vec4 semanticRead(ivec2 xy,int channel,int level){
  return texelFetch(u_semanticAtlas,ivec3(xy,channel),level);
}
void semanticMaterial(vec2 uv,float span,out vec4 pigment,out vec4 detail,out vec3 emission,out vec4 captured,out float metalness){
  int ranks[16]=int[16](0,8,2,10,12,4,14,6,3,11,1,9,15,7,13,5);
  float lod=u_semanticLevels?clamp(log2(max(1.,span)),0.,float(u_semanticMax)):0.;
  int lo=int(floor(lod)),hi=int(ceil(lod));
  ivec2 a=semanticTexel(uv,lo),b=semanticTexel(uv,hi);
  vec4 first=semanticRead(a,0,lo),second=semanticRead(b,0,hi);
  bool chooseB=fract(lod)>=.5;
  ivec2 cell=chooseB?b:a;
  pigment=chooseB?second:first;
  pigment.a=step((float(ranks[(cell.y%4)*4+cell.x%4])+.5)/16.,pigment.a);
  vec2 covered=vec2((1.-fract(lod))*first.a,fract(lod)*second.a);
  covered/=max(1e-8,covered.x+covered.y);
  detail=semanticRead(a,1,lo)*covered.x+semanticRead(b,1,hi)*covered.y;
  vec4 emitA=semanticRead(a,2,lo),emitB=semanticRead(b,2,hi);
  emission=(semanticLinear(emitA.rgb)*covered.x+semanticLinear(emitB.rgb)*covered.y)*u_semanticStrength;
  metalness=dot(vec2(emitA.a,emitB.a),covered);
  captured=u_scanCaptured&&u_scanAppearance>0.?semanticRead(cell,3,chooseB?hi:lo):vec4(0);
}
vec3 semanticObserver(vec3 point){
  return u_semanticOrtho?u_semanticBack:normalize(u_semanticEye-point);
}
vec3 semanticFinish(vec3 pigment,vec3 normal,float roughness,float sheen,vec3 emission,vec3 world,
  vec3 irradiance,vec3 highlight,vec4 captured){
  vec3 view=semanticObserver(world);
  vec3 h=u_semanticLight+view;
  h*=inversesqrt(max(dot(h,h),1e-18));
  float gloss=1.-roughness;
  float direct=max(0.,dot(normal,u_semanticLight));
  float intensity=u_texelSun?u_texelSunIntensity:1.;
  float shine=dot(normal,view)>0.&&direct>0.?
    intensity*sheen*gloss*pow(max(0.,dot(normal,h)),4.+96.*gloss*gloss):0.;
  float diffuse=u_semanticAmbient+(u_texelSun?intensity:1.-u_semanticAmbient)*direct;
  if(u_scanMaterial&&!u_scanDaylight){diffuse=u_semanticAmbient;shine=0.;}
  vec3 lit=clamp(semanticLinear(pigment)*(diffuse+irradiance)+emission+shine+highlight,0.,1.);
  if(captured.a>0.){
    vec3 source=u_scanSourceView.xyz-world*u_scanSourceView.w;
    source*=inversesqrt(max(dot(source,source),1e-16));
    float weight=smoothstep(u_scanAngles.x,u_scanAngles.y,dot(view,source));
    lit=mix(lit,semanticLinear(captured.rgb),weight*captured.a*u_scanAppearance);
  }
  return semanticDisplay(lit);
}
`;

// Opaque pixel cutouts share an atlas and depth buffer with the splat pass.
export function createSurfacePipeline(gl, { updatesSince, geometryUpdatesSince, cutaway, samplingPlan, samplingGLSL, projectionPlan, spriteLOD, sharedShaders, print, pbrGLSL } = {}) {
  const resources = createResources(gl);
  try {
    const shaders = sharedShaders?.cache ?? new Map();
    const program = sharedShaders?.program ?? resources.program;
    function shaderFor(wet, section, lighting, normalMapped, environment, paletteSampling, styledProjection, maskedPart, glassLayers,
      anchors,engraving,semantic,simulation,blockColor,objectTexels,pbr,pbrSpecularSampling,pbrBands,authored) {
      const chart = lighting && !normalMapped;
      const marks=anchors || engraving, rest=marks||simulation||objectTexels||pbrBands, needsFrame=normalMapped||semantic||pbr||authored;
      const key = [
        wet?.airPockets,
        wet?.layered,
        wet?.atlas,
        !!wet,
        !!section,
        !!lighting,
        normalMapped,
        !!environment?.media,
        !!environment?.lit,
        paletteSampling,
        styledProjection,
        maskedPart,
        glassLayers,anchors,engraving,semantic,simulation,blockColor,objectTexels,pbr,pbrSpecularSampling,pbrBands,authored
      ].join('/');
      if (shaders.has(key)) return shaders.get(key);
      const shader = program(
        `#version 300 es
precision highp float;
layout(location=0) in vec3 a_point;
layout(location=1) in vec2 a_texel;
uniform mat4 u_view,u_proj;
uniform highp int u_historyFirst;
flat out highp int v_historyId;
out vec2 v_texel;
out vec3 v_world;
${chart ? 'layout(location=2)in vec2 a_light;out vec2 v_light;' : ''}
${rest?'layout(location=3)in vec3 a_rest;out vec3 v_rest;':''}
${marks?'layout(location=4)in vec2 a_marks;flat out vec2 v_marks;':''}
${simulation?'layout(location=5)in vec2 a_simUV;out vec2 v_simUV;':''}
${authored?'layout(location=6)in vec3 a_shadingNormal;out vec3 v_shadingNormal;':''}
out float v_depth;
out vec3 v_camera;
${styledProjection ? 'uniform vec2 u_surfaceViewport;uniform float u_vertexSnap;out vec2 v_affine;' : ''}
void main(){
  vec4 p=u_view*vec4(a_point,1.0);
  v_historyId=u_historyFirst>0 ? u_historyFirst+gl_VertexID/3 : 0;
  ${chart ? 'v_light=a_light;' : ''}
  ${rest?'v_rest=a_rest;':''}
  ${marks?'v_marks=a_marks;':''}
  ${simulation?'v_simUV=a_simUV;':''}
  ${authored?'v_shadingNormal=a_shadingNormal;':''}
  v_texel=a_texel;v_world=a_point;v_depth=-p.z;v_camera=p.xyz;
  gl_Position=u_proj*p;
  ${styledProjection ? `
  v_affine=a_texel*gl_Position.w;
  if(u_vertexSnap>0.0&&gl_Position.w>0.0){
    vec2 screen=(gl_Position.xy/gl_Position.w*.5+.5)*u_surfaceViewport;
    screen.y=u_surfaceViewport.y-screen.y;
    screen=floor(screen/u_vertexSnap+.5)*u_vertexSnap;
    screen.y=u_surfaceViewport.y-screen.y;
    gl_Position.xy=(screen/u_surfaceViewport*2.0-1.0)*gl_Position.w;
  }` : ''}
}`,
        `#version 300 es
precision highp float;
precision highp int;
flat in highp int v_historyId;
uniform sampler2D u_atlas;
${blockColor ? 'uniform highp sampler2D u_blockColor;uniform bool u_blockOn;uniform ivec2 u_blockOrigin;uniform vec3 u_blockTint;' : ''}
uniform vec4 u_range,u_splatPlane;
${wet ? waterGLSL(wet.airPockets, wet.layered, wet.atlas) : 'uniform vec4 u_arrivalPlane;'}
uniform vec2 u_surfaceViewport;
uniform vec3 u_fog;
uniform bool u_index, u_graphDepth, u_portalCover;
uniform bool u_uvSlip, u_filterMatte, u_frontOnly;
uniform float u_filterSoftness;
${maskedPart ? 'uniform int u_geometryCode;uniform vec2 u_partGrid;'+spriteLOD : ''}
uniform vec4 u_uvTransform, u_uvBounds;
${paletteSampling ? `uniform float u_samplingScale;
${samplingGLSL}` : ''}
${section ? cutaway : ''}
${environment?.media ? environmentMediaSource(environment.lit) : ''}
${lighting ? emitterGLSL : ''}
${chart ? 'uniform highp sampler2DArray u_surfaceLit;uniform int u_surfaceLightCount;in vec2 v_light;' : ''}
${normalMapped ? 'uniform highp sampler2D u_surfaceNormalMap;uniform bool u_hasSurfaceNormalMap;' : ''}
${needsFrame ? 'uniform bool u_surfaceNormals;uniform vec3 u_surfaceEye;' : ''}
${authored?'in vec3 v_shadingNormal;':''}
${semantic ? semanticSource : ''}
${pbr ? 'uniform int u_surfacePbr;uniform bool u_pbrOrtho;uniform vec3 u_pbrBack;uniform float u_pbrArtScale;\n'+
  (pbrSpecularSampling?'#define INK_PBR_FOOTPRINT\n':'')+(pbrBands?'#define INK_PBR_BANDS\n':'')+pbrGLSL : ''}
${rest?'in vec3 v_rest;':''}
${marks?'flat in vec2 v_marks;':''}
${simulation?simulationMaterialSource:''}
${engraving ? `uniform float u_engravingArtScale;${surfaceEngravingFragment(print)}` : ''}
in vec2 v_texel;
in vec3 v_world;
in float v_depth;
in vec3 v_camera;
${styledProjection ? 'uniform bool u_affine;uniform vec4 u_projectionBounds;in vec2 v_affine;' : ''}
layout(location=0) out vec4 color;
layout(location=1) out vec4 depth;
${glassLayers && !wet ? 'vec4 water;' : 'layout(location=2) out vec4 water;'}
${glassLayers ? glassSurfaceOutputs(wet) : ''}
${needsFrame ? `
vec3 surfaceNormal(vec3 normal,vec3 dx,vec3 dy,vec2 du,vec2 dv,vec4 map,bool continuous){
  float det=du.x*dv.y-du.y*dv.x;
  if(map.a<.5||abs(det)<=1e-5*length(du)*length(dv))return normal;
  vec3 local=max(vec3(-127),(continuous?map.rgb*255.0:round(map.rgb*255.0))-128.0);
  if(dot(local,local)==0.0)return normal;
  vec3 tangent=normalize(dx*dv.y-dy*du.y)*sign(det);
  ${authored ? `tangent-=normal*dot(normal,tangent);
  if(dot(tangent,tangent)<1e-12)return normal;
  tangent=normalize(tangent);` : ''}
  vec3 bitangent=cross(normal,tangent);
  // Positive map Y faces up the texture, opposite increasing row coordinates.
  bitangent*=dot(bitangent,(dx*dv.x-dy*du.x)*sign(det))<0.0?-1.0:1.0;
  return normalize(mat3(tangent,bitangent,normal)*local);
}
` : ''}
${objectTexels ? `
mat3 texelFrame(vec3 dx,vec3 dy){
  vec3 right=normalize(dx),normal=normalize(cross(right,dy));
  return mat3(right,cross(normal,right),normal);
}
` : ''}
void main(){
  if(u_frontOnly&&!gl_FrontFacing)discard;
  ${maskedPart ? 'if(!spriteGeometry(u_geometryCode,gl_FragCoord.xy*u_partGrid))discard;' : ''}
  vec2 uv=${styledProjection ? 'u_affine?v_affine*gl_FragCoord.w:v_texel' : 'v_texel'};
  vec2 rawTexel=u_uvSlip?uv*u_uvTransform.xy+u_uvTransform.zw:uv;
  vec2 texel=u_uvSlip?clamp(rawTexel,u_uvBounds.xy,u_uvBounds.zw):rawTexel;
  ${styledProjection ? 'if(u_affine&&!u_uvSlip)texel=clamp(texel,u_projectionBounds.xy,u_projectionBounds.zw);' : ''}
  ${paletteSampling ? `
  if(u_paletteSampling)texel=clamp(texel,u_sampleRegion.xy,u_sampleRegion.xy+u_sampleRegion.zw-1.0);
  ` : ''}
  vec4 ink=texelFetch(u_atlas,ivec2(texel),0);
  ${paletteSampling ? `
  float q=gl_FragCoord.w;
  vec2 numerator=v_texel*q;
  vec2 scale=u_samplingScale*(u_uvSlip?u_uvTransform.xy:vec2(1.0));
  vec2 dx=(dFdx(numerator)-v_texel*dFdx(q))*scale/q;
  vec2 dy=(dFdy(numerator)-v_texel*dFdy(q))*scale/q;
  ${styledProjection ? 'if(u_affine){dx=dFdx(uv)*scale;dy=dFdy(uv)*scale;}' : ''}
  if(u_paletteSampling)ink=samplePalette(rawTexel,dx,dy,ink);
  ` : ''}
  ${blockColor ? `
  if(u_blockOn){
    ivec2 coord=clamp(ivec2(texel)-u_blockOrigin,ivec2(0),textureSize(u_blockColor,0)-1);
    ink=vec4(round(texelFetch(u_blockColor,coord,0).rgb*u_blockTint)/255.0,1);
  }
  ` : ''}
  ${semantic ? `
  float semanticSpan=max(length(dFdx(rawTexel)),length(dFdy(rawTexel)))*u_semanticScale;
  vec4 detail,captured;
  vec3 emission;
  float metalness;
  if(u_semanticOn){
    vec2 coord=(rawTexel-u_semanticRect.xy-.5)/max(vec2(1),u_semanticRect.zw-1.);
    semanticMaterial(coord,semanticSpan,ink,detail,emission,captured,metalness);
  }
  ` : ''}
  ${needsFrame ? `
  // Use the surface UV frame even when affine sampling or snapping changes the lookup.
  vec3 dpdx=dFdx(v_world),dpdy=dFdy(v_world);
  vec3 geometricNormal=normalize(cross(dpdx,dpdy));
  vec3 normalSight=${semantic ? 'u_semanticOn&&u_semanticOrtho?u_semanticBack:u_surfaceEye-v_world' : 'u_surfaceEye-v_world'};
  if(dot(geometricNormal,normalSight)<0.0)geometricNormal=-geometricNormal;
  vec2 scaleUV=u_uvSlip?u_uvTransform.xy:vec2(1);
  vec2 uvDx=dFdx(v_texel)*scaleUV,uvDy=dFdy(v_texel)*scaleUV;
  ` : ''}
  ${objectTexels?'vec3 texelRestX=dFdx(v_rest),texelRestY=dFdy(v_rest);':''}
  ${simulation?'vec2 simUVx=dFdx(v_simUV),simUVy=dFdy(v_simUV);vec3 simRestX=dFdx(v_rest),simRestY=dFdy(v_rest);':''}
  ${engraving ? 'vec3 restX=dFdx(v_rest),restY=dFdy(v_rest),worldX=dFdx(v_world),worldY=dFdy(v_world);vec2 texX=dFdx(v_texel),texY=dFdy(v_texel);' : ''}
  if(ink.a<.5||dot(u_arrivalPlane,vec4(v_world,1.0))<0.0||dot(u_splatPlane,vec4(v_camera,1.0))<0.0)discard;
  ${section ? 'float margin=sectionMargin(v_world,v_depth); if(margin<0.0)discard;' : ''}
  ${anchors ? 'color=vec4(v_world,v_marks.x);depth=vec4(v_rest,1);return;' : ''}
  if(u_filterMatte){color=vec4(vec3(u_filterSoftness),1);depth=vec4(0);water=vec4(0);return;}
  float d=clamp((v_depth-u_range.x)/(u_range.y-u_range.x),0.0,1.0);
  float fog=u_range.z<1.0?smoothstep(u_range.z,1.0,d)*u_range.w:0.0;
  vec3 mark=vec3(0.0), rgb=ink.rgb;
  ${chart ? 'vec3 irradiance=vec3(0);for(int i=0;i<32;i++){if(i>=u_surfaceLightCount)break;irradiance+=texelFetch(u_surfaceLit,ivec3(ivec2(v_light),i),0).rgb;}rgb=eSRGB(min(vec3(1),eLinear(rgb)*(vec3(1)+irradiance)));' : ''}
  ${needsFrame ? `
  vec4 map=${normalMapped ? 'u_surfaceNormals&&u_hasSurfaceNormalMap?texelFetch(u_surfaceNormalMap,ivec2(texel),0):vec4(0)' : 'vec4(0)'};
  ${semantic?'if(u_semanticOn)map=u_semanticNormals&&u_surfaceNormals?vec4(detail.rgb,1):vec4(0);':''}
  vec3 base=geometricNormal;
  ${authored ? `if(dot(v_shadingNormal,v_shadingNormal)>1e-12){
    base=normalize(v_shadingNormal);
    if(dot(base,geometricNormal)<0.0)base=-base;
  }` : ''}
  vec3 normal=surfaceNormal(base,dpdx,dpdy,uvDx,uvDy,map,${semantic?'u_semanticOn':'false'});
  ` : ''}
  ${objectTexels ? `
  if(u_semanticOn&&u_texelMaterial&&u_surfaceNormals){
    vec3 local=detail.rgb*255.-128.;
    normal=normalize(texelFrame(dpdx,dpdy)*transpose(texelFrame(texelRestX,texelRestY))*local);
    if(!gl_FrontFacing)normal=-normal;
  }
  ` : ''}
  ${semantic?'float sheen=u_semanticSpecular;':''}
  ${simulation?`
  if(u_semanticOn)simApply(v_simUV,v_rest,geometricNormal,simUVx,simUVy,simRestX,simRestY,dpdx,dpdy,rgb,detail.a,sheen,normal);
  `:''}
  ${pbr && simulation ? 'vec3 pbrPigment=u_semanticOn&&u_simCount>0?rgb:ink.rgb;' : ''}
  ${semantic ? 'vec3 scanIrradiance=vec3(0),scanHighlight=vec3(0);':''}
  ${normalMapped ? `
  ${semantic?'if(u_semanticOn&&(u_scanMaterial||u_texelMaterial))scanIrradiance=eSurfaceLighting(v_world,normal,false,vec3(0),geometricNormal,semanticObserver(v_world),detail.a,sheen,scanHighlight);else':''}
  rgb=eSRGB(min(vec3(1),eLinear(rgb)*(vec3(1)+eLighting(v_world,normal,false,vec3(0),geometricNormal))));
  ` : ''}
  ${semantic?'if(u_semanticOn&&!u_index)rgb=semanticFinish(rgb,normal,detail.a,sheen,emission,v_world,scanIrradiance,scanHighlight,captured);':''}
  ${pbr ? `
  if(u_surfacePbr!=0){
    float det=uvDx.x*uvDy.y-uvDx.y*uvDy.x;
    vec3 tangent=(dpdx*uvDy.y-dpdy*uvDx.y)*sign(det);
    PbrTap taps[4];
    if(pbrSampleSpecular)for(int j=0;j<4;j++){
      vec2 offset=pbrTapOffset(gl_FragCoord.xy,u_pbrArtScale,j);
      vec3 at=v_world+dpdx*offset.x+dpdy*offset.y;
      float coverage=1.0;
      ${section ? 'if(sectionMargin(at,v_depth)<0.0)coverage=0.0;' : ''}
      taps[j]=PbrTap(at,normal,geometricNormal,u_pbrOrtho?u_pbrBack:u_surfaceEye-at,coverage);
    }
    rgb=pbrShade(u_surfacePbr,${simulation?'pbrPigment':'ink.rgb'},normal,geometricNormal,tangent,
      u_pbrOrtho?u_pbrBack:u_surfaceEye-v_world,v_world,${pbrBands?'v_rest':'v_world'},
      ${semantic?'u_semanticOn?detail.a:.45,u_semanticOn?metalness:0.,0.,u_semanticOn?emission:vec3(0)':'0.45,0.0,0.0,vec3(0)'},taps);
  }` : ''}
  ${engraving ? `
  if(v_marks.y>0.0){
    vec3 restN=cross(restX,restY),liveN=cross(worldX,worldY);
    if(length(restN)>1e-12&&length(liveN)>1e-12){
      restN=normalize(restN);
      float det=texX.x*texY.y-texX.y*texY.x;
      vec3 tangent=abs(det)>1e-9 ? (restX*texY.y-restY*texX.y)*sign(det) :
        cross(abs(restN.x)<.8?vec3(1,0,0):vec3(0,1,0),restN);
      rgb=surfaceEngraving(int(v_marks.y),rgb,v_rest,restN,normalize(tangent),normalize(liveN),
        max(length(restX),length(restY)),u_engravingArtScale,0.,vec3(0));
    }
  }` : ''}
  ${wet ? 'if(!u_index)rgb=waterPaint(rgb,gl_FragCoord.xy,u_surfaceViewport,v_depth,mark);' : ''}
  ${environment?.media ? 'if(!u_index)rgb=envMedia(rgb,gl_FragCoord.xy,u_surfaceViewport,v_depth);' : ''}
  rgb=mix(rgb,u_fog,fog);
  ${section ? 'rgb=mix(rgb,u_sectionEdge.rgb,sectionInk(margin));' : ''}
  color=u_index?vec4(float(u_portalCover)):vec4(rgb,1.0);
  depth=vec4(d,0.0,0.0,1.0);water=vec4(mark,1.0);
  ${glassLayers ? 'splitOpaqueSurface(color,gl_FragCoord.z);' : ''}
  if(u_index&&v_historyId>0){
    depth=vec4(float(v_historyId&255),float((v_historyId>>8)&255),float(v_historyId>>16),255.)/255.;
  }
  if(u_index&&u_graphDepth){float z=round(d*16777215.0);water=vec4(mod(z,256.0),mod(floor(z/256.0),256.0),floor(z/65536.0),255.0)/255.0;}
}`
      );
      shaders.set(key, shader);
      return shader;
    }
    const vao = resources.allocate('VertexArray'),
      buffer = resources.allocate('Buffer'),
      texture = resources.texture();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
    gl.bindVertexArray(null);
    let active,
      activeVertices,
      activeBlockOnly,
      activeNormals,
      activeCornerNormals,
      cornerBuffer,
      normalTexture,
      revision = -1,
      geometryRevision = -1,
      width = 0,
      height = 0,
      disposed = false,
      staging = new Uint8Array(0);
    const stats = { textureBytes: 0, geometryBytes: 0, blockBytes: 0 };
    const blockUnit = 7, blockTextures = new Map();
    let s3tc, blockOwner;
    function bindBlock(texture) {
      gl.activeTexture(gl.TEXTURE0 + blockUnit);
      const existing = blockTextures.get(texture);
      if (existing) {
        gl.bindTexture(gl.TEXTURE_2D,existing);
      } else {
        const gpu = resources.texture();
        // The surface pass reads sRGB bytes and applies its own transfer for lighting.
        gl.compressedTexImage2D(gl.TEXTURE_2D,0,s3tc.COMPRESSED_RGB_S3TC_DXT1_EXT,
          texture.width,texture.height,0,texture.blocks);
        blockTextures.set(texture,gpu);
        stats.blockBytes+=texture.blocks.byteLength;
        stats.textureBytes+=texture.blocks.byteLength;
      }
    }
    function upload(atlas, blockOnly) {
      stats.blockBytes = 0;
      stats.textureBytes = 0;
      stats.geometryBytes = 0;
      if (disposed) throw new Error('Surface pipeline has been disposed');
      const next = atlas?.revision ?? 0,
        nextGeometry = atlas?.geometryRevision ?? 0;
      if (
        active === atlas &&
        revision === next &&
        geometryRevision === nextGeometry &&
        activeVertices === atlas.vertices &&
        activeBlockOnly === blockOnly &&
        activeNormals === atlas.normalPixels &&
        activeCornerNormals === atlas.cornerNormals &&
        width === atlas.width &&
        height === atlas.height
      )
        return;
      if (
        !atlas ||
        !(atlas.vertices instanceof Float32Array) ||
        atlas.vertices.length % 15 ||
        !Number.isSafeInteger(atlas.width) ||
        atlas.width < 1 ||
        !Number.isSafeInteger(atlas.height) ||
        atlas.height < 1 ||
        !(atlas.pixels instanceof Uint8Array) ||
        atlas.pixels.length !== atlas.width * atlas.height * 4 ||
        (atlas.normalPixels !== undefined &&
          (!(atlas.normalPixels instanceof Uint8Array) || atlas.normalPixels.length !== atlas.pixels.length)) ||
        atlas.cornerNormals !== undefined && (!(atlas.cornerNormals instanceof Float32Array) ||
          atlas.cornerNormals.length !== atlas.vertices.length / 5 * 3) ||
        !Number.isSafeInteger(next) ||
        next < 0 ||
        !Number.isSafeInteger(nextGeometry) ||
        nextGeometry < 0
      )
        throw new RangeError('Invalid sprite surface atlas');
      if (Math.max(atlas.width, atlas.height) > gl.getParameter(gl.MAX_TEXTURE_SIZE))
        throw new RangeError('Surface atlas exceeds GPU size limit');
      const replace = active !== atlas || width !== atlas.width || height !== atlas.height;
      if (replace || activeVertices !== atlas.vertices) {
        if (!atlas.vertices.every(Number.isFinite)) throw new RangeError('Invalid sprite surface geometry');
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferData(gl.ARRAY_BUFFER, atlas.vertices, nextGeometry ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
        stats.geometryBytes = atlas.vertices.byteLength;
      }
      if (!replace && activeVertices === atlas.vertices && geometryRevision !== nextGeometry) {
        const changes = geometryUpdatesSince?.(atlas, geometryRevision);
        const first = changes?.length ? Math.min(...changes.map(r => r.first)) : 0,
          end = changes?.length ? Math.max(...changes.map(r => r.first + r.count)) : atlas.vertices.length / 5;
        const values = atlas.vertices.subarray(first * 5, end * 5);
        if (!values.every(Number.isFinite)) throw new RangeError('Invalid sprite surface geometry');
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
        gl.bufferSubData(gl.ARRAY_BUFFER, first * 20, values);
        stats.geometryBytes = values.byteLength;
      }
      if (atlas.cornerNormals && (replace || activeCornerNormals !== atlas.cornerNormals)) {
        if (!atlas.cornerNormals.every(Number.isFinite)) throw new RangeError('Invalid surface corner normals');
        cornerBuffer ??= resources.allocate('Buffer');
        gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, atlas.cornerNormals, nextGeometry ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
        stats.geometryBytes += atlas.cornerNormals.byteLength;
      }
      let x = 0, y = 0, w = atlas.width, h = atlas.height;
      if (!replace && revision !== next) {
        const changes = updatesSince?.(atlas, revision);
        if (changes?.length) {
          x = Math.min(...changes.map(r => r.x));
          y = Math.min(...changes.map(r => r.y));
          w = Math.max(...changes.map(r => r.x + r.width)) - x;
          h = Math.max(...changes.map(r => r.y + r.height)) - y;
        }
      }
      function uploadPixels(data, full) {
        if (full) {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, atlas.width, atlas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
          stats.textureBytes += data.byteLength;
          return;
        }
        const bytes = w * h * 4;
        if (staging.length < bytes) staging = new Uint8Array(bytes);
        for (let row = 0; row < h; row++)
          staging.set(
            data.subarray(((y + row) * atlas.width + x) * 4, ((y + row) * atlas.width + x + w) * 4),
            row * w * 4
          );
        gl.texSubImage2D(gl.TEXTURE_2D, 0, x, y, w, h, gl.RGBA, gl.UNSIGNED_BYTE, staging.subarray(0, bytes));
        stats.textureBytes += bytes;
      }
      gl.activeTexture(gl.TEXTURE0 + 6);
      if (replace || activeBlockOnly !== blockOnly || revision !== next) {
        gl.bindTexture(gl.TEXTURE_2D, texture);
        if (blockOnly) {
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
          stats.textureBytes+=4;
        } else uploadPixels(atlas.pixels, replace || activeBlockOnly !== blockOnly);
      }
      const newNormals = activeNormals !== atlas.normalPixels;
      if (atlas.normalPixels && (replace || newNormals || revision !== next)) {
        gl.activeTexture(gl.TEXTURE0 + 24);
        normalTexture ??= resources.texture();
        gl.bindTexture(gl.TEXTURE_2D, normalTexture);
        uploadPixels(atlas.normalPixels, replace || newNormals);
      }
      active = atlas;
      activeVertices = atlas.vertices;
      activeBlockOnly = blockOnly;
      activeNormals = atlas.normalPixels;
      activeCornerNormals = atlas.cornerNormals;
      revision = next;
      geometryRevision = nextGeometry;
      width = atlas.width;
      height = atlas.height;
    }
    let lightProgram,
      lightTexture,
      lightBuffer,
      lightSize = [],
      lightKey,
      lightRows,
      lastSide,
      lightGeometry,
      lightRevision,
      lightPositions,
      lightCount,
      lightResolution;
    const lightVAO = resources.allocate('VertexArray'),
      lightInstanceBuffer = resources.allocate('Buffer'),
      lightUVBuffer = resources.allocate('Buffer');
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, lightUVBuffer);
    gl.enableVertexAttribArray(2);
    gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 8, 0);
    gl.bindVertexArray(lightVAO);
    gl.bindBuffer(gl.ARRAY_BUFFER, lightInstanceBuffer);
    for (let i = 0; i < 3; i++) {
      gl.enableVertexAttribArray(i);
      gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 36, i * 12);
      gl.vertexAttribDivisor(i, 1);
    }
    gl.bindVertexArray(null);
    let markBuffer, activeMarkAtlas, markSignature, markRevision;
    function bindMarks(atlas,anchors,engraving,simulation) {
      const signature=(anchors?.key??'')+'|'+(engraving?.key??'')+'|'+Number(simulation), rev=atlas.geometryRevision??0;
      const charts=atlas.surfaceCharts??[];
      const needsRevision=!atlas.sceneMeshId || charts.some(chart=>!chart.rest);
      if(markBuffer && activeMarkAtlas===atlas && markSignature===signature && (!needsRevision || rev===markRevision))return;
      const count=atlas.vertices.length/5,stride=simulation?7:5,data=new Float32Array(count*stride);
      let cursor=0;
      for(let i=0;i<count;i++){
        while(charts[cursor]?.end<=i)cursor++;
        const chart=charts[cursor]?.first<=i ? charts[cursor] : undefined;
        const j=(i-(chart?.first??0))*3;
        const at=i*stride;
        data.set(chart?.rest?.subarray(j,j+3)??atlas.vertices.subarray(i*5,i*5+3),at);
        data[at+3]=anchors?.select(chart?.objectId,chart?.materialId)??0;
        data[at+4]=engraving?.mesh(chart)??0;
        if(simulation&&chart){
          const {x,y,width,height}=chart.region, k=(i-chart.first)*2;
          data[at+5]=chart.uv?.[k]??(atlas.vertices[i*5+3]-x-.5)/Math.max(1,width-1);
          data[at+6]=chart.uv?.[k+1]??(atlas.vertices[i*5+4]-y-.5)/Math.max(1,height-1);
        }
      }
      markBuffer??=resources.allocate('Buffer');
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER,markBuffer);
      gl.bufferData(gl.ARRAY_BUFFER,data,needsRevision ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
      gl.vertexAttribPointer(3,3,gl.FLOAT,false,stride*4,0);
      gl.vertexAttribPointer(4,2,gl.FLOAT,false,stride*4,12);
      if(simulation)gl.vertexAttribPointer(5,2,gl.FLOAT,false,stride*4,20);
      activeMarkAtlas=atlas;markSignature=signature;markRevision=rev;
    }
    function shadeAtlas(atlas, lighting, view) {
      const resolution = lighting.frame.emitterSurfaceResolution ?? 16;
      if (!Number.isSafeInteger(resolution) || resolution < 4 || resolution > 64)
        throw new RangeError('Surface light resolution must be 4..64');
      const columns = Math.min(64, Math.floor(gl.getParameter(gl.MAX_TEXTURE_SIZE) / resolution));
      if (
        lightGeometry !== atlas.vertices ||
        lightRevision !== (atlas.geometryRevision ?? 0) ||
        lightResolution !== resolution
      ) {
        lightResolution = resolution;
        lightCount = atlas.vertices.length / 15;
        lightPositions = new Float32Array(lightCount * 9);
        const uv = new Float32Array(lightCount * 6);
        for (let i = 0; i < lightCount; i++) {
          const x = (i % columns) * resolution,
            y = Math.floor(i / columns) * resolution;
          for (let j = 0; j < 3; j++) {
            lightPositions.set(atlas.vertices.subarray(i * 15 + j * 5, i * 15 + j * 5 + 3), i * 9 + j * 3);
            uv.set([x + 0.5 + (j === 1 ? resolution - 1 : 0), y + 0.5 + (j === 2 ? resolution - 1 : 0)], i * 6 + j * 2);
          }
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, lightInstanceBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, lightPositions, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, lightUVBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, uv, gl.DYNAMIC_DRAW);
        lightGeometry = atlas.vertices;
        lightRevision = atlas.geometryRevision ?? 0;
      }
      const eye = eyeFromView(view);
      let side = 0;
      for (let i = 0; i < lightPositions.length; i += 9) {
        const a = lightPositions.subarray(i, i + 3),
          u = [0, 1, 2].map(k => lightPositions[i + 3 + k] - a[k]),
          v = [0, 1, 2].map(k => lightPositions[i + 6 + k] - a[k]),
          n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
        side = (Math.imul(side, 31) + (n.reduce((s, v, k) => s + v * (eye[k] - a[k]), 0) < 0 ? 1 : 0)) | 0;
      }
      const key = [lightGeometry, lightRevision, lighting.shadow, lighting.cutoutRevision, resolution];
      const same = (a, b) => a?.length === b.length && b.every((v, i) => v === a[i]);
      if (same(lightKey, key) && same(lightRows, lighting.rows) && side === lastSide) {
        gl.activeTexture(gl.TEXTURE0 + 24);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, lightTexture);
        return;
      }
      let allChanged = !same(lightKey, key) || side !== lastSide;
      const previous = lightRows;
      lightKey = key;
      lightRows = lighting.rows.slice();
      lastSide = side;
      if (!lightTexture) {
        lightTexture = resources.allocate('Texture');
        gl.activeTexture(gl.TEXTURE0 + 24);
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, lightTexture);
        for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER])
          gl.texParameteri(gl.TEXTURE_2D_ARRAY, p, gl.NEAREST);
        for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T])
          gl.texParameteri(gl.TEXTURE_2D_ARRAY, p, gl.CLAMP_TO_EDGE);
      }
      lightBuffer ??= resources.allocate('Framebuffer');
      gl.activeTexture(gl.TEXTURE0 + 24);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY, lightTexture);
      const size = [columns * resolution, Math.max(1, Math.ceil(lightCount / columns)) * resolution];
      if (size[1] > gl.getParameter(gl.MAX_TEXTURE_SIZE))
        throw new RangeError('Surface lighting charts exceed the GPU atlas budget');
      if (lightSize[0] !== size[0] || lightSize[1] !== size[1] || lightSize[2] !== lighting.count) {
        if (size[0] * size[1] * lighting.count > 16777216)
          throw new RangeError('Surface lighting cache exceeds its texel budget');
        gl.texImage3D(
          gl.TEXTURE_2D_ARRAY,
          0,
          gl.RGBA16F,
          ...size,
          Math.max(1, lighting.count),
          0,
          gl.RGBA,
          gl.FLOAT,
          null
        );
        lightSize = [...size, lighting.count];
        allChanged = true;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, lightBuffer);
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, lightTexture, 0, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error('Surface lightmap target is incomplete');
      lightProgram ??= resources.program(
        `#version 300 es
precision highp float;layout(location=0)in vec3 a_a;layout(location=1)in vec3 a_b;layout(location=2)in vec3 a_c;uniform vec2 u_lightSize;uniform float u_lightResolution;uniform int u_lightColumns;uniform vec3 u_surfaceEye;out vec3 v_world;flat out vec3 v_normal;void main(){vec2 q=vec2(gl_VertexID&1,gl_VertexID>>1);vec2 uv=(q*u_lightResolution-.5)/(u_lightResolution-1.0);v_world=a_a+(a_b-a_a)*uv.x+(a_c-a_a)*uv.y;v_normal=normalize(cross(a_b-a_a,a_c-a_a));if(dot(v_normal,u_surfaceEye-a_a)<0.0)v_normal=-v_normal;vec2 tile=vec2(gl_InstanceID%u_lightColumns,gl_InstanceID/u_lightColumns)*u_lightResolution;gl_Position=vec4((tile+q*u_lightResolution)/u_lightSize*2.0-1.0,0,1);}`,
        `#version 300 es
precision highp float;${emitterGLSL}
in vec3 v_world;flat in vec3 v_normal;out vec4 color;void main(){color=vec4(eLighting(v_world,v_normal,false,vec3(0)),1);}`
      );
      gl.disable(gl.BLEND);
      gl.disable(gl.DEPTH_TEST);
      gl.viewport(0, 0, lightSize[0], lightSize[1]);
      gl.useProgram(lightProgram.p);
      lightProgram.set({
        ...lighting.uniforms,
        u_lightSize: lightSize.slice(0, 2),
        u_lightColumns: columns,
        u_lightResolution: resolution,
        u_surfaceEye: eye
      });
      gl.bindVertexArray(lightVAO);
      for (let i = 0; i < lighting.count; i++) {
        if (!allChanged && same(previous?.subarray(i * 44, i * 44 + 44), lighting.rows.subarray(i * 44, i * 44 + 44)))
          continue;
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, lightTexture, 0, i);
        gl.clearBufferfv(gl.COLOR, 0, [0, 0, 0, 0]);
        lightProgram.set({ u_ePick: i + 1 });
        gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, lightCount);
      }
    }
    const integralTextures = new Map();
    const prefixUnits = [1, 2];
    function bindFilter(filter) {
      let entry = integralTextures.get(filter);
      if (!entry) {
        const { width, height } = filter.rect;
        function upload(unit, data) {
          gl.activeTexture(gl.TEXTURE0 + unit);
          const table = resources.texture();
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, width + 1, height + 1, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, data);
          stats.textureBytes += data.byteLength;
          return table;
        }
        const totals = upload(prefixUnits[0], filter.totals);
        const accents = filter.accents ? upload(prefixUnits[1], filter.accents) : totals;
        entry = [totals, accents];
        integralTextures.set(filter, entry);
      }
      for (let i = 0; i < prefixUnits.length; i++) {
        gl.activeTexture(gl.TEXTURE0 + prefixUnits[i]);
        gl.bindTexture(gl.TEXTURE_2D, entry[i]);
      }
    }

    const semanticUnit=5, semanticTextures=new Map(), simulationUnit=11;
    const masks=simulationMasks(gl,resources,simulationUnit);
    let semanticOwner, semanticRanges;
    function bindSemantic(chain){
      gl.activeTexture(gl.TEXTURE0+semanticUnit);
      let texture=semanticTextures.get(chain);
      if(texture){
        gl.bindTexture(gl.TEXTURE_2D_ARRAY,texture);
        return;
      }
      texture=resources.allocate('Texture');
      semanticTextures.set(chain,texture);
      gl.bindTexture(gl.TEXTURE_2D_ARRAY,texture);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.NEAREST_MIPMAP_NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAX_LEVEL,chain.levels.length-1);
      for(const [lod,{width,height,rgba}] of chain.levels.entries()){
        gl.texImage3D(gl.TEXTURE_2D_ARRAY,lod,gl.RGBA8,width,height,chain.captured?4:3,0,gl.RGBA,gl.UNSIGNED_BYTE,rgba);
        stats.textureBytes+=rgba.byteLength;
      }
    }
    function draw(
      atlas,
      {
        lighting,
        environment,
        view,
        projection,
        range,
        fog,
        index = false,
        graphDepth = false,
        glassLayers = false,
        portalCover = false,
        arrivalPlane = [0, 0, 0, 0],
        clip = [0, 0, 0, 0],
        water,
        wet,
        width = 1,
        height = 1,
        paletteSampling,
        surfaceNormals = true, sceneryBlockTextures = true,
        semanticMipmaps = true, scanFrame, simulationLayers, eye, back=[0,0,1], ortho, sun,
        samplingScale = 1,
        geometryCode = 0,
        partGrid = [1,1],
        section, anchors, engraving, coverageFilter, historyFirst = 0, pbr, pbrUniforms, bindPbr
      }
    ) {
      const planBlocks=atlas?.blockTextures;
      const blockReady=planBlocks && planBlocks.revision===atlas.revision && sceneryBlockTextures && !atlas[uvSlip]?.ranges.length &&
        (s3tc===undefined ? (s3tc=gl.getExtension('WEBGL_compressed_texture_s3tc')) : s3tc);
      const blockRanges=blockReady ? planBlocks.ranges : undefined;
      if(blockOwner!==planBlocks || !blockReady && blockTextures.size) {
        for(const gpu of blockTextures.values())resources.free(gpu);
        blockTextures.clear();
        blockOwner=planBlocks;
      }
      upload(atlas,Boolean(blockReady && planBlocks.complete));
      stats.blockMode=planBlocks ? blockReady ? 'native-bc1' : 'decoded-rgba' : undefined;
      const semantic=atlas.semanticMipmaps;
      const objectTexels=Boolean(semantic?.some(row=>row.objectTexels));
      const firstSimulation=!index&&!coverageFilter&&semantic?.find(row=>simulationLayers?.get(row.simulation)?.active);
      const simulation=Boolean(firstSimulation);
      masks.start(atlas);
      if(semanticOwner!==atlas || semanticRanges!==semantic){
        for(const texture of semanticTextures.values())resources.free(texture);
        semanticTextures.clear();
        semanticOwner=atlas;
        semanticRanges=semantic;
      }
      if (paletteSampling !== false && atlas.paletteSampling !== undefined && (!samplingPlan || !samplingGLSL))
        throw new TypeError('Palette-filtered surfaces need SpriteSurface.samplingPlan and samplingGLSL');
      if (atlas.projectionRanges !== undefined && !projectionPlan)
        throw new TypeError('Styled surface projection needs SpriteSurface.projectionPlan');
      const projections = projectionPlan?.(atlas) ?? [];
      const rangesWithFilters = paletteSampling === false ? null : samplingPlan?.(atlas);
      const filters = new Set((rangesWithFilters ?? []).map(range => range.filter).filter(Boolean));
      for (const [filter, entry] of integralTextures) if (!filters.has(filter)) {
        for (const table of new Set(entry)) resources.free(table);
        integralTextures.delete(filter);
      }
      const useSampling = filters.size > 0;
      const lit = Boolean(lighting?.active && !index && !coverageFilter);
      const authored = Boolean(atlas.cornerNormals && surfaceNormals && !index && !coverageFilter);
      const normalMapped = lit && Boolean(authored||atlas.normalPixels||simulation||objectTexels||semantic?.some(row=>row.chain.scan));
      if (normalMapped) {
        gl.activeTexture(gl.TEXTURE0 + 24);
        gl.bindTexture(gl.TEXTURE_2D, atlas.normalPixels?normalTexture:texture);
      } else if (lit) {
        const framebuffer = gl.getParameter(gl.DRAW_FRAMEBUFFER_BINDING),
          viewport = gl.getParameter(gl.VIEWPORT),
          blend = gl.isEnabled(gl.BLEND),
          depth = gl.isEnabled(gl.DEPTH_TEST);
        shadeAtlas(atlas, lighting, view);
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.viewport(...viewport);
        if (blend) gl.enable(gl.BLEND);
        else gl.disable(gl.BLEND);
        if (depth) gl.enable(gl.DEPTH_TEST);
        else gl.disable(gl.DEPTH_TEST);
      }
      gl.activeTexture(gl.TEXTURE0 + 6);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      const etched=!coverageFilter && !index && engraving && atlas.surfaceCharts?.some(chart=>engraving.mesh(chart)) ? engraving : null;
      if(etched || anchors || simulation || objectTexels || pbr?.hasBands)bindMarks(atlas,anchors,etched,simulation);
      if(etched)etched.bind();
      const shader = shaderFor(wet, section, lit, normalMapped, environment, useSampling, projections.length > 0, Boolean(geometryCode), glassLayers,
        !!anchors,!!etched,!!semantic,simulation,!!blockRanges,objectTexels,!!pbr,!!pbr?.specularSampling,!!pbr?.hasBands,authored);
      if (pbr) bindPbr();
      gl.useProgram(shader.p);
      shader.set({
        u_view: view,
        u_proj: projection,
        u_range: range,
        u_fog: fog,
        u_atlas: 6,
        ...(blockRanges&&{u_blockColor:blockUnit}),
        u_frontOnly: Boolean(atlas.frontOnly),
        ...(useSampling && { u_sampleTotals: prefixUnits[0], u_sampleAccents: prefixUnits[1], u_samplingScale: samplingScale }),
        u_surfaceLightCount: lighting?.count ?? 0,
        u_surfaceLit: 24,
        ...(normalMapped && {u_surfaceNormalMap:24,u_hasSurfaceNormalMap:!!atlas.normalPixels}),
        ...((normalMapped||semantic||pbr||authored)&&{u_surfaceNormals:surfaceNormals,u_surfaceEye:eye??eyeFromView(view)}),
        ...(pbr&&{...pbrUniforms,u_pbrOrtho:!!ortho,u_pbrBack:back,u_pbrArtScale:samplingScale}),
        ...(semantic&&{u_semanticAtlas:semanticUnit,u_semanticLevels:semanticMipmaps!==false,u_semanticScale:samplingScale,
          u_semanticEye:eye??eyeFromView(view),u_semanticBack:back,u_semanticOrtho:!!ortho}),
        ...(simulation&&{u_simMasks:simulationUnit}),
        u_index: +index,
        u_historyFirst:historyFirst,
        u_filterMatte: !!coverageFilter,
        u_glassBoundary: 13,
        u_graphDepth: graphDepth,
        u_portalCover: portalCover,
        u_arrivalPlane: arrivalPlane,
        u_splatPlane: clip,
        u_surfaceViewport: [width, height],
        u_geometryCode: geometryCode, u_partGrid: partGrid,
        ...(etched && {...etched.uniforms,u_engravingArtScale:samplingScale}),
        ...water,
        ...section,
        ...environment?.uniforms,
        ...lighting?.uniforms
      });
      if(semantic)bindSemantic(semantic[0].chain);
      let previousBlock;
      if(blockRanges){
        gl.activeTexture(gl.TEXTURE0+blockUnit);
        previousBlock=gl.getParameter(gl.TEXTURE_BINDING_2D);
        bindBlock(blockRanges[0].texture);
      }
      let previousMask;
      if(simulation){
        gl.activeTexture(gl.TEXTURE0+simulationUnit);
        previousMask=gl.getParameter(gl.TEXTURE_BINDING_2D);
        const plan=firstSimulation.simulation;
        masks.bind(plan,simulationLayers.get(plan),surfaceNormals);
      }
      const restoreTextures=()=>{
        if(blockRanges){
          gl.activeTexture(gl.TEXTURE0+blockUnit);
          gl.bindTexture(gl.TEXTURE_2D,previousBlock);
        }
        if(simulation){
          gl.activeTexture(gl.TEXTURE0+simulationUnit);
          gl.bindTexture(gl.TEXTURE_2D,previousMask);
        }
      };
      gl.bindVertexArray(vao);
      if (authored) {
        gl.bindBuffer(gl.ARRAY_BUFFER, cornerBuffer);
        gl.enableVertexAttribArray(6);
        gl.vertexAttribPointer(6, 3, gl.FLOAT, false, 12, 0);
      } else gl.disableVertexAttribArray(6);
      for(const [i,on] of [[3,etched||anchors||simulation||objectTexels||pbr?.hasBands],[4,etched||anchors],[5,simulation]])
        if(on)gl.enableVertexAttribArray(i);else gl.disableVertexAttribArray(i);
      if (lit && !normalMapped) gl.enableVertexAttribArray(2);
      else gl.disableVertexAttribArray(2);
      let projectionIndex = 0;
      function drawProjected(first, end) {
        if (!projections.length) {
          gl.drawArrays(gl.TRIANGLES, first, end - first);
          return;
        }
        while (first < end) {
          while (projections[projectionIndex]?.end <= first) projectionIndex++;
          const next = projections[projectionIndex], style = next?.first <= first ? next : null;
          const stop = Math.min(end, (style ? style.end : next?.first) ?? end);
          shader.set({ u_affine: style?.affine ?? false, u_vertexSnap: (style?.vertexSnap ?? 0) * samplingScale,
            ...(style?.affine && { u_projectionBounds: style.bounds }) });
          gl.drawArrays(gl.TRIANGLES, first, stop - first);
          first = stop;
        }
      }
      let chartIndex = 0, semanticIndex=0, blockIndex=0;
      function drawRange(first, end) {
        if(!coverageFilter && !semantic && !blockRanges && !pbr){
          drawProjected(first,end);
          return;
        }
        const charts=coverageFilter || pbr ? atlas.surfaceCharts??[] : [];
        while(first<end){
          while(charts[chartIndex]?.end<=first)chartIndex++;
          while(semantic?.[semanticIndex]?.end<=first)semanticIndex++;
          const next=charts[chartIndex], chart=next?.first<=first?next:null;
          const candidate=semantic?.[semanticIndex], mip=candidate?.first<=first?candidate:null;
          while(blockRanges?.[blockIndex]?.end<=first)blockIndex++;
          const nextBlock=blockRanges?.[blockIndex], block=nextBlock?.first<=first?nextBlock:null;
          const stop=Math.min(end,(chart?chart.end:next?.first)??end,(mip?mip.end:candidate?.first)??end,
            (block?block.end:nextBlock?.first)??end);
          if(block){
            bindBlock(block.texture);
            shader.set({u_blockOn:true,u_blockOrigin:[block.region.x,block.region.y],u_blockTint:block.tint});
          } else if(blockRanges)shader.set({u_blockOn:false});
          if(coverageFilter)shader.set({u_filterSoftness:coverageFilter.at(chart?.materialId)});
          if(pbr)shader.set({u_surfacePbr:pbr.surface(chart)*(pbr.shadows?.receivesSurface(chart)===false?-1:1)});
          if(mip){
            const {chain,region:r}=mip;
            const key=(mip.objectTexels||mip.liveLight) && sun;
            bindSemantic(chain);
            shader.set({u_semanticOn:true,u_semanticRect:[r.x,r.y,r.width,r.height],u_semanticMax:chain.levels.length-1,
              u_semanticStrength:chain.emissionStrength,u_semanticLight:key?.direction??chain.rule.light,
              u_semanticSpecular:chain.rule.specular,u_semanticAmbient:key?.ambient??chain.rule.ambient,u_semanticNormals:chain.normal,
              u_texelMaterial:!!mip.objectTexels,u_texelSun:!!key,u_texelSunIntensity:key?.intensity??1,
              u_scanMaterial:!!chain.scan,u_scanCaptured:chain.captured,u_scanDaylight:scanFrame?.daylight ?? true,
              u_scanAppearance:scanFrame?.appearance ?? 0,
              ...(chain.captured&&{u_scanSourceView:chain.scan.sourceView,u_scanAngles:chain.scan.angles})});
          } else if(semantic)shader.set({u_semanticOn:false});
          if(simulation){
            const plan=mip?.simulation, state=simulationLayers.get(plan);
            shader.set(state?.active?masks.bind(plan,state,surfaceNormals):{u_simCount:0});
          }
          drawProjected(first,stop);
          first=stop;
        }
      }
      if (useSampling) {
        const previous = prefixUnits.map(unit => {
          gl.activeTexture(gl.TEXTURE0 + unit);
          return gl.getParameter(gl.TEXTURE_BINDING_2D);
        });
        try {
          let bound = filters.values().next().value;
          bindFilter(bound);
          for (const { first, end, shift, filter } of rangesWithFilters) {
            if (filter && filter !== bound) {
              bindFilter(filter);
              bound = filter;
            }
            shader.set({
              u_uvSlip: Boolean(shift), ...(shift && { u_uvTransform: shift.transform, u_uvBounds: shift.bounds }),
              u_paletteSampling: Boolean(filter),
              ...(filter && { u_sampleRegion: [filter.rect.x, filter.rect.y, filter.rect.width, filter.rect.height],
                u_sampleRule: [filter.rule.alphaCoverage, filter.rule.protectCoverage, filter.rule.tieTolerance],
                u_sampleCount: filter.colors.length,
                ...(filter.swatches.length && { u_sampleSwatches: filter.swatches }),
                u_protectCount: filter.rule.protect.length,
                ...(filter.highlights.length && { u_sampleProtected: filter.highlights }) })
            });
            drawRange(first, end);
          }
        } finally {
          for (let i = 0; i < prefixUnits.length; i++) {
            gl.activeTexture(gl.TEXTURE0 + prefixUnits[i]);
            gl.bindTexture(gl.TEXTURE_2D, previous[i]);
          }
          restoreTextures();
          gl.activeTexture(gl.TEXTURE0);
        }
        return;
      }
      const plan = atlas[uvSlip];
      const ranges = plan?.vertices === atlas.vertices && plan.width === atlas.width && plan.height === atlas.height
        ? plan.ranges : [];
      let first = 0;
      for (const { first: start, end, transform, bounds } of ranges) {
        if (start > first) {
          shader.set({ u_uvSlip: false });
          drawRange(first, start);
        }
        shader.set({ u_uvSlip: true, u_uvTransform: transform, u_uvBounds: bounds });
        drawRange(start, end);
        first = end;
      }
      const count = active.vertices.length / 5;
      if (first < count) {
        shader.set({ u_uvSlip: false });
        drawRange(first, count);
      }
      restoreTextures();
      gl.activeTexture(gl.TEXTURE0);
    }
    return {
      draw,
      get stats() {
        return { ...stats };
      },
      dispose() {
        if (disposed) return;
        disposed = true;
        resources.dispose();
      }
    };
  } catch (error) {
    resources.dispose();
    throw error;
  }
}
