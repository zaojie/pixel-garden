export const engravingVertexUniforms = `uniform int u_engravingRows,u_engravingSettings;
uniform vec3 u_engravingLight;
flat out vec4 v_engravingAnchor,v_engravingX,v_engravingY,v_engravingRestNormal;
flat out vec3 v_brushTangent;
vec4 engravingData(int texel){return texelFetch(u_data,ivec2(texel%2048,texel/2048),0);}
`;
export const engravingVertex = `
 int eq=u_engravingRows*2048+a_index*4;
 vec4 ea=engravingData(eq),eb=engravingData(eq+1),ec=engravingData(eq+2),ed=engravingData(eq+3);
 int slot=(kraw&(32|256|32768))!=0?0:int(ea.x);v_engravingAnchor=vec4(ea.yzw,float(slot));v_engravingX=vec4(0);v_engravingY=vec4(0);v_engravingRestNormal=vec4(0,1,0,fogk);v_brushTangent=vec3(0.);
 if(slot>0){
  vec3 ru=eb.xyz,rv=ec.xyz,lu=vec3(eb.w,ec.w,ed.x),lv=ed.yzw;
  vec3 wu=FM*lu*sc,wv=FM*lv*sc;vec2 pu=(T*wu).xy,pv=(T*wv).xy;float area=pu.x*pv.y-pu.y*pv.x;
  if(abs(area)>1e-10&&length(cross(wu,wv))>1e-10){
   v_engravingX.xyz=(ru*pv.y-rv*pu.y)/area;v_engravingY.xyz=(-ru*pv.x+rv*pu.x)/area;
   int at=u_engravingSettings*2048+(slot-1)*9;vec4 info=engravingData(at),settings=engravingData(at+1),paper=engravingData(at+3),axis=engravingData(at+5);
   float footprint=max(length(v_engravingX.xyz),length(v_engravingY.xyz));
   vec3 normal=normalize(cross(ru,rv));
   if(info.x==0.){
    v_engravingRestNormal.xyz=normal;
    v_brushTangent=normalize(ru);
    v_engravingX.w=footprint*u_materialGridScale;
   }else{
    v_engravingRestNormal.xyz=normal;
    v_engravingX.w=clamp(log2(max(1e-8,footprint*info.y/settings.x))+axis.w,0.,3.);
   }
   v_engravingY.w=1.-clamp(paper.w+(1.-paper.w)*max(0.,dot(normalize(cross(wu,wv)),normalize(u_engravingLight))),0.,1.);
  }else v_engravingAnchor.w=0.;
 }
`;
const engravingCommon = `vec4 engravingFetch(int texel){return texelFetch(ENGRAVING_TABLE,ivec2(texel%2048,texel/2048),0);}
vec4 screenPrintFetch(int texel){return engravingFetch(texel);}
/*SCREEN_PRINT*/
vec4 engravingSetting(int slot,int row){return engravingFetch(u_engravingSettings*2048+(slot-1)*9+row);}
vec4 engravingTile(int start,int n,int group,vec2 uv){
 vec2 q=fract(uv)*float(n)-.5;ivec2 at=ivec2(floor(q));vec2 f=fract(q);
 ivec2 a=(at+ivec2(n))%n,b=(at+ivec2(n+1))%n;int page=group*n*n;
 return mix(mix(engravingFetch(start+page+a.y*n+a.x),engravingFetch(start+page+a.y*n+b.x),f.x),mix(engravingFetch(start+page+b.y*n+a.x),engravingFetch(start+page+b.y*n+b.x),f.x),f.y);
}
float engravingMip(int start,int n,int mip,float tone,vec2 uv){
 int offset=0;for(int i=0;i<2;i++){if(i>=mip)break;offset+=2*n*n;n/=2;}
 int lo=int(floor(tone)),hi=min(7,lo+1),a=lo/4,b=hi/4;
 vec4 p=engravingTile(start+offset,n,a,uv),q=a==b?p:engravingTile(start+offset,n,b,uv);
 return mix(p[lo%4],q[hi%4],fract(tone));
}
float engravingMap(vec4 info,vec2 uv,float tone,float lod){
 int lo=min(2,int(floor(lod))),hi=min(2,lo+1),start=int(info.x)*2048;
 float first=engravingMip(start,int(info.y),lo,tone,uv);
 return lo==hi?first:mix(first,engravingMip(start,int(info.y),hi,tone,uv),fract(min(lod,2.)));
}
uint brushHash(ivec3 p,uint seed){
 uvec3 q=uvec3(p);
 uint n=(q.x*0x8da6b343u)^(q.y*0xd8163841u)^(q.z*0xcb1ab31fu)^seed;
 n=(n^(n>>16u))*0x7feb352du;
 n=(n^(n>>15u))*0x846ca68bu;
 return n^(n>>16u);
}
float brushByte(uint code,uint shift){return float((code>>shift)&255u)/255.;}
float brushPigment(vec3 q,uint seed){
 ivec3 cell=ivec3(floor(q));vec3 f=smoothstep(vec3(0.),vec3(1.),fract(q));
 float pigment=0.;
 for(int x=0;x<2;x++)for(int y=0;y<2;y++)for(int z=0;z<2;z++){
  ivec3 corner=ivec3(x,y,z);vec3 weight=mix(1.-f,f,vec3(corner));
  pigment+=weight.x*weight.y*weight.z*brushByte(brushHash(cell+corner,seed),24u);
 }
 return pigment;
}
float brushDetail(vec3 q,float width,uint seed,vec3 n,vec3 u){
 ivec3 cell=ivec3(floor(q));vec3 f=fract(q),v=cross(n,u);
 float pigment=0.,total=0.;
 for(int x=0;x<2;x++)for(int y=0;y<2;y++)for(int z=0;z<2;z++){
  ivec3 corner=ivec3(x,y,z);uint h=brushHash(cell+corner,seed);
  vec3 variance=vec3(brushByte(h,0u),brushByte(h,8u),brushByte(h,16u));
  vec3 d=f-vec3(corner)-.12*(variance-.5);
  float along=dot(d,u)/(.68+.2*variance.y);
  if(abs(along)>=1.)continue;
  vec3 brush=vec3(along,dot(d,v)/(width*(.8+.4*variance.z)),dot(d,n)/.68);
  float weight=1.-smoothstep(.54,1.,dot(brush,brush));
  pigment+=weight*(brushByte(h,24u)-.5);total+=weight;
 }
 return pigment/max(1.,total);
}
float brushCoverage(vec3 q,uint seed,vec4 shape,float groupScale,float pixels,float tone,vec3 n,vec3 u){
 float groupSize=smoothstep(.65,1.8,pixels*groupScale),grain=smoothstep(shape.y,shape.z,2.*shape.x*pixels);
 if(groupSize*shape.w>0.)
  tone+=shape.w*groupSize*(smoothstep(.24,.76,brushPigment(q/groupScale,seed^0x9e3779b9u))-.5);
 if(grain>0.)tone+=grain*.9*brushDetail(q,shape.x,seed,n,u);
 return clamp(tone,0.,1.);
}
vec3 applyEngraving(int slot,vec3 rgb,vec3 point,vec3 normal,vec3 tangent,float footprint,
  float shade,float brushFootprint,float fogAmount,vec3 fogColor){
 if(slot<=0)return rgb;
 vec4 info=engravingSetting(slot,0),settings=engravingSetting(slot,1),ink=engravingSetting(slot,2),paper=engravingSetting(slot,3),origin=engravingSetting(slot,4);
 if(ink.w<=0.)return rgb;
 vec3 p=point-origin.xyz;
 vec3 litRGB=fogAmount<.999?clamp((rgb-fogColor*fogAmount)/(1.-fogAmount),0.,1.):rgb;
 vec3 w=pow(abs(normal),vec3(8));w/=max(1e-8,w.x+w.y+w.z);
 if(info.x<0.){
  if(!u_screenPrintOn)return rgb;
  int start=int(-info.x)*2048;vec3 printed=vec3(0.);
  if(info.y>.5)printed=screenPrintShade(start,vec2(0.),litRGB,footprint,1.-shade);
  else if(info.z>2.5){if(w.x>.001)printed+=w.x*screenPrintShade(start,p.yz,litRGB,footprint,1.-shade);if(w.y>.001)printed+=w.y*screenPrintShade(start,p.zx,litRGB,footprint,1.-shade);if(w.z>.001)printed+=w.z*screenPrintShade(start,p.xy,litRGB,footprint,1.-shade);}
  else{vec3 along=engravingSetting(slot,5).xyz,across=engravingSetting(slot,6).xyz;float x=dot(p,along),y=dot(p,across),z=dot(p,cross(along,across));vec2 uv=vec2(x,y);if(info.z>.5)uv=vec2(atan(z,x)*origin.w,info.z>1.5?atan(y,length(vec2(x,z)))*origin.w:y);printed=screenPrintShade(start,uv,litRGB,footprint,1.-shade);}
  return mix(printed,fogColor,fogAmount);
 }
 float lit=shade,lum=1.-clamp(dot(litRGB,vec3(.2126,.7152,.0722)),0.,1.);
 float darkness=info.w<.5?lit:info.w<1.5?lum:lit*.65+lum*.35;
 float tone=clamp(clamp((darkness-.5)*settings.z+.5+settings.w,0.,1.)*settings.y,0.,1.);
 if(info.x==0.){
  if(!u_brushworkOn)return rgb;
  uint seed=uint(info.y)|(uint(info.z)<<24u);
  float pixels=settings.x/max(1e-8,brushFootprint),groupScale=engravingSetting(slot,5).w;
  float coverage=brushCoverage(p/settings.x,seed,engravingSetting(slot,6),groupScale,pixels,tone,normal,tangent);
  return mix(rgb,mix(mix(paper.rgb,ink.rgb,coverage),fogColor,fogAmount),ink.w);
 }
 tone*=7.;float lod=clamp(log2(max(1e-8,footprint*info.y/settings.x))+engravingSetting(slot,5).w,0.,3.),coverage=0.;
 if(info.z>2.5){if(w.x>.001)coverage+=w.x*engravingMap(info,p.yz/settings.x,tone,lod);if(w.y>.001)coverage+=w.y*engravingMap(info,p.zx/settings.x,tone,lod);if(w.z>.001)coverage+=w.z*engravingMap(info,p.xy/settings.x,tone,lod);}
 else{vec3 along=engravingSetting(slot,5).xyz,across=engravingSetting(slot,6).xyz;float x=dot(p,along),y=dot(p,across),z=dot(p,cross(along,across));vec2 uv=vec2(x,y);if(info.z>.5)uv=vec2(atan(z,x)*origin.w,info.z>1.5?atan(y,length(vec2(x,z)))*origin.w:y);coverage=engravingMap(info,uv/settings.x,tone,lod);}
 if(lod>2.){int t=int(floor(tone));vec4 a=engravingSetting(slot,7),b=engravingSetting(slot,8);float m0=t<4?a[t]:b[t-4];int t1=min(7,t+1);float m1=t1<4?a[t1]:b[t1-4];coverage=mix(coverage,mix(m0,m1,fract(tone)),lod-2.);}
 return mix(rgb,mix(mix(paper.rgb,ink.rgb,coverage),fogColor,fogAmount),ink.w);
}
`;

export const engravingFragment = `
uniform highp sampler2D u_data;uniform int u_engravingSettings;uniform vec3 u_engravingLight;uniform bool u_screenPrintOn,u_brushworkOn;
flat in vec4 v_engravingAnchor,v_engravingX,v_engravingY,v_engravingRestNormal;
flat in vec3 v_brushTangent;
#define v_engravingSlot int(v_engravingAnchor.w)
#define ENGRAVING_TABLE u_data
${engravingCommon}
vec3 engravingShade(vec3 rgb,vec2 offset){
  vec3 p=v_engravingAnchor.xyz+v_engravingX.xyz*offset.x+v_engravingY.xyz*offset.y;
  float footprint=max(length(v_engravingX.xyz),length(v_engravingY.xyz));
  return applyEngraving(v_engravingSlot,rgb,p,v_engravingRestNormal.xyz,v_brushTangent,footprint,v_engravingY.w,
    v_engravingX.w,clamp(v_engravingRestNormal.w,0.,1.),u_fog);
}
`;

export function surfaceEngravingFragment(print){
return `
uniform highp sampler2D u_engravingTex;uniform int u_engravingSettings;uniform vec3 u_engravingLight;uniform bool u_screenPrintOn,u_brushworkOn;
#define ENGRAVING_TABLE u_engravingTex
${engravingCommon.replace('/*SCREEN_PRINT*/', print??'vec3 screenPrintShade(int id,vec2 uv,vec3 rgb,float f,float lit){return rgb;}')}
vec3 surfaceEngraving(int slot,vec3 rgb,vec3 point,vec3 restNormal,vec3 tangent,vec3 normal,float footprint,
  float artScale,float fogAmount,vec3 fogColor){
  if(slot<=0)return rgb;
  float ambient=engravingSetting(slot,3).w;
  float shade=1.-clamp(ambient+(1.-ambient)*max(0.,dot(normal,normalize(u_engravingLight))),0.,1.);
  return applyEngraving(slot,rgb,point,restNormal,tangent,footprint,shade,footprint*artScale,fogAmount,fogColor);
}
`;
}
