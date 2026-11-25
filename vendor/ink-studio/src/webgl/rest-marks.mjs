const spaces=['screen','world','object','view'];
const noRule={dither:'screen',spacing:1,origin:[0,0,0]};
const ruleWords = rule => [...rule.origin,rule.spacing,spaces.indexOf(rule.dither),0,0,0];

export function createRestMarks(api,table,maxSize) {
  const rules=table(256,27,maxSize), splats=table(2048,25,maxSize);
  let values, slots, previous, ruleKey, ruleRows;
  return (world,stable,dirty,view) => {
    const authored=stable ?? {...noRule,rules:[]}, keys=[authored,...authored.rules];
    const key=JSON.stringify(keys.map(row=>[row.dither,row.spacing,row.origin,row.object,row.material]));
    if (key!==ruleKey) {
      ruleRows=Float32Array.from(keys.flatMap(ruleWords));
      rules.upload(ruleRows);
      ruleKey=key;
    } else rules.bind();
    const select=(object,material)=>keys.indexOf(api.matching(authored,object,material))+1;
    const meta=world.renderMetadata, rest=world.reconstructionPose?.base ??
      (world.anim?.base instanceof Float32Array ? world.anim.base : world.rest ?? world.data);
    const engraving=world.engraving, live=world.splatFootprints?.frames;
    const sources=[world,world.count,meta,key,rest,engraving,live];
    const refill=!previous || sources.some((value,i)=>value!==previous[i]) || rest===world.data && dirty?.length!==0;
    if (refill) {
      if (!values || values.length!==world.count*16) values=new Float32Array(world.count*16);
      slots=new Uint8Array(world.count);
      for(let i=0;i<world.count;i++){
        const start=i*16,n=i*3,f=i*6;
        const slot=select(meta?.objects?.[meta.objectIds?.[i]],meta?.materials?.[meta.materialIds?.[i]]);
        slots[i]=slot;
        const retained=engraving?.indices[i]>0;
        const anchor=(retained ? engraving.anchors.subarray(n,n+3) : rest.subarray(start,start+3));
        const normal=meta?.normals?.subarray(n,n+3) ?? [0,0,0];
        const a=retained && engraving.restFrames.subarray(f,f+6), b=retained && live?.subarray(f,f+6);
        values.set([...anchor,slot,...(a || [0,0,0,0,0,0]),...(b || [...normal,0,0,0])],start);
      }
      splats.upload(values);
      previous=sources;
    } else splats.bind();
    const project=api.viewProjection(view), basis=[[1,0,0],[0,1,0],[0,0,1]].map(v=>project(...v));
    return {select,slot:i=>slots[i]??1,key,bindRules:()=>rules.bind(),bindSites:()=>splats.bind(),uniforms:{u_markRules:27,u_markSites:25,
      u_markViewX:basis.map(v=>v[0]),u_markViewY:basis.map(v=>v[1])}};
  };
}

export const markVertexUniforms=`
uniform bool u_markPass;
uniform highp sampler2D u_markSites;
flat out vec4 v_markWorld,v_markRest;
flat out vec3 v_markWorldX,v_markWorldY,v_markRestX,v_markRestY;
`;
export const markVertex=`
if(u_markPass){
  ivec2 q=ivec2((a_index&511)*4,a_index>>9);
  vec4 a=texelFetch(u_markSites,q,0),b=texelFetch(u_markSites,q+ivec2(1,0),0);
  vec4 c=texelFetch(u_markSites,q+ivec2(2,0),0),d=texelFetch(u_markSites,q+ivec2(3,0),0);
  vec3 referenceU=b.xyz,referenceV=vec3(b.w,c.xy);
  vec3 liveU=vec3(c.zw,d.x),liveV=d.yzw;
  if(dot(referenceU,referenceU)<1e-14){
    vec3 n=liveU;
    if(dot(n,n)<1e-14) n=transpose(mat3(u_view))*vec3(0.,0.,1.);
    n=normalize(n);
    referenceU=normalize(cross(abs(n.x)<.8?vec3(1.,0.,0.):vec3(0.,1.,0.),n));
    referenceV=cross(n,referenceU);
    liveU=referenceU;liveV=referenceV;
  }
  vec3 u=FM*liveU*sc,v=FM*liveV*sc;
  vec2 pu=(T*u).xy,pv=(T*v).xy;
  float det=pu.x*pv.y-pu.y*pv.x;
  v_markWorld=vec4(pos,a.w);v_markRest=vec4(a.xyz,1.);
  v_markWorldX=vec3(0);v_markWorldY=vec3(0);v_markRestX=vec3(0);v_markRestY=vec3(0);
  if(abs(det)>1e-9){
    v_markWorldX=(u*pv.y-v*pu.y)/det;v_markWorldY=(v*pu.x-u*pv.x)/det;
    v_markRestX=(referenceU*pv.y-referenceV*pu.y)/det;v_markRestY=(referenceV*pu.x-referenceU*pv.x)/det;
  }
}
`;
export const markFragmentUniforms=`
uniform bool u_markPass;
flat in vec4 v_markWorld,v_markRest;
flat in vec3 v_markWorldX,v_markWorldY,v_markRestX,v_markRestY;
`;
export const markSplatWrite=`
if(u_markPass){
  if(v_gl>0.5)discard;
  o=vec4(v_markWorld.xyz+v_markWorldX*v_d.x+v_markWorldY*v_d.y,v_markWorld.w);
  od=vec4(v_markRest.xyz+v_markRestX*v_d.x+v_markRestY*v_d.y,v_markRest.w);
  return;
}
`;

export const markResolve=`
uniform highp sampler2D u_markWorld,u_markRest,u_markRules;
uniform vec3 u_markViewX,u_markViewY;
vec2 markView(vec3 p){return vec2(dot(p,u_markViewX),dot(p,u_markViewY));}
vec4 markRule(int slot,int offset){return texelFetch(u_markRules,ivec2(2*(slot-1)+offset,0),0);}
ivec2 markCoordinates(ivec2 pixel,ivec2 screen,vec4 explicitRule,int explicitMode,float scale){
  vec4 world=texelFetch(u_markWorld,pixel,0);
  int slot=int(world.w);
  if(slot<=0)return ivec2(floor(vec2(screen)/scale));
  int mode=explicitMode>=0 ? explicitMode : int(markRule(slot,1).x);
  if(mode==0)return ivec2(floor(vec2(screen)/scale));
  vec4 rule=explicitMode>=0 ? explicitRule : markRule(slot,0);
  vec3 point=(mode==2 ? texelFetch(u_markRest,pixel,0).xyz : world.xyz)-rule.xyz;
  float span=rule.w*scale;
  if(mode==3)return ivec2(floor(markView(point)/span+1e-5));
  ivec3 cell=ivec3(floor(point/span+1e-5));
  return ivec2(dot(vec3(cell),vec3(1,3,2)),dot(vec3(cell),vec3(2,1,3)));
}
ivec2 markCoordinates(ivec2 pixel,ivec2 screen){return markCoordinates(pixel,screen,vec4(0),-1,1.);}
vec3 markPoint(ivec2 pixel,int mode){
  return mode==2 ? texelFetch(u_markRest,pixel,0).xyz : texelFetch(u_markWorld,pixel,0).xyz;
}
float markFootprint(ivec2 pixel,ivec2 step,int slot,int mode,vec3 point){
  ivec2 size=textureSize(u_markWorld,0);
  for(int side=1;side>=-1;side-=2){
    ivec2 neighbor=pixel+step*side;
    if(any(lessThan(neighbor,ivec2(0)))||any(greaterThanEqual(neighbor,size)))continue;
    if(int(texelFetch(u_markWorld,neighbor,0).w)!=slot)continue;
    vec3 delta=markPoint(neighbor,mode)-point;
    return mode==3 ? length(markView(delta)) : length(delta);
  }
  return 0.;
}
float markGain(ivec2 pixel,vec4 rule,int mode,float scale,vec2 fade){
  if(fade.y<=0.)return 1.;
  int slot=int(texelFetch(u_markWorld,pixel,0).w);
  if(mode<0 && slot>0){ mode=int(markRule(slot,1).x); rule=markRule(slot,0); }
  float width=scale;
  if(mode>0&&slot>0){
    vec3 point=markPoint(pixel,mode);
    float footprint=max(markFootprint(pixel,ivec2(1,0),slot,mode,point),
      markFootprint(pixel,ivec2(0,1),slot,mode,point));
    if(footprint<=1e-7)return 1.;
    width=rule.w*scale/footprint;
  }
  if(fade.x==fade.y)return float(width>=fade.y);
  return smoothstep(fade.x,fade.y,width);
}
ivec2 positiveGrid(ivec2 p,ivec2 size){return (p%size+size)%size;}
`;
