const maxLayers=8;

export const simulationMaterialSource=`
uniform highp sampler2D u_simMasks;
uniform int u_simCount;
uniform bool u_simRelief;
uniform vec4 u_simFrom[${maxLayers}],u_simTo[${maxLayers}],u_simInk[${maxLayers}],u_simResponse[${maxLayers}];
in vec2 v_simUV;
float simMask(int i,vec2 uv,vec3 rest){
  vec4 a=u_simFrom[i],b=u_simTo[i];
  int kind=int(a.w);
  if(kind==0)return 1.;
  if(kind==3){
    if(any(lessThan(uv,vec2(0)))||any(greaterThan(uv,vec2(1))))return 0.;
    ivec2 size=ivec2(a.yz);
    vec2 xy=uv*vec2(size-1), f=fract(xy);
    ivec2 p=ivec2(xy),q=min(p+1,size-1),start=ivec2(0,int(a.x));
    float upper=mix(texelFetch(u_simMasks,start+p,0).r,texelFetch(u_simMasks,start+ivec2(q.x,p.y),0).r,f.x);
    float lower=mix(texelFetch(u_simMasks,start+ivec2(p.x,q.y),0).r,texelFetch(u_simMasks,start+q,0).r,f.x);
    return mix(upper,lower,f.y);
  }
  vec3 p=kind==1?vec3(uv,0):rest;
  vec3 dist=min(p-a.xyz,b.xyz-p);
  float d=min(min(dist.x,dist.y),kind==1?1e8:dist.z);
  return b.w>0.?smoothstep(0.,b.w,d):float(d>0.);
}
float simHeight(vec2 uv,vec3 rest){
  float height=0.;
  for(int i=0;i<${maxLayers};i++){
    if(i>=u_simCount)break;
    vec4 response=u_simResponse[i];
    if(response.z!=0.&&response.w>0.)height+=response.z*response.w*simMask(i,uv,rest);
  }
  return height;
}
void simApply(vec2 uv,vec3 rest,vec3 normal,vec2 uvx,vec2 uvy,vec3 rx,vec3 ry,vec3 dx,vec3 dy,
  inout vec3 pigment,inout float roughness,inout float sheen,inout vec3 shading){
  if(u_simCount<=0)return;
  vec3 rgb=semanticLinear(pigment);
  bool painted=false;
  for(int i=0;i<${maxLayers};i++){
    if(i>=u_simCount)break;
    vec4 response=u_simResponse[i], color=u_simInk[i];
    if(response.w<=0.)continue;
    float w=response.w*simMask(i,uv,rest);
    if(w<=0.)continue;
    if(color.a>0.){rgb=mix(rgb,color.rgb,w);painted=true;}
    if(response.x>=0.)roughness=mix(roughness,response.x,w);
    if(response.y>=0.)sheen=mix(sheen,response.y,w);
  }
  if(painted)pigment=semanticDisplay(rgb);
  if(!u_simRelief)return;
  vec3 a=cross(dy,normal),b=cross(normal,dx);
  float det=dot(dx,a);
  if(abs(det)<=1e-12)return;
  float fx=2.*(simHeight(uv+uvx*.25,rest+rx*.25)-simHeight(uv-uvx*.25,rest-rx*.25));
  float fy=2.*(simHeight(uv+uvy*.25,rest+ry*.25)-simHeight(uv-uvy*.25,rest-ry*.25));
  shading=normalize(shading-(a*fx+b*fy)/det);
}
`;

export function simulationMasks(gl,resources,unit){
  const textures=new Map();
  let owner;
  function start(atlas){
    if(owner===atlas)return;
    for(const item of textures.values())resources.free(item.texture);
    textures.clear();
    owner=atlas;
  }
  function bind(plan,state,normals){
    let entry=textures.get(plan);
    gl.activeTexture(gl.TEXTURE0+unit);
    if(!entry){
      const texture=resources.texture();
      gl.texImage2D(gl.TEXTURE_2D,0,gl.R8,plan.width,plan.height,0,gl.RED,gl.UNSIGNED_BYTE,plan.bitmap);
      const from=new Float32Array(maxLayers*4),to=new Float32Array(maxLayers*4),ink=new Float32Array(maxLayers*4);
      for(const [i,row] of plan.layers.entries()){
        const mask=row.mask, at=i*4;
        if(mask.kind===3)from.set([mask.y,mask.width,mask.height,3],at);
        else if(mask.kind){
          from.set([...mask.min,mask.kind],at);
          to.set([...mask.max,mask.feather],at);
        }
        if(row.pigment)ink.set([...row.pigment,1],at);
      }
      entry={texture,from,to,ink};
      textures.set(plan,entry);
    } else gl.bindTexture(gl.TEXTURE_2D,entry.texture);
    if(entry.state!==state){
      entry.state=state;
      entry.response??=new Float32Array(maxLayers*4);
      for(const [i,row] of plan.layers.entries())
        entry.response.set([row.roughness??-1,row.sheen??-1,row.relief,state.amounts[i]],i*4);
    }
    return {u_simCount:plan.layers.length,u_simRelief:state.relief&&normals,u_simFrom:entry.from,
      u_simTo:entry.to,u_simInk:entry.ink,u_simResponse:entry.response};
  }
  return {start,bind};
}
