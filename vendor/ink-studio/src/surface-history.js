/* Native art-grid history shared by the software and browser resolvers. */
const SurfaceHistory = (() => {
  const reasons = ['reset','accepted','unowned','outside','owner','depth'];
  const limits = { weight:[0,.95,.55], depthPixels:[.05,4,.75], maxGap:[.001,10,.2] };
  function prepare(value) {
    if (value === undefined || value === false) return null;
    const raw = value === true ? {} : value;
    if (!raw || Object.getPrototypeOf(raw)!==Object.prototype || Object.keys(raw).some(key=>!Object.hasOwn(limits,key)))
      throw new TypeError('Surface history expects a boolean or weight, depthPixels and maxGap');
    const config={};
    for(const [key,[min,max,fallback]] of Object.entries(limits)){
      const n=Object.hasOwn(raw,key)?raw[key]:fallback;
      if(!Number.isFinite(n)||n<min||n>max)throw new RangeError('Invalid surface history '+key);
      config[key]=n;
    }
    return config.weight>0 ? {...config,key:JSON.stringify(config)} : null;
  }

  function resets(previous,current,settings,cut) {
    if(cut)return 'cut';
    if(!previous)return 'first';
    if(previous.source!==current.source||previous.layout!==current.layout)return 'source';
    if(previous.w!==current.w||previous.h!==current.h)return 'resize';
    if(previous.camera.ortho!==current.camera.ortho)return 'projection';
    if(current.time<previous.time||current.time-previous.time>settings.maxGap)return 'time';
    if(previous.key!==settings.key)return 'settings';
    return null;
  }

  function summary(mask,w,h,reset) {
    const counts=Object.fromEntries(reasons.map(reason=>[reason,0]));
    for(const code of mask)counts[reasons[code]]++;
    return {w,h,reset,counts,mask};
  }

  // Map a current art-pixel center at its owned depth into the old camera.
  function reproject(camera,previous,w,h,x,y,z,result) {
    const a=camera.view,b=previous.view;
    const units=2*camera.tangent*(camera.ortho?1:z)/h;
    const px=(x+.5-w*.5)*units-a[12],py=(h*.5-y-.5)*units-a[13],pz=-z-a[14];
    const wx=a[0]*px+a[1]*py+a[2]*pz,wy=a[4]*px+a[5]*py+a[6]*pz,wz=a[8]*px+a[9]*py+a[10]*pz;
    const qx=b[0]*wx+b[4]*wy+b[8]*wz+b[12],qy=b[1]*wx+b[5]*wy+b[9]*wz+b[13];
    const depth=-(b[2]*wx+b[6]*wy+b[10]*wz+b[14]);
    const scale=2*previous.tangent*(previous.ortho?1:depth)/h;
    result[0]=Math.floor(w*.5+qx/scale);
    result[1]=Math.floor(h*.5-qy/scale);
    result[2]=depth;result[3]=scale;
    return depth >= previous.clipNear && depth <= previous.clipFar &&
      result[0]>=0&&result[0]<w&&result[1]>=0&&result[1]<h;
  }

  function history() {
    let previous;
    return Object.freeze({
      clear(){previous=undefined;},
      resolve(current,settings,{cut=false,capture=false}={}){
        const {w,h,camera,color,alpha,depth,owners,keys}=current;
        const reset=resets(previous,current,settings,cut);
        const mask=capture&&new Uint8Array(w*h);
        if(!reset){
          const prior=previous, sample=[0,0,0,0];
          for(let y=0;y<h;y++)for(let x=0;x<w;x++){
            const p=y*w+x;
            let code;
            if(!owners[p]||!Number.isFinite(depth[p]))code=2;
            else if(!reproject(camera,prior.camera,w,h,x,y,depth[p],sample))code=3;
            else {
              const q=sample[1]*w+sample[0];
              if(!prior.owners[q] || keys[owners[p]]!==prior.keys[prior.owners[q]])code=4;
              else if(!Number.isFinite(prior.depth[q])||Math.abs(prior.depth[q]-sample[2])>Math.max(1e-4,settings.depthPixels*sample[3]))code=5;
              else {
                code=1;
                const offset=((h-1-y)*w+x)*3, old=((h-1-sample[1])*w+sample[0])*3;
                const a=alpha?alpha[p]/255:1, b=prior.alpha?prior.alpha[q]/255:1, gain=settings.weight;
                const out=(1-gain)*a+gain*b;
                for(let c=0;c<3;c++)color[offset+c]=out>1e-6?
                  (color[offset+c]*a*(1-gain)+prior.color[old+c]*b*gain)/out:0;
                if(alpha)alpha[p]=Math.round(out*255);
              }
            }
            if(mask)mask[p]=code;
          }
        }
        previous={...current,key:settings.key,color:color.slice(),alpha:alpha?.slice(),depth:depth.slice(),
          owners:owners.slice(),keys:keys.slice()};
        return mask ? summary(mask,w,h,reset) : undefined;
      }
    });
  }
  return Object.freeze({prepare,resets,summary,history});
})();
if (typeof module !== 'undefined' && module.exports) module.exports = SurfaceHistory;
else globalThis[Symbol.for('ink-studio.surface-history')] = SurfaceHistory;
