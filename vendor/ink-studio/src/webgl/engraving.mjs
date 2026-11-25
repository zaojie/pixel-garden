import { createAttachedData } from './attached-data.mjs';
const emptyMeans=new Float32Array(8);
function packEngravingStyles(api,styles,firstRow){
  const styleFields = styles.map(s => s.brushwork ? null : s.screenPrint ? api.print.pack(s.screenPrint) : api.maps(s).data);
  const fields = [...new Set(styleFields.filter(Boolean))];
  const params = new Float32Array(styles.length * 9 * 4);
  let row =
    firstRow + Math.ceil(params.length / 8192);
  const fieldRows = new Map();
  for (const field of fields) {
    fieldRows.set(field, row);
    row += Math.ceil(field.length / 8192);
  }
  styles.forEach((s, i) => {
    const brush = s.brushwork, field = styleFields[i],
      means = brush || s.screenPrint ? emptyMeans : api.maps(s).levels[2].means;
    params.set(
      [
        brush ? 0 : fieldRows.get(field) * (s.screenPrint ? -1 : 1),
        brush ? s.seed & 0xffffff : s.screenPrint ? +(s.screenPrint.space === 'screen') : s.resolution,
        brush ? s.seed >>> 24 : ['plane', 'cylinder', 'sphere', 'triplanar'].indexOf(s.projection),
        ['normal', 'luminance', 'hybrid'].indexOf(s.tone),
        s.scale,
        s.density,
        s.contrast,
        s.bias,
        ...s.ink,
        s.strength,
        ...s.paper,
        s.ambient,
        ...s.origin,
        s.radius,
        ...s.along,
        brush ? brush.groupScale : s.lodBias,
        ...(brush ? [brush.width, ...brush.detailPixels] : s.across),
        brush?.groupContrast ?? 0,
        ...means
      ],
      i * 36
    );
  });
  return {params,fields};
}

export function createEngravingBinding(api) {
  const pack = createAttachedData();
  let bag, frames, count, metadata, params, fields, styles;
  return (world, base, dirty, light) => {
    const b = api.validate(world),
      live = world.splatFootprints?.frames;
    if (b !== bag || live !== frames || world.count !== count) {
      api.validate(world);
      if (!live) throw new Error('Engraving needs retained surface axes');
      const sameRows =
        bag &&
        b.indices === bag.indices &&
        b.anchors === bag.anchors &&
        b.restFrames === bag.restFrames &&
        frames === live &&
        count === world.count;
      if (!sameRows) {
        metadata = new Float32Array(world.count * 16);
        for (let i = 0; i < b.indices.length; i++) {
          const id = b.indices[i];
          if (!id) continue;
          const k = i * 6,
            q = i * 16,
            a = i * 3;
          metadata.set(
            [
              id,
              ...b.anchors.subarray(a, a + 3),
              ...b.restFrames.subarray(k, k + 3),
              live[k],
              ...b.restFrames.subarray(k + 3, k + 6),
              live[k + 1],
              live[k + 2],
              live[k + 3],
              live[k + 4],
              live[k + 5]
            ],
            q
          );
        }
      }
      if (styles !== b.styles || count !== world.count) {
        ({params,fields}=packEngravingStyles(api,b.styles,2*Math.ceil(world.count*16/8192)));
      }
      bag = b;
      frames = live;
      count = world.count;
      styles = b.styles;
    }
    const result = pack(base, world.count * 16, [metadata, params, ...fields], dirty);
    return {
      ...result,
      uniforms: {
        u_engravingRows: result.offsets[0],
        u_engravingSettings: result.offsets[1],
        u_engravingLight: api.light(light)
      }
    };
  };
}

export function createSurfaceEngraving(api,table,maxSize,kernel) {
  const stylesTable=table(2048,14,maxSize), registry=new WeakMap();
  let previous=[];
  return (world,atlases,frame,finiteCurves) => {
    if(frame.engraving===false)return;
    const styles=[], keys=new Map(), curve=finiteCurves ? api.validate(world) : null;
    function register(style) {
      if(!style)return 0;
      const key=JSON.stringify(style);
      let id=keys.get(key);
      if(id)return id;
      if(styles.length>=64)throw new RangeError('Surface engraving exceeds 64 styles per view');
      styles.push(style); id=styles.length;keys.set(key,id);
      return id;
    }
    const used=new Set();
    if(curve)for(let i=0;i<world.count;i++){
      const id=curve.indices[i];
      if(id && kernel.isRow(world.data,i*16))used.add(id);
    }
    const curveSlots=curve ? curve.styles.map((style,i)=>used.has(i+1)?register(style):0) : undefined;
    const meshSlots=new WeakMap();
    for(const atlas of atlases)for(const chart of atlas?.surfaceCharts??[]){
      const definition=chart.engraving ?? (atlas.sceneMeshId ? world.look?.engraving : undefined);
      const style=definition && (typeof definition==='object' && registry.get(definition) || api.prepare(definition));
      if(style && typeof definition==='object')registry.set(definition,style);
      meshSlots.set(chart,register(style));
    }
    if(!styles.length)return;
    const signature=[...keys.keys()];
    if(signature.some((key,i)=>key!==previous[i]) || signature.length!==previous.length){
      const {params,fields}=packEngravingStyles(api,styles,0);
      const size=[params,...fields].reduce((sum,field)=>sum+Math.ceil(field.length/8192)*8192,0);
      if(size/8192>maxSize)throw new RangeError('Surface engraving exceeds the GPU map budget');
      const values=new Float32Array(size);
      let cursor=0;
      for(const field of [params,...fields]){values.set(field,cursor);cursor+=Math.ceil(field.length/8192)*8192;}
      stylesTable.upload(values);
      previous=signature;
    }
    return {key:JSON.stringify(signature),bind:()=>stylesTable.bind(),mesh:chart=>meshSlots.get(chart)??0,
      curve:i=>curveSlots?.[(curve?.indices[i]??0)-1]??0,
      uniforms:{u_engravingTex:14,u_engravingSettings:0,u_engravingLight:api.light(frame.engravingLight??[-.45,.8,.55]),
        u_screenPrintOn:frame.screenPrint!==false,u_brushworkOn:frame.brushwork!==false}};
  };
}
