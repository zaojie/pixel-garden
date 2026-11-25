import { finiteCurveFragment } from './finite-curve-shaders.mjs';

const tileSize = 8, layers = 64, rowWidth = 512, spanStride = 44;
const maxTileSpans = 1024, maxReferences = 4_000_000;
const tableUnits = { spans: 1, tiles: 2, solid: 5, proxy: 6, owner: 7, mask: 8, glass: 13 };
const zero = [0, 0, 0, 0];
const dot = (a,b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const none = [0, 0, 0];

function vector(view, p) {
  return [0,1,2].map(i => view[i]*p[0] + view[i+4]*p[1] + view[i+8]*p[2]);
}

function point(view, p) {
  return vector(view,p).map((v,i) => v + view[i+12]);
}

export function finiteCurveMode(world, frame) {
  const mode = frame.curveMode;
  if (mode !== undefined && mode !== 'curves' && mode !== 'splats')
    throw new RangeError('curveMode expects splats or curves');
  return Boolean(world.curveSplats && (mode === undefined ?
    (frame.curveSplats ?? world.look?.curveSplats ?? true) : mode === 'curves'));
}

// This is the same preparation used by the headless renderer. The GPU sees
// bounded analytic spans, with no tessellation or CPU coverage samples.
export function prepareFiniteCurves(kit, world, data, camera, width, height, options) {
  const kernel = kit.curveSplats.renderKernel, { view, clipNear: near, ortho } = camera;
  const lens = { view, near, focal: camera.projection[5]*height*.5, width, height, ortho };
  const artScale = options.artScale ?? 1;
  const artLens = artScale === 1 ? lens : { ...lens, width:width/artScale, height:height/artScale, focal:lens.focal/artScale };
  const columns = Math.ceil(width/tileSize), lines = Math.ceil(height/tileSize);
  const tiles = Array.from({ length: columns*lines }, () => []);
  const rows = new Uint8Array(world.count), spans = [], marks=[];
  const marking=options.restMarks || options.engraving;
  const { materials, materialIds } = world.renderMetadata ?? {};
  const posed = world.anim?.rig || world.anim?.bones || world.reconstructionPose;
  const pose = posed && kit.splatFrames.create(world,options.time,kit.M);
  const selected = options.drawOrder && new Set(options.drawOrder);
  const keep = i => (!options.keep || options.keep(i)) && (!selected || selected.has(i)) &&
    (!options.visible || options.visible(data[i*16+11]));
  const subdivision = kit.curveSplats.subdivision;
  const strandLOD = kit.curveSplats.strandLod;
  const retained=i=>options.engraving?.curve(i) || options.restMarks && world.curveRest?.[i*16+3];
  const strands = options.strandLod && strandLOD.coverageFrame(world,artLens,{data,keep,
    partitionAt:options.geometryCode,protectedLOD:options.protectedLOD,report:options.strandLODFrame,
    forbidden:i=>retained(i) || options.strandForbidden?.(i)});
  const settings = subdivision.prepare(options.curveSubdivision);
  const detail = subdivision.frame(world,artLens,settings,{ data, planes:options.planes, partitionAt:options.geometryCode,
    keep:i=>keep(i) && !strands?.hidden?.has(i) && !strands?.codes?.has(i) && !strands?.opacity?.has(i) &&
      !retained(i) });
  const programs = new Map();
  let references = 0, peakTile = 0, masked = false, strandMasked = false;

  for (let i=0; i<world.count; i++) {
    const at = i*16;
    if (!kernel.isRow(data,at)) continue;
    rows[i] = 1;
    if (!keep(i) || detail?.hidden.has(i) || strands?.hidden?.has(i)) continue;
    const replacement = detail?.geometry.get(i);
    const center = replacement?.center ?? Array.from(data.subarray(at,at+3));
    let opacity = Math.min(1,strands?.opacity?.get(i) ?? data[at+3]);
    if (!(opacity > 0)) continue;

    const tag = data[at+15], timedAt = i*kernel.timingStride, timing = world.curveTiming;
    const program = timing?.[timedAt] && world.curveTimePrograms?.[timing[timedAt]-1];
    let window;
    const basis = pose && pose(i);
    if (program) {
      if (!programs.has(program)) programs.set(program,program.sample(options.time));
      window = kernel.timeWindow(timing,timedAt,programs.get(program),program.trail,options.time);
      if (!window) continue;
      opacity = kernel.timedOpacity(opacity,window.fade,tag);
      if (basis) window = { ...window, direction:kit.M.app(basis.basis().anchor,window.direction) };
    }
    const clips = kernel.clipPlanes(world.curveClips,i,center,1,basis);
    const span = kernel.prepare(replacement?.data ?? data,replacement ? 0 : at,center,1,opacity,lens,clips,window);
    if (!span) continue;
    const [x0,y0,x1,y1] = span.bounds;
    if (x0 >= width || x1 < 0 || y0 >= height || y1 < 0) continue;
    const col0 = Math.max(0,Math.floor(x0/tileSize)), col1 = Math.min(columns-1,Math.floor(x1/tileSize));
    const row0 = Math.max(0,Math.floor(y0/tileSize)), row1 = Math.min(lines-1,Math.floor(y1/tileSize));
    const id = spans.length/spanStride;
    for (let y=row0; y<=row1; y++) for (let x=col0; x<=col1; x++) {
      const bin = tiles[y*columns+x];
      bin.push(id);
      peakTile = Math.max(peakTile,bin.length);
      if (++references > maxReferences || peakTile > maxTileSpans)
        throw new RangeError('Interactive finite curves exceeded the tile budget. Reduce curve density or use curveMode: splats.');
    }
    const caps = (span.clips ?? []).map(({ point:p, normal }) => {
      const n = vector(view,normal), at = point(view,p);
      return { plane:[...n,-dot(n,at)], at:[...at,0] };
    });
    const mode = Number(span.beta) + 2*Number(!!span.group) + 4*Number(!!span.major) + 8*Number(span.flattened);
    const rgb = data[at+7], glow = !world.renderMetadata?.muteEmission?.[i] && kernel.emissionAt(world.curveMaterials,i);
    const physical = options.pbrTargets && options.pbrTargets(i);
    const roughness = world.curveMaterials?.[i * kernel.materialStride];
    const albedo = physical && options.pbrUseAlbedo && world.materialAlbedo?.[i];
    const base = albedo ? [(albedo-1) >> 16, (albedo-1) >> 8 & 255, (albedo-1) & 255] :
      [rgb & 255, rgb >> 8 & 255, rgb >> 16 & 255];
    const color = base.map(
      (byte,k) => Math.min(1,byte/255 + (physical ? 0 : glow?.[k] ?? 0))
    );
    const geometryCode = options.geometryCode?.(i) ?? 0;
    const strandCode = strands?.codes?.get(i) ?? 0;
    masked ||= geometryCode !== 0;
    strandMasked ||= strandCode !== 0;
    const endCap=caps[1]?.at ?? zero;
    const softness=options.filter?.at(materials?.[materialIds?.[i]]) ?? 1;
    spans.push(...point(view,span.from),span.radius,
      ...vector(view,span.axis),span.length,
      ...(span.major ? vector(view,span.major) : none),mode,
      ...(span.minor ? vector(view,span.minor) : none),span.opacity,
      ...(caps[0]?.plane ?? zero),...(caps[1]?.plane ?? zero),
      ...color,i+1,span.group,span.tau,geometryCode,strandCode,
      ...(caps[0]?.at ?? zero),endCap[0],endCap[1],endCap[2],softness,
      ...(glow || none),physical ? (physical+.5*(roughness ? roughness-1 : world.splatRoughness?.[i] ?? .45))*
        (options.pbrReceives?.(i)===false?-1:1) : 0);
    if (marking) {
      const {V}=kit, stored=!replacement && world.curveRest?.[at+3] ? world.curveRest.subarray(at,at+16) : undefined;
      const shape=kernel.spanShape(replacement?.data ?? data,replacement ? 0 : at,1);
      const authored=world.curveStations?.subarray(i*kernel.stationStride+4,i*kernel.stationStride+7);
      const axis=!replacement && authored?.some(Boolean) && dot(authored,shape.axis)<0 ? V.mul(shape.axis,-1) : shape.axis;
      const restAxis=stored ? Array.from(stored.subarray(4,7)) : axis;
      const liveU=stored ? Array.from(stored.subarray(12,15)) : span.major ?? V.perp(axis);
      const restU=stored ? Array.from(stored.subarray(8,11)) : liveU;
      marks.push(...point(view,center),shape.half,...vector(view,axis),(stored?.[7]??shape.radius)/shape.radius,
        ...vector(view,liveU),options.restMarks?.slot(i)??0,
        ...(stored?.subarray(0,3)??center),stored?.[3]??shape.half,
        ...restAxis,options.engraving?.curve(i)??0,...restU,0);
    }
  }
  const starts = tiles.length*4, tileData = new Float32Array(starts + references);
  let cursor = starts;
  for (const [i, bin] of tiles.entries()) {
    tileData[i*4] = cursor;
    tileData[i*4+1] = bin.length;
    tileData.set(bin,cursor);
    cursor += bin.length;
  }
  return { spans:Float32Array.from(spans), tiles:tileData, columns, rows,
    count:spans.length/spanStride, references, peakTile, masked, strandMasked,
    ...(marking && {marks:Float32Array.from(marks)}),
    ...(detail && { subdivision:detail.stats }), ...(strands && { strandLod:strands.stats }) };
}

export function createFiniteCurvePass(gl, { target, table, program, maxSize, textureUnit, fullscreen, cutaway, spriteLOD, strandLOD, print, pbrGLSL }) {
  const spans = table(rowWidth,tableUnits.spans,maxSize);
  const tiles = table(rowWidth,tableUnits.tiles,maxSize);
  const programs = new Map();
  const owner = target(3);
  let anchorTarget, markTable;
  // The ordinary splat pipeline uses some of these units as well. Curve
  // compositing runs after all its geometry draws and before the art resolve.
  function draw(plan, buffer, base, camera, options) {
    const { mode = 'color', wet, water, env, layeredGlass, glass, fog, section, clip, cover, proxy, engraving, anchors, filterMatte, historyOwner, artScale = 1, pbrUniforms, bindPbr } = options;
    const isOwner = mode === 'owner';
    const etched=mode==='color' && !!engraving && !filterMatte;
    const physical = mode==='color' && !filterMatte && !!pbrUniforms;
    const sampleSpecular=physical && !!options.pbrSpecularSampling, pbrBands=physical && !!options.pbrBands;
    const key = [mode,!!wet,wet?.airPockets,wet?.layered,wet?.atlas,!!env?.media,!!env?.lit,!!section,!!layeredGlass,plan.masked,plan.strandMasked,etched,physical,sampleSpecular,pbrBands].join('/');
    if (!programs.has(key)) programs.set(key,program(finiteCurveFragment({
      mode, tileSize, layers, wet, layeredGlass, environment:env, section:section ? cutaway : undefined,
      spriteLOD:plan.masked ? spriteLOD : undefined,strandLOD:plan.strandMasked ? strandLOD : undefined,
      engraving:etched,print,pbr:physical ? (sampleSpecular?'#define INK_PBR_FOOTPRINT\n':'')+(pbrBands?'#define INK_PBR_BANDS\n':'')+pbrGLSL : undefined
    })));
    const shader = programs.get(key), { width, height } = base;
    spans.bind();
    tiles.bind();
    textureUnit(tableUnits.solid,base.textures[2]);
    textureUnit(tableUnits.proxy,proxy ?? base.textures[2]);
    if (isOwner) {
      textureUnit(tableUnits.owner,base.textures[0]);
      textureUnit(tableUnits.mask,base.textures[1]);
    }
    if (glass) textureUnit(tableUnits.glass,glass);
    if (etched) engraving.bind();
    if (anchors) {
      textureUnit(10,anchors.textures[0]);
      textureUnit(11,anchors.textures[1]);
    }
    if (etched || anchors) markTable.bind();
    if (physical) bindPbr();
    gl.disable(gl.DEPTH_TEST);
    if (mode === 'color') {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);
    } else gl.disable(gl.BLEND);
    fullscreen(shader,buffer.framebuffer,width,height,{
      u_curveSpans:tableUnits.spans, u_curveTiles:tableUnits.tiles,
      u_curveSolid:tableUnits.solid, u_curveProxy:tableUnits.proxy,
      u_curveOwner:tableUnits.owner, u_curveOwnerMask:tableUnits.mask, u_curveGlass:tableUnits.glass,
      u_curveViewport:[width,height],u_curveRange:camera.depthRange,
      u_curveInvProjection:[1/camera.projection[0],1/camera.projection[5]],u_curveTileColumns:plan.columns,
      u_curveEntry:clip,u_curveCover:cover ?? zero,u_curveView:camera.view,u_curveProjection:camera.projection,
      u_curveNear:camera.clipNear,u_curveFar:camera.clipFar,u_curveSetback:camera.setback,
      u_curveOrtho:camera.ortho,u_curveHasProxy:!!proxy,u_curveArtScale:artScale,u_filterMatte:!!filterMatte,u_historyOwner:!!historyOwner,u_curveFog:fog.color,u_curveFogRange:fog.range,
      u_curveMarks:12,u_curveBaseWorld:10,u_curveBaseRest:11,
      ...water,...env?.uniforms,...section,...(etched && engraving.uniforms),...(physical && pbrUniforms)
    });
    gl.disable(gl.BLEND);
  }
  return {
    prepare(plan) {
      if (plan.marks) {
        markTable??=table(rowWidth,12,maxSize);
        markTable.upload(plan.marks);
      }
      spans.upload(plan.spans); tiles.upload(plan.tiles);
    },
    draw,
    anchors(plan,base,camera,options,anchors) {
      anchorTarget??=target(2,{floating:true});
      anchorTarget.resize(base.width,base.height);
      draw(plan,anchorTarget,base,camera,{...options,mode:'anchors',anchors});
      return anchorTarget;
    },
    owner(plan, base, camera, options) {
      owner.resize(base.width,base.height);
      draw(plan,owner,base,camera,{ ...options, mode: 'owner' });
      return owner;
    }
  };
}
