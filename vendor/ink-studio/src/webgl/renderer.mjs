import { gaussianFilter } from './gaussian-filter.mjs';
import { surfaceHistory, createSurfaceHistory } from './surface-history.mjs';
import { createLivePixelBoundaries } from './live-pixel-boundaries.mjs';
import { createCoverageVote } from './coverage-vote.mjs';
import { coverageFilter, createCoverageFilter, filterUnit } from './coverage-filter.mjs';
import { createGaussianProxyRenderer, gaussianProxyPlan } from './gaussian-part-proxies.mjs';
import { captureGaussianTurntable, gaussianCoverageFrame, gaussianNormalFrame } from './gaussian-turntable.mjs';
import { createSplatPaintPass, splatPaintStyle } from './splat-paint-pass.mjs';
import { createGaussianHarmonics } from './gaussian-harmonics.mjs';
import { createSurfaceOrder, surfaceOrderRegion, surfaceOrderSettings } from './surface-order.mjs';
import { createGaussianMaterialBinding, createGaussianMaterialPalette, gaussianMaterialPlan } from './gaussian-materials.mjs';
import {createEngravingBinding,createSurfaceEngraving} from './engraving.mjs';
import {createRestMarks} from './rest-marks.mjs';
import { createFootprintBinding } from './footprints.mjs';
import { createArtPixelFinish } from './art-pixel-finish.mjs';
import { createRenderGrid } from './render-grid.mjs';
import { createMaterialControls } from './material-controls.mjs';
import { applyDiffuseRamps } from './diffuse-ramps.mjs';
import { createLightEmitters } from './light-emitters.mjs';
import { createNativeCelDraw, createNativeDepthProbe } from './native-cel-pixels.mjs';
import { createInspection, inspectionDepth } from './inspection.mjs';
import { waterReflectionPlan, selectReflectionWorld } from './reflections.mjs';
import { readHybridDepth } from './hybrid-depth.mjs';
import { createFiniteCurvePass, finiteCurveMode, prepareFiniteCurves } from './finite-curves.mjs';
import { createRenderGraph, graphOwner, graphCapture, graphEffectPixel } from './render-graph.mjs';
import { createPixelLightingPass } from './pixel-lighting.mjs';
import { createFrameComposite, frameCamera, captureFrameCamera } from './frame-composite.mjs';
import { createSurfacePipeline } from './surfaces.mjs';
import { createDepthMeshPass } from './depth-mesh-proxies.mjs';
import { createSpriteReconstructionPipeline } from './sprite-reconstruction.mjs';
import { createResources } from './resources.mjs';
import { createTargets } from './targets.mjs';
import { createSplatPipeline } from './pipeline.mjs';
import { pixelShaderSources, scenePaletteSize } from './shaders.mjs';
import { createDepthSorter, makeCamera, cameraPose, withCameraPose, captureStreamingCamera } from './view.mjs';
import { waterBackgroundShader } from './water-optics.mjs';
import { createEnvironment } from './environment.mjs';
import { environmentBackgroundShader } from './environment-shaders.mjs';
import { portalComposite } from './portal-composite.mjs';
import { createExpandedInstances, createInstanceDraw, groupVisibility } from './instance-draw.mjs';
import { instancePlan, validateSize } from './instances.mjs';
import { prepareMaskRows } from './contribution-plan.mjs';
import { createViewerInput } from './viewer-input.mjs';

const zero = new Float32Array(4);
const empty = new Float32Array(0);
const reconstructionSource = Symbol('reconstruction source');
const portalDepth = Symbol('portal depth'), hybridReflection = Symbol('reflection plane');
const portalLimit = 2;

function reconstruct(toolkit, world, frame, definition) {
  const api = toolkit.KIT?.reconstruction;
  if (!api) throw new Error('World reconstruction needs the Ink Studio KIT');
  return api.frame(world, frame.time ?? world.lookSnapshot?.time ?? 0, definition);
}
const rgb = hex => [(hex >> 16 & 255) / 255, (hex >> 8 & 255) / 255, (hex & 255) / 255];

function palette(values, max = scenePaletteSize) {
  const result = new Float32Array(max * 3);
  for (let i = 0; i < Math.min(max, values.length); i++) result.set(rgb(values[i]), i * 3);
  return result;
}

function viewProjection(projection, view) {
  const matrix = new Float32Array(16);
  for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) {
    let v = 0;
    for (let k = 0; k < 4; k++) v += projection[k * 4 + row] * view[col * 4 + k];
    matrix[col * 4 + row] = v;
  }
  return matrix;
}

function inverseView(view, eye) {
  return new Float32Array([
    view[0], view[4], view[8], 0, view[1], view[5], view[9], 0,
    view[2], view[6], view[10], 0, ...eye, 1
  ]);
}

function anchorPlane({ origin, normal: [x, y, z] }) {
  return [x, y, z, -(x * origin[0] + y * origin[1] + z * origin[2])];
}

function planeInView([x, y, z, d], { view, eye }) {
  return [view[0] * x + view[4] * y + view[8] * z, view[1] * x + view[5] * y + view[9] * z,
    view[2] * x + view[6] * y + view[10] * z, x * eye[0] + y * eye[1] + z * eye[2] + d];
}

function materialData(pixel, world) {
  const data = pixel.gpu(world);
  const lens = Float32Array.from(data.lens);
  const colors = Array.from(lens, (v, i) => Array.from(
    data.ramps.slice(i * 8, i * 8 + (Math.round(v) & 15)), Math.round));
  const group = colors.map((_, i) => i);
  const root = i => {
    while (group[i] !== i) i = group[i];
    return i;
  };
  for (let a = 0; a < lens.length; a++) for (let b = a + 1; b < lens.length; b++) {
    let shared = 0;
    for (const c of colors[a]) if (colors[b].includes(c)) shared++;
    if (shared >= 2) group[root(b)] = root(a);
  }
  lens.forEach((v, i) => { lens[i] = (Math.round(v) & 31) + root(i) * 32; });
  return { ...data, lens };
}

function simpleResolve(source) {
  return source
    .replaceAll('u_pixOn > 0.5', 'false')
    .replaceAll('u_ao > 0.0', 'false')
    .replaceAll('u_toneOn > 0.5', 'false')
    .replaceAll('u_remapLine > 0', 'false')
    .replaceAll('u_grade > 0.5', 'false')
    .replace('if (u_gk <= 0.0) return c;', 'return c;');
}

function sameValues(a, b) {
  return a?.length === b.length && b.every((value, i) => value === a[i]);
}

function createBackend(gl, toolkit, { specialize, reuseOrder, simpleOnly }) {
  const resources = createResources(gl);
  const sampleCoverage = toolkit.KIT?.procedural?.sampleCoverage;
  const coverageRules = toolkit.ContributionRules ?? toolkit.KIT?.procedural?.contributionRules;
  const stableDetailAPI = toolkit.StableDetail ?? toolkit.KIT?.stableDetail;
  const cutawayAPI = toolkit.KIT?.cutaway;
  let pipeline;
  try {
    const pixelSources = options => pixelShaderSources(toolkit.PixelLook, { packedTone: true, ...options });
    const sources = pixelSources();
    const program = source => resources.program(sources.POST_VS, source);
    let simple = simpleOnly ? program(simpleResolve(sources.POST_FS)) : undefined;
    let simpleGaps;
    const clean = program(sources.CLEAN_FS);
    // Rebirth hosts can opt into geometry specialization through the lower
    // level pipeline. The canvas path specializes its own fullscreen resolve.
    const motionAPI = toolkit.ArtDirection ?? toolkit.KIT?.artDirection;
    const footprintAPI = toolkit.KIT?.splatFootprints;
    const spriteLOD = toolkit.KIT?.sprite?.reconstruction;
    const spriteLodUnit = 28, meshDepthUnit = 29; // Temporarily borrows the surface emitter shadow slot.
    const pbrUnit = 30;
    let pbrRows, pbrTable, pbrRowSource, uploadedPbrRamps, pbrRest;
    let spriteLodTable, uploadedSpriteMask, spriteReconstructionPipeline;
    let footprintBinding,engravingBinding,restBinding,restTarget,surfaceEngraving;
    const engravingAPI=toolkit.KIT?.engraving;
    const stability=toolkit.ArtStability??toolkit.KIT?.artStability;
    const airflow = toolkit.KIT?.procedural?.airflow, climateAPI = toolkit.KIT?.environmentFrame;
    pipeline = createSplatPipeline(gl, { specialize: false, sampleCoverage, coverageRules, stableDetail: stableDetailAPI, cutaway:cutawayAPI?.glsl, airflow, footprintGLSL:footprintAPI?.glsl,screenPrintGLSL:engravingAPI?.print?.glsl, spriteReconstructionGLSL:spriteLOD?.glsl, pbrGLSL:toolkit.KIT?.pbr?.glsl });
    const emptyVao = resources.allocate('VertexArray');
    const maxSize = gl.getParameter(gl.MAX_TEXTURE_SIZE), maxBuffers = gl.getParameter(gl.MAX_DRAW_BUFFERS);
    const targets = createTargets(gl, resources);
    const { target, table, textureUnit } = targets;
    let coverageResolve, lastFilterDefinition, filterSettings;
    let splatPaintPass, coverageVotePass, gaussianMaterialBinding, gaussianMaterialPalette, surfaceOrdering;
    // The material-controls pass uses unit 31 before the splat draw.
    let gaussianProxies;
    const gaussianRows = createGaussianHarmonics(gl, resources, 31, maxSize);
    const environment = createEnvironment(gl,resources,targets,maxSize,toolkit);
    const emitterLighting=createLightEmitters(gl,resources,targets,maxSize,toolkit);
    const highlightAPI = toolkit.KIT?.highlightCels;
    const foamAPI = toolkit.KIT?.shoreFoam;
    const nativeCelAPI = toolkit.KIT?.nativeCelPixels;
    const retainedAPI = toolkit.KIT?.protectedPixels;
    let drawNativeCels, probeNativeDepth;
    const materialControls=createMaterialControls(gl,resources,targets,maxSize,toolkit);
    const skyPrograms = new Map();
    const dryGeometry = target(2), dryContours = target(2), plainOwner = target(), key = target(1, { integer: true });
    const wetVariants = new Map(), filteredTargets = new Map();
    let halfColor;
    let graphOwnership, normalOwner, wetGeometry, wetContours, waterTexture, packedWater, maskOwner, glassBoundary, dryGlassGeometry, wetGlassGeometry;
    let portalSpriteLayer;
    const portalPrograms = new Map();
    const portalTargets = { destinations: [] }, reflectedPortalTargets = { destinations: [] };
    const portalApertures = [];
    let lastWater, lastWaterTime, uploadedWater, waterWidth = 0, waterHeight = 0;
    let waterLookSource, waterLookPlan, waterOpticalPlan, mappedWater;
    let opticsDefinition, opticsWorld, opticsPlan, opticsTime;
    const cleaned = target(), finished = target();
    const holdSettings = surfaceHistory.prepare(true);
    let lightingHistory, paletteHistory, liveBoundaries;
    const detailAPI = toolkit.DetailBudgets ?? toolkit.KIT?.detailBudgets;
    const detailResolveUnit = 26; // Unit 7 can hold a palette constraint or layered glass during resolve.
    const qualityAPI = toolkit.KIT?.passQuality;
    const contract = toolkit.KIT?.renderContract;
    const lighting = toolkit.KIT?.splatRelighting, shotLighting = toolkit.KIT?.shotLighting, relightUnit = 21;
    const relitSprites = new Map();
    let relightData, relightRows, shotRows, lastRelighting;
    const opticalUnit = 22;
    let opticalData, opticalRows, lastOptical;
    const detailPlans = new WeakMap();
    const qualityPlans = new WeakMap();
    let detailTable, uploadedDetailFlags;
    let materialStyleTable, materialStyleUpload;
    const facingUnit = 24, facingPlans = new WeakMap();
    // Optical splat rows are only sampled during geometry, so the resolve can
    // use that unit for bones without replacing its glass and depth inputs.
    const facingBoneUnit = opticalUnit;
    const facingSurface = toolkit.SurfaceLook ?? toolkit.KIT?.surface;
    const facingAPI = facingSurface?.facingColors;
    let facingTable, uploadedFacing;
    let paintOverlay, paintData, lastPaintView;
    const data = table(2048, 0, maxSize), bones = table(768, 4, maxSize);
    const fine = table(1536, 7, maxSize), pattern = table(2048, 10, maxSize);
    const toneStops = resources.texture();
    let constraintRows, constraintSlots, constraintPlan, constraintKey, constraintWorld, constraintMetadata, constraintCount, constraintMap, maskAt, constraintLab, constraintPairs, constraintAnchor;
    const contributionPost = new Map(), constrainedPosts = new Map();
    let maskSites, siteState, maskTexture, glassWorld, glassData, glassIndices;
    const rasterKeys = new WeakMap(), warnedPalettes = new WeakSet();
    let rasterKey = 0;
    const sort = createDepthSorter(), transparencyAPI = toolkit.KIT?.procedural?.splatOrder;
    const afterimagesAPI = toolkit.KIT?.afterimages;
    const previewPipelines = [];
    const surfacePipelines = [], brickPipelines=new Map();
    const brickShaders={cache:new Map(),program:resources.program};
    let lastBrickLayers=[],lastReliefMeshes=[],lastSimulationMaterials;
    let depthMeshPass, finiteCurvePass, lastDepthProxy = false;
    let emitterSurfacePipeline, reflectionTarget, secondaryReflectionTarget, reflectionAtlas, reflectionAtlasSize, reflectionAtlasFramebuffer;
    let artPixelFinish, correctionStage;
    const clusterAPI = toolkit.KIT?.colorClusters, boundaryAPI = toolkit.KIT?.boundaryCels;
    const historiesByDepth = [];
    function historiesAt(depth) {
      return historiesByDepth[depth] ??= {
        transparency: new WeakMap(), afterimages: afterimagesAPI?.history(), clusters: clusterAPI?.history()
      };
    }
    let post, simpleUnavailable = false, keyProgram, copy, spriteProgram, spriteEnvProgram, spriteLayer, spriteTable, anchors;
    let lastWorld, lastData, lastCount, lastOutput, lastDetail, lastEnvironment, lastStaged, lastDiffuseRamps, fineValues, materials, anchorCount = 0;
    let packed = empty, historyValid = false, previousVP, previousPalette, previousDetailRanges, previousSection, lastQualityKey;
    let toneKey, tone, uploadedTone, sorted, instanceDraw, expandInstances, lastInstances;

    function orderFor(world, frame, camera, time, isNew, plan, width, height, scale,
      tolerance, histories, source, rainClocks, rainWind, quality, spriteMask, curveRows) {
      const far = spriteMask?.hasFar ? spriteMask : null;
      const keep = i => !curveRows?.[i] && (!quality || quality.keep[i]) && !spriteLOD?.hidden(far?.codes[i]);
      function submitted(indices, references) {
        if (!quality && !far && !curveRows) return indices;
        if (!references) return indices.filter(keep);
        let count=0;
        const visible=new Int32Array(indices.length);
        for(let i=0;i<indices.length;i+=2) if (keep(indices[i])) {
          visible[count++]=indices[i];
          visible[count++]=indices[i+1];
        }
        return visible.subarray(0,count);
      }
      const sceneTime = frame.time ?? world.lookSnapshot?.time ?? 0;
      const seeking = tolerance && sorted?.source === source && sorted.histories === histories && sceneTime < sorted.sceneTime;
      let history = histories.get(source);
      if (!tolerance || frame.order) histories.delete(source);
      else {
        if (!history) histories.set(source, history = transparencyAPI.history());
        if (frame.resetTransparency || seeking) history.reset();
      }
      if (frame.order) {
        pipeline.setOrder(submitted(frame.order,false));
        sorted = null;
        return;
      }
      const moving = world.moving ?? [], back = world.back ?? [];
      const canReuse = reuseOrder && !world.anim?.bones?.length;
      const clock = moving.length ? [time,...rainClocks,...rainWind] : [];
      const viewport = plan ? [width,height,scale,camera.tangent,camera.clipNear,+camera.ortho] : [];
      if (canReuse && !isNew && (frame.changedRows?.length === 0 || plan && !world.count && !frame.changedRows) &&
          !frame.resetTransparency && !seeking && sorted?.tolerance === tolerance && sameValues(sorted.clock,clock) &&
          (!tolerance || sorted.histories === histories) &&
          sorted.quality === quality && sorted.far === far &&
          (sorted.curveRows === curveRows || curveRows && sameValues(sorted.curveRows,curveRows)) &&
          sorted.revision === plan?.orderRevision && sameValues(sorted.viewport,viewport) &&
          sameValues(sorted.view, camera.view) && sameValues(sorted.moving, moving) && sameValues(sorted.back, back)) {
        sorted.sceneTime = sceneTime;
        return;
      }
      const visible = plan && groupVisibility(plan,camera,width,height,scale);
      const backRows = tolerance && back.length ? new Set(back) : null;
      const { data, opticalSplats } = world;
      const stabilize = tolerance ? (order, depths) => history.sort(order, {
        data: source.data, layout: frame.instances, view: camera.view, time: sceneTime, tolerance, depthAt: i => depths[i],
        transparentAt: i => transparencyAPI.isTransparent(data, i, data[i * 16 + 3], opticalSplats?.[i * 12 + 3], backRows)
      }) : undefined;
      const order = sort(world, camera.view, time, visible, plan?.rows, stabilize, rainClocks, rainWind, camera.ortho ? undefined : camera.eye);
      pipeline.setOrder(submitted(order,!!plan), !!plan);
      sorted = canReuse ? { view: camera.view, moving: Int32Array.from(moving), back: Int32Array.from(back),
        clock, tolerance, sceneTime, source, histories, revision:plan?.orderRevision, viewport, quality, far, curveRows } : null;
    }

    function prepareDetail(world, budget) {
      if (!budget?.affectsCleanup && !budget?.fineOff) return null;
      const definition = world.look.detailBudgets;
      const previous = detailPlans.get(world);
      if (previous?.definition === definition && previous.count === world.count &&
          previous.metadata === world.renderMetadata && previous.tiles === world.fine) return previous;
      const at = budget.bind(world);
      const flags = budget.affectsCleanup && Float32Array.from({ length: world.count }, (_, i) => budget.flags(at(i)));
      let fine = world.fine;
      if (fine && budget.fineOff) {
        for (let i = 0; i < world.count; i++) {
          const marker = i * 12 + 11;
          if (fine[marker] && at(i)?.fine === false) {
            if (fine === world.fine) fine = fine.slice();
            fine[marker] = 0;
          }
        }
      }
      const detail = { definition, count: world.count, metadata: world.renderMetadata, tiles: world.fine,
        fine, flags: flags && flags.some(Boolean) ? flags : null };
      detailPlans.set(world, detail);
      return detail;
    }

    function prepareQualityRows(world, frame, quality, key, particleStart) {
      if (!quality?.rows) return null;
      const prior = qualityPlans.get(world);
      if (frame.changedRows?.length === 0 && prior?.key === key && prior.data === world.data &&
          prior.metadata === world.renderMetadata && prior.count === world.count && prior.particleStart === particleStart)
        return prior.selection;
      const selection = qualityAPI.planRows(world, quality, { particleStart, silhouette: frame.silhouetteProtection });
      qualityPlans.set(world, { key, data: world.data, metadata: world.renderMetadata, count: world.count, particleStart, selection });
      return selection;
    }

    function bindMaterialStyles(world, group, plan, dirtyRows) {
      const source = world.materialPixelStyles, previous = materialStyleUpload;
      const fresh = !previous || previous.group !== group || previous.source !== source ||
        previous.count !== world.count;
      if (fresh) {
        let rows = source;
        if (plan) {
          rows = new Float32Array((world.count + plan.sharedCount) * 4);
          if (source) rows.set(source.subarray(0, world.count * 4));
          for (const asset of plan.assets) if (asset.materialPixelStyles)
            rows.set(asset.materialPixelStyles, (world.count + asset.offset) * 4);
        }
        materialStyleUpload = { group, source, rows, count: world.count };
        materialStyleTable.upload(rows, (world.count + (plan?.sharedCount ?? 0)) * 4);
      } else if (dirtyRows?.length && source) {
        if (plan) for (const row of dirtyRows) {
          const from = row * 512 * 4, end = Math.min(from + 512 * 4, world.count * 4);
          previous.rows.set(source.subarray(from, end), from);
        }
        materialStyleTable.upload(previous.rows, (world.count + (plan?.sharedCount ?? 0)) * 4, dirtyRows);
      } else materialStyleTable.bind();
    }

    function resolveProgram(needsFull) {
      if (specialize && !simpleUnavailable && !needsFull) {
        try {
          return simple ??= program(simpleResolve(sources.POST_FS));
        } catch (error) {
          if (gl.isContextLost()) throw error;
          simpleUnavailable = true;
        }
      }
      return post ??= program(sources.POST_FS);
    }

    function fullscreen(shader, destination, width, height, values) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, destination);
      gl.viewport(0, 0, width, height);
      gl.bindVertexArray(emptyVao);
      gl.useProgram(shader.p);
      shader.set(values);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    function prepareTone(definition) {
      if (!definition) return null;
      const next = JSON.stringify(definition);
      if (toneKey !== next) {
        tone = toolkit.PixelLook.prepareTone(definition, toolkit.KIT?.procedural?.materials?.blueNoise);
        toneKey = next;
      }
      return tone;
    }

    function bindTone(prepared) {
      if (!prepared) return { u_toneOn: 0 };
      textureUnit(12, toneStops);
      if (uploadedTone !== prepared) {
        // The first row holds the stops and the rest hold the threshold tile.
        const width = Math.max(prepared.stops.length, prepared.width), height = prepared.height + 1;
        const table = new Float32Array(width * height * 4);
        prepared.stops.forEach(([at, hex], i) => table.set([...rgb(parseInt(hex.slice(1), 16)), at], i * 4));
        for (let y = 0; y < prepared.height; y++) for (let x = 0; x < prepared.width; x++)
          table[((y + 1) * width + x) * 4] = prepared.thresholds[y * prepared.width + x];
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, table);
        uploadedTone = prepared;
      }
      return { u_toneOn: 1, u_toneCut: prepared.cutoff, u_toneAmt: prepared.amount, u_toneN: prepared.stops.length,
        u_toneStops: 12, u_toneThSize: [prepared.width, prepared.height] };
    }

    function prepareConstraints(world, definition, skipMasks) {
      if (definition === undefined || definition === false) return null;
      const key = JSON.stringify([definition, world.pal, world.paletteRoles, skipMasks],(name,value)=>{
        if (name !== 'frames' || !Array.isArray(value)) return value;
        if (!rasterKeys.has(value)) rasterKeys.set(value,++rasterKey);
        return { rasterFrames:rasterKeys.get(value) };
      });
      if (key === constraintKey && world === constraintWorld && world.renderMetadata === constraintMetadata &&
          world.count === constraintCount) return constraintPlan;
      const plan = toolkit.PixelLook.preparePaletteConstraints(definition, {
        palette: world.pal, roles: world.paletteRoles, blueNoise: toolkit.KIT?.procedural?.materials?.blueNoise, skipMasks, skipDitherRegions: true,
        maskSupported:mask=>Boolean(coverageRules?.supportsMask(mask))
      });
      if (plan) {
        constraintRows ??= resources.texture();
        constraintSlots ??= table(1024, 4, maxSize);
        constraintMap = 49 + Math.ceil(Math.max(...plan.palettes.map(row => row.ranks.length)) / 4);
        maskAt = constraintMap + Math.max(...plan.palettes.map(row => (row.grade?.length ?? 0) / 3));
        plan.masked = plan.palettes.some(row => row.maskSource);
        const masks = plan.masked && prepareMaskRows(plan.palettes,coverageRules,maxSize);
        plan.textured = Boolean(masks?.texture);
        constraintLab = maskAt + (plan.masked ? 6 : 0);
        constraintPairs = constraintLab + (plan.palettes.some(row=>row.matchId) ? 48 : 0);
        constraintAnchor=constraintPairs+Math.max(...plan.palettes.map(row=>row.pairs?.length??0));
        const width=constraintAnchor+2;
        const rows = new Float32Array(width * 4 * plan.palettes.length);
        plan.palettes.forEach((palette, i) => {
          const offset = i * width * 4;
          const anchor=palette.anchor, mode=anchor ? ['screen','world','object','view'].indexOf(anchor.dither) : -1;
          rows.set([...(anchor?.origin??[0,0,0]),anchor?.spacing??1,mode,...(palette.tuning?.fade??[0,0]),0],
            offset+constraintAnchor*4);
          rows.set([palette.count, palette.packed.length, palette.width, palette.height], offset);
          if (!palette.mapOnly) palette.swatches.forEach((swatch, j) => rows.set(swatch, offset + (j + 1) * 4));
          rows[offset + 7] = palette.amount;
          rows[offset + 11] = Number(palette.source === 'graded');
          rows[offset + 15] = palette.mapOnly ? 2 : Number(Boolean(palette.grade));
          rows[offset + 19] = palette.matchId ?? 0;
          rows[offset + 23] = palette.pairs ? palette.pairs.length+1 : 0;
          rows[offset + 27] = palette.tuning?.scale ?? 1;
          rows[offset + 31] = palette.tuning?.screenGain ?? 1;
          // Two 24-bit words per permission fit exactly in the float texture.
          palette.reserveAccess?.forEach((access, i) => {
            const slot = palette.count + i, word = Math.floor(slot / 24), bit = 1 << (slot % 24);
            if (access & 1) rows[offset + (8 + word) * 4 + 3] += bit;
            if (access & 2) rows[offset + (10 + word) * 4 + 3] += bit;
          });
          palette.pairs?.forEach(({slots,lightness},j)=>
            rows.set([...slots,...(lightness ?? [-1,1])],offset+(constraintPairs+j)*4));
          palette.lab?.forEach((row,j)=>rows.set(row,offset+(constraintLab+j)*4));
          rows.set(palette.ranks, offset + 49 * 4);
          if (palette.grade) for (let j = 0; j < palette.grade.length / 3; j++)
            rows.set(palette.grade.subarray(j * 3, j * 3 + 3), offset + (constraintMap + j) * 4);
          if (masks?.rows?.[i]) rows.set(masks.rows[i],offset+maskAt*4);
        });
        textureUnit(8, constraintRows);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, plan.palettes.length, 0, gl.RGBA, gl.FLOAT, rows);
        constraintSlots.upload(Float32Array.from(plan.slots(world)));
        if (masks?.texture) {
          const {width,height,values} = masks.texture;
          maskTexture ??= resources.texture();
          textureUnit(14,maskTexture);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,width,height,0,gl.RED,gl.FLOAT,values);
        }
      }
      constraintKey = key;
      constraintWorld = world;
      constraintMetadata = world.renderMetadata;
      constraintCount = world.count;
      constraintPlan = plan;
      return plan;
    }

    function bindMaskSites(world,dirty) {
      maskSites ??= table(1024,10,maxSize);
      const {data,pix,renderMetadata,surfaceData,anim,count} = world, rest = world.rest ?? anim?.base ?? data;
      const sources = [world,pix,renderMetadata,surfaceData,rest,count], reusable = sameValues(siteState?.sources,sources);
      const values = reusable ? siteState.values : new Float32Array(count*8);
      const rows = reusable ? (rest === data ? dirty : []) : undefined;
      const fill = (first,end) => {
        const normals = renderMetadata?.normals;
        for (let i=first; i<end; i++) {
          const a=i*3, b=i*8, o=i*16;
          let array, at;
          if (pix?.[b]) { array=pix; at=b+3; }
          else if (normals?.[a] || normals?.[a+1] || normals?.[a+2]) { array=normals; at=a; }
          else if (surfaceData?.[b]) { array=surfaceData; at=b+1; }
          for (let n=0;n<3;n++) {
            values[b+n]=array?.[at+n]??0;
            values[b+4+n]=rest[o+n];
          }
        }
      };
      if (rows === undefined) fill(0,count);
      else for (const row of rows) fill(row*512,Math.min(count,(row+1)*512));
      maskSites.upload(values,values.length,rows);
      siteState={sources,values};
    }

    function uploadBones(values) {
      if (!values) {
        bones.bind();
        return;
      }
      if (packed.length !== values.length) packed = new Float32Array(values.length);
      for (let b = 0; b < values.length; b += 12) for (let row = 0; row < 3; row++) {
        const from = b + row * 3, to = b + row * 4;
        packed[to] = values[from];
        packed[to + 1] = values[from + 1];
        packed[to + 2] = values[from + 2];
        packed[to + 3] = values[b + 9 + row];
      }
      bones.upload(packed);
    }

    function updateSprites(world, pixel) {
      anchorCount = 0;
      const descriptions = pixel.edgeTable?.(world);
      if (!descriptions?.any || !pixel.anchors) return;
      // The toolkit caches anchors on its input. Keep that cache private to
      // this upload so a replaced material table can rebuild the sprite list.
      const { count, data, detail, pix, pats } = world;
      const points = world.reconstructionPose ? world._anchors : pixel.anchors({ count, data, detail, pix, pats });
      if (!points?.n) return;
      spriteProgram ??= resources.program(sources.SPR_VS, sources.SPR_FS);
      spriteLayer ??= target(1, { depth: true });
      spriteTable ??= resources.texture();
      anchors ??= table(768, 14, maxSize);
      const sprites = new Float32Array(12 * 96 * 4);
      for (let i = 0; i < 32; i++) sprites.set(descriptions.per.subarray(i * 8, i * 8 + 8), i * 48);
      descriptions.sprites.forEach((sprite, i) => {
        const start = (32 + i) * 48;
        sprites.set([sprite.w, sprite.h, sprite.ax, sprite.sink], start);
        sprites.set(sprite.cells.slice(0, 44), start + 4);
      });
      textureUnit(12, spriteTable);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 12, 96, 0, gl.RGBA, gl.FLOAT, sprites);
      anchors.upload(points.data);
      anchorCount = points.n;
    }

    function waterUniforms(scene, camera, time, plane, captures, width, height, detailOn, frame) {
      if (!scene) return null;
      waterTexture ??= resources.texture();
      textureUnit(3, waterTexture);
      if (scene !== uploadedWater || !packedWater) {
        const { width, height, data, uniforms } = scene.pack();
        if (width === waterWidth && height === waterHeight)
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, width, height, gl.RGBA, gl.FLOAT, data);
        else {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, data);
          waterWidth = width;
          waterHeight = height;
        }
        packedWater = uniforms;
        uploadedWater = scene;
      }
      return { ...packedWater, u_waterData: 3, u_waterView: camera.view, u_waterEye: camera.eye, u_arrivalPlane: plane,
        u_waterCamera: [camera.tangent * camera.aspect, camera.tangent, +camera.ortho], u_waterTime: time,
        u_waterReflections: captures ? 23 : 0, u_waterReflectionCount: captures?.count ?? 0,
        u_waterReflectionActive: captures?.active ?? zero, u_waterReflectionHeights: captures?.heights ?? zero,
        u_waterReflectionGrid: captures?.grid ?? [2,1], u_waterReflectionViewport:[width,height],
        u_waterReflectionDetailOn:detailOn,u_waterProjection:camera.projection,
        u_waterReflectionProxyBase:packedWater.u_waterReflectionProxyBase??0,u_waterProxySetback:camera.setback??0,
        u_waterProxyOn:frame.reflections!==false&&frame.reflectionProxies!==false };
    }

    function wetVariant(water, media) {
      const airPockets = Boolean(water.u_waterAirCount), layered = Boolean(water.u_waterLayered), atlas = Boolean(water.u_waterAtlas);
      const key = +airPockets + 2 * layered + 4 * atlas + 8 * Number(media);
      if (!wetVariants.has(key)) wetVariants.set(key, {
        airPockets, layered, atlas, sources: pixelSources({ water: true, airPockets, layered, atlas, environment:media })
      });
      return wetVariants.get(key);
    }
    function drawGeometry(shader, buffer, width, height, options, water, wet, env) {
      let meshDepth, previousMeshBinding, previousMarks;
      if (options.surfaceFrame?.depthProxy) {
        depthMeshPass ??= createDepthMeshPass(gl,meshDepthUnit,cutawayAPI?.glsl);
        meshDepth=depthMeshPass.prepare(options.surfaceFrame,width,height);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, buffer.framebuffer);
      gl.viewport(0, 0, width, height);
      if (options.u_coverageVotePass) {
        gl.clearBufferfv(gl.COLOR, 0, [0,0,0,1]);
        gl.clearBufferfv(gl.COLOR, 1, zero);
        gl.disable(gl.BLEND);
      } else if (options.u_paintPass || options.u_gaussianMaterialPass) {
        gl.clearBufferfv(gl.COLOR, 0, zero);
        gl.clearBufferfv(gl.COLOR, 1, zero);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      } else if (options.portalForeground) {
        for (let i = 0; i < buffer.textures.length; i++) gl.clearBufferfv(gl.COLOR, i, zero);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      } else if (options.u_idx) {
        gl.disable(gl.BLEND);
        if(options.u_idx===3){
          gl.clearBufferfv(gl.DEPTH,0,zero);
          gl.enable(gl.DEPTH_TEST);
          gl.depthFunc(gl.GREATER);
        }
        gl.clearBufferfv(gl.COLOR, 0, zero);
        if (options.u_idx === 1 && buffer.textures.length > 1) gl.clearBufferfv(gl.COLOR,1,zero);
        if (options.u_idx === 1 && buffer.textures.length > 2) gl.clearBufferfv(gl.COLOR,2,zero);
      } else {
        if (env) {
          let programs = wet ? (wet.skyPrograms ??= new Map()) : skyPrograms;
          const glass = !!options.layeredGlass, skyKey=+glass+2*!!env.lit;
          if (!programs.has(skyKey)) programs.set(skyKey,program(wet ?
            waterBackgroundShader(wet.airPockets,wet.layered,wet.atlas,true,glass,env.lit) : environmentBackgroundShader(glass,env.lit)));
          gl.disable(gl.BLEND);
          fullscreen(programs.get(skyKey),buffer.framebuffer,width,height,{ ...water, ...env.uniforms,
            u_envViewport:[width,height],u_waterViewport:[width,height] });
        } else {
          if (water) {
            const backgroundProgram = options.layeredGlass
              ? (wet.glassBackground ??= program(waterBackgroundShader(wet.airPockets, wet.layered, wet.atlas, false, true)))
              : (wet.background ??= program(waterBackgroundShader(wet.airPockets, wet.layered, wet.atlas)));
            gl.disable(gl.BLEND);
            fullscreen(backgroundProgram, buffer.framebuffer, width, height, { ...water,
              u_waterViewport: [width, height], u_waterBackground: options.background.slice(0, 3) });
          } else {
            gl.clearBufferfv(gl.COLOR, 0, options.background);
            gl.clearBufferfv(gl.COLOR, 1, [1, 0, 0, 1]);
          }
          if (options.layeredGlass && !water) {
            const at = buffer.textures.length-2;
            gl.clearBufferfv(gl.COLOR,at,options.background);
            gl.clearBufferfv(gl.COLOR,at+1,[1,1,1,1]);
          }
        }
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      }
      const surfaces = options.surfaces, previews = options.scaleSurfaces, spriteRows = options.spriteRows;
      const hasSurfaces = Boolean(surfaces || previews?.some(Boolean) || spriteRows?.length);
      const hasProxies = !options.u_markPass && !!options.proxy && gaussianProxies.hasGeometry;
      const hasPlanes = !!shader.footprints && footprintBinding.hasPlanes && options.u_idx!==3;
      if (hasSurfaces || hasProxies || hasPlanes || options.surfaceFrame?.lighting?.display) {
        const weightsOnly = options.u_splatPaintPass || options.u_coverageVotePass || options.u_gaussianMaterialPass;
        const surfaceOptions = { geometryUpdatesSince: toolkit.SpriteSurface?.geometryUpdatesSince,pbrGLSL:toolkit.KIT?.pbr?.glsl,
          updatesSince: toolkit.SpriteSurface?.updatesSince ?? toolkit.KIT?.ecology?.atlasUpdates, cutaway: cutawayAPI?.glsl,
          projectionPlan: toolkit.SpriteSurface?.projectionPlan, spriteLOD:spriteLOD?.glsl,
          samplingPlan: toolkit.SpriteSurface?.samplingPlan, samplingGLSL: toolkit.SpriteSurface?.samplingGLSL,print:engravingAPI?.print?.glsl };
        buffer.ensureDepth();
        gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.depthMask(true);
        gl.clearBufferfv(gl.DEPTH,0,[1]);gl.disable(gl.BLEND);
        if (weightsOnly) gl.colorMask(false,false,false,false);
        const surfaceFrame = { ...options.surfaceFrame, water, wet, width, height,
          glassLayers: !!options.layeredGlass,
          anchors:options.u_markPass ? options.surfaceFrame?.anchors : undefined,
          pbr:!weightsOnly && !options.u_markPass && !options.coverageFilter ? options.surfaceFrame?.pbr : undefined,
          coverageFilter: options.coverageFilter, historyOwner:!!options.historyOwner };
        if(surfaceFrame.engraving && !surfaceFrame.index){
          gl.activeTexture(gl.TEXTURE0+14);previousMarks=gl.getParameter(gl.TEXTURE_BINDING_2D);
        }
        if (surfaces) for (const [i, atlas] of (Array.isArray(surfaces) ? surfaces : [surfaces]).entries()) {
          if (!atlas) continue;
          let surface;
          if (atlas.brickField) {
            if (!brickPipelines.has(atlas.sceneMeshId))
              brickPipelines.set(atlas.sceneMeshId,createSurfacePipeline(gl,{...surfaceOptions,sharedShaders:brickShaders}));
            surface=brickPipelines.get(atlas.sceneMeshId);
          } else surface=surfacePipelines[i] ??= createSurfacePipeline(gl,surfaceOptions);
          if (atlas.sceneMeshId) gl.depthFunc(gl.LESS);
          surface.draw(atlas, { ...surfaceFrame, historyFirst:options.historyOwner?.(atlas),
            samplingScale: options.u_materialGridScale ?? 1,
            geometryCode: options.surfaceFrame?.meshCodes?.get(atlas.sceneMeshId),
            partGrid: options.spriteGrid });
          if (atlas.sceneMeshId && surfaceFrame.onModelingWork) {
            const {geometryBytes,textureBytes,blockBytes}=surface.stats;
            if (geometryBytes || textureBytes) surfaceFrame.onModelingWork({kind:'upload',
              sourceIds:[atlas.sceneMeshId], counts:{geometryBytes,textureBytes,blockBytes}});
          }
          if (atlas.blockTextures && !surfaceFrame.index && !options.u_markPass)
            surfaceFrame.sceneryDeliveryFrame?.({id:atlas.sceneMeshId,...surface.stats});
          if (atlas.sceneMeshId) gl.depthFunc(gl.LEQUAL);
        }
        if (previews) {
          gl.depthFunc(gl.LESS);
          for (const [i, atlas] of previews.entries()) if (atlas) {
            previewPipelines[i] ??= createSurfacePipeline(gl, surfaceOptions);
            previewPipelines[i].draw(atlas, { ...surfaceFrame, historyFirst:options.historyOwner?.(atlas), samplingScale: options.u_materialGridScale ?? 1 });
          }
          gl.depthFunc(gl.LEQUAL);
        }
        if(spriteRows?.length){
          spriteReconstructionPipeline ??= createSpriteReconstructionPipeline(gl, spriteLOD, cutawayAPI?.glsl);
          spriteReconstructionPipeline.draw(spriteRows, {
            ...surfaceFrame, artScale:options.spriteGrid??[1,1]
          });
        }
        if(!options.u_markPass&&options.surfaceFrame?.lighting?.display){emitterSurfacePipeline??=createSurfacePipeline(gl,{cutaway:cutawayAPI?.glsl});emitterSurfacePipeline.draw(options.surfaceFrame.lighting.display,{...surfaceFrame,pbr:null,lighting:null,historyFirst:options.historyOwner?.(options.surfaceFrame.lighting.display)});}
        if (hasProxies) {
          if (shader.gaussianRows) gaussianRows.rebind();
          if (options.u_splatPaintPass) gl.colorMask(true,true,true,true);
          gaussianProxies.draw({ ...options.proxy, width, height, wet, glassLayers: surfaceFrame.glassLayers, scale: options.u_materialGridScale ?? 1 },
            options.u_paintPass || options.u_gaussianMaterialPass ? 3 : options.u_idx || 0);
        }
        if(previousMarks!==undefined)textureUnit(14,previousMarks);
        data.bind();
        if (weightsOnly) gl.colorMask(true,true,true,true);
        gl.depthMask(false);
        if(!options.u_idx&&!options.u_coverageVotePass){gl.enable(gl.BLEND);gl.blendFunc(gl.ONE,gl.ONE_MINUS_SRC_ALPHA);}
      }else if(options.u_idx!==3)gl.disable(gl.DEPTH_TEST);
      shader.bindFilterRows?.();
      // Surface atlas allocation and material draws can change texture bindings.
      if (shader.gaussianRows) gaussianRows.rebind();
      if (shader.footprints) footprintBinding.rebind();
      if (shader.gaussianMaterials) gaussianMaterialBinding.rebind();
      if (shader.pbr) pbrTable.bind();
      gl.useProgram(shader.p);
      shader.set({
        ...options,u_markPass:!!options.u_markPass,u_filterMatte:!!options.coverageFilter,u_meshDepthProxy:meshDepthUnit,u_geometryDepth:+(hasSurfaces||hasProxies||hasPlanes),
        u_proxyGrid: options.proxy ? [options.proxy.width/width, options.proxy.height/height] : [1,1],
        u_portalCoverPlane: options.u_portalCoverPlane ?? zero,
        u_paintPass: !!options.u_paintPass, u_splatPaintPass: !!options.u_splatPaintPass, u_coverageVotePass: !!options.u_coverageVotePass,
        u_historyOwner:!!options.historyOwner,
        u_gaussianMaterialPass: !!options.u_gaussianMaterialPass, u_surfacePeel: false,
        u_gaussianNormalCutoff: options.u_gaussianNormalCutoff ?? 0,
        u_materialGridScale: options.u_materialGridScale ?? 1,
        u_facingGridScale: options.u_materialGridScale ?? 1,
        u_spriteLodGrid: options.spriteGrid ?? [1,1],
        u_viewport: [width, height], u_lookViewport: [width,height],
        ...(options.lookGrid && { u_lookViewportScale: [options.lookGrid[0]/width,options.lookGrid[1]/height] }),
        u_focal: height / 2 / options.tangent
      });
      if (meshDepth) {
        gl.activeTexture(gl.TEXTURE0+meshDepthUnit);
        previousMeshBinding=gl.getParameter(gl.TEXTURE_BINDING_2D);
        gl.bindTexture(gl.TEXTURE_2D,meshDepth);
      }
      if(hasPlanes){
        // Only opaque retained planes enter the depth prepass. Ordinary soft
        // splats and glass keep their established contribution blending.
        gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.depthMask(true);
        gl.colorMask(false,false,false,false);shader.set({u_idx:4});pipeline.draw();
        gl.colorMask(true,true,true,true);gl.depthMask(false);shader.set({u_idx:options.u_idx??0});
      }
      if(options.u_idx===3){
        pipeline.drawSubset(options.indices);
        gl.disable(gl.DEPTH_TEST);
      } else pipeline.draw();
      if (meshDepth) textureUnit(meshDepthUnit,previousMeshBinding);
      gl.disable(gl.DEPTH_TEST);gl.depthMask(true);
    }

    function drawSprites(camera, world, width, height, time, pixelOutline, water, pal, wet, clip, foreground, env, relighting, section) {
      const buffer = foreground ? portalSpriteLayer : spriteLayer;
      buffer.resize(width, height);
      data.bind();
      textureUnit(13, null);
      textureUnit(12, spriteTable);
      anchors.bind();
      gl.bindFramebuffer(gl.FRAMEBUFFER, buffer.framebuffer);
      gl.viewport(0, 0, width, height);
      gl.clearBufferfv(gl.COLOR, 0, zero);
      if (foreground) gl.clearBufferfv(gl.COLOR, 1, zero);
      gl.clearBufferfv(gl.DEPTH, 0, [1]);
      gl.disable(gl.BLEND);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      let sprites;
      if (relighting || section) {
        const key = [!!wet, wet?.airPockets, wet?.layered, wet?.atlas, !!env?.media, !!relighting,!!env?.lit, !!section].join('/');
        if (!relitSprites.has(key)) {
          const shaders = pixelSources({
            water: !!wet, airPockets: wet?.airPockets, layered: wet?.layered, atlas: wet?.atlas,
            environment: !!env?.media,emitterFog:!!env?.lit, relighting, cutaway:section ? cutawayAPI.glsl : undefined
          });
          relitSprites.set(key, resources.program(shaders.SPR_VS, shaders.SPR_FS));
        }
        sprites = relitSprites.get(key);
      } else sprites = wet ? (wet.sprites ??= resources.program(wet.sources.SPR_VS, wet.sources.SPR_FS)) :
        env?.media ? (spriteEnvProgram ??= resources.program(sources.SPR_VS,
          pixelSources({environment:true}).SPR_FS)) : spriteProgram;
      gl.useProgram(sprites.p);
      sprites.set({
        u_anc: 14, u_sprTex: 12, u_dat: 0, u_depth: 2, u_idxTex: 9, u_pixTex: 10,
        u_view: camera.view, u_proj: camera.projection, u_vp: [width, height],
        u_splatPlane: clip, u_spriteCover: foreground ?? zero,
        u_rng: camera.depthRange, u_time: time, u_keep: toolkit.KIT.pixel.anchorKeep(world, camera.footprint),
        u_ortho: +camera.ortho, u_rmpN4: materials.gpu.lens, u_pview: camera.view, u_peye: camera.eye,
        u_tanF: camera.tangent, u_asp: camera.aspect, u_portho: +camera.ortho,
        u_eGap: (pixelOutline?.gap ?? 0.3) / (2 * camera.radius), u_tol: 3 * camera.footprint + 0.02,
        ...water, ...env?.uniforms, ...section,
        ...(relighting && { u_relight: relightUnit }),
        ...(water || env?.media || relighting ? { u_waterSpritePalette: pal, u_rmp4: materials.gpu.ramps } : {})
      });
      gl.bindVertexArray(emptyVao);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, anchorCount);
      gl.disable(gl.DEPTH_TEST);
      textureUnit(13, buffer.textures[0]);
      return buffer.textures[1];
    }

    function captureWater(world, frame, camera, width, height, trailStart, quality) {
      const plan = waterReflectionPlan(world, frame, camera, width, height, toolkit.KIT?.reflection, maxSize,
        quality?.captures && (ids => qualityAPI.reflectionPlan(quality, ids)));
      if(plan?.sources && frame.reflectionSourceFrame)frame.reflectionSourceFrame({
        sources:plan.sources,planarCaptures:plan.captures.length,
        planarPixels:plan.captures.reduce((sum,entry)=>sum+entry.width*entry.height,0)
      });
      if (!plan?.captures.length) return null;
      reflectionAtlas ??= resources.texture();
      const size = [plan.width*plan.grid[0],plan.height*plan.grid[1]];
      textureUnit(23,reflectionAtlas);
      if (!sameValues(reflectionAtlasSize,size)) {
        gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,...size,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
        reflectionAtlasSize=size;
      }
      // Read the current geometry each frame; no baked or manually synchronized mirror splats.
      for (const capture of plan.captures) {
        const { width, height } = capture;
        const scaled = width !== plan.width || height !== plan.height;
        const buffer = scaled ? (secondaryReflectionTarget ??= target()) : (reflectionTarget ??= target());
        buffer.resize(width, height);
        let source = world, instances = frame.instances;
        if (instances && capture.plane.select) {
          expandInstances ??= createExpandedInstances();
          source = expandInstances(world,instances,frame.changedRows).world;
          instances = undefined;
        }
        const t = frame.time ?? world.lookSnapshot?.time ?? 0;
        const time = contract?.clock(world, frame).scene ?? t;
        source = selectReflectionWorld(source,capture.plane,time,toolkit);
        const reflectedFrame = {...frame, instances, order: undefined, changedRows: undefined,
          water: false, reflections: false, raw: frame.raw ?? false, outline: false, dither: 0,
          damp: false, edges: false, clean: false, tone: false, fade: 0, [graphOwner]: undefined,
          [hybridReflection]: capture.plane.id };
        present(source, reflectedFrame, width, height, buffer.framebuffer, capture.camera,
          capture.equation, 0, true, trailStart);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER,buffer.framebuffer);
        textureUnit(23,reflectionAtlas);
        const x = (capture.slot % 2) * plan.width, y = Math.floor(capture.slot / 2) * plan.height;
        if (scaled) {
          reflectionAtlasFramebuffer ??= resources.allocate('Framebuffer');
          gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, reflectionAtlasFramebuffer);
          gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reflectionAtlas, 0);
          gl.blitFramebuffer(0, 0, width, height, x, y, x + plan.width, y + plan.height, gl.COLOR_BUFFER_BIT, gl.LINEAR);
        } else {
          gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, x, y, 0, 0, width, height);
        }
      }
      const clusters = frame.reflectionDetail!==false && world.water.reflectionClusters;
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,clusters ? gl.NEAREST_MIPMAP_LINEAR : gl.NEAREST);
      if (clusters) {
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);
        gl.generateMipmap(gl.TEXTURE_2D);
      }
      gl.activeTexture(gl.TEXTURE0);
      return plan;
    }

    const reconstructionInstances = createExpandedInstances(), portalReconstructionInstances = [];
    function reconstructFrame(world, frame, expand) {
      const definition = frame.reconstruction ?? world.look?.reconstruction;
      if (definition === undefined || definition === false) return { world, frame };
      const source = world, layout = frame.instances;
      const expanded = layout ? expand(world, layout, frame.changedRows) : null;
      const posed = reconstruct(toolkit, expanded?.world ?? world, frame, definition);
      if (posed === (expanded?.world ?? world)) return { world, frame };
      return { world: posed, frame: { ...frame, [reconstructionSource]: { source, layout }, instances: undefined,
        changedRows: undefined } };
    }

    function render(world, frame, width, height, output, cameraOverride, plane = zero, cover, reflectionCapture = false,
      reflectedTrailStart = 0) {
      const reconstruction = frame[reconstructionSource];
      const source = reconstruction?.source ?? world;
      const viewHistories = historiesAt(plane === zero ? 0 : frame[portalDepth] ?? 1);
      const histories = viewHistories.transparency;
      const camera = cameraOverride ?? makeCamera(world, frame, width, height);
      const viewPoses = toolkit.KIT?.viewPoses;
      if (viewPoses?.forWorld(world)?.active)
        world = viewPoses.frame(world, { ...cameraPose(camera), ortho:camera.ortho }, {enabled:frame.viewPoses!==false});
      if (world.shadowLod && frame.shadowArt !== false)
        world = toolkit.KIT?.shadowArt?.frame(world, camera.shadowFootprint) ?? world;
      const harmonics = world.gaussianHarmonics;
      const colorMode = frame.gaussianColor ?? harmonics?.colorMode ?? 'baked';
      if (!['baked', 'rgb', 'palette'].includes(colorMode) || colorMode !== 'baked' && !harmonics)
        throw new TypeError('Gaussian RGB and palette modes need live imported harmonics');
      const proxyPlan = gaussianProxyPlan(world, frame.gaussianPartProxies ?? world.gaussianPartProxies);
      const gaussianMaterial = colorMode !== 'baked' &&
        gaussianMaterialPlan(world, frame.gaussianMaterials ?? world.gaussianMaterials);
      const raw = colorMode === 'baked' ? frame.raw ?? false : colorMode === 'rgb';
      const { outline = true, dither = 0.35, scale = 1 } = frame;
      const look = world.look ?? {};
      const pbrDefinition = frame.pbrMaterials ?? look.pbrMaterials;
      if (pbrDefinition && !toolkit.KIT?.pbr) throw new Error('PBR materials need the Ink Studio KIT');
      const pbrRampOverrides = frame.surfaceLightRamps ?? look.surfaceLightRamps;
      const pbr = toolkit.KIT?.pbr?.prepare(pbrDefinition, frame.pbrLights, {
        palette:world.pal, paletteRoles:world.paletteRoles, surfaceLightRamps:pbrRampOverrides,
        specularSampling:frame.pbrSpecularSampling, paintShadowBands:frame.shadowBandPaint !== false
      });
      const boundaryView = plane === zero && !cover && !reflectionCapture;
      const liveBoundaryDefinition = boundaryView && (frame.livePixelBoundaries ?? look.livePixelBoundaries);
      const liveBoundaryAPI = toolkit.KIT?.livePixelBoundaries;
      if (liveBoundaryDefinition !== undefined && liveBoundaryDefinition !== false && !liveBoundaryAPI)
        throw new Error('Live pixel boundaries need the Ink Studio KIT');
      const liveBoundarySettings = liveBoundaryAPI?.prepare(liveBoundaryDefinition);
      const gaussianProfile = gaussianFilter.prepare(frame.gaussianFilter ?? look.gaussianFilter);
      const stable=!raw && stability?.prepare(look.stable,frame.stable,frame.ditherSpace);
      const orderMode = frame.splatOrder ?? look.splatOrder;
      const localOrder = surfaceOrderSettings(orderMode === 'selective' ? 'center' : orderMode, frame.surfaceOrder ?? look.surfaceOrder);
      let engraved=frame.engraving!==false&&!!world.engraving;
      let engravingUniforms;
      const clusterDefinition = raw || reflectionCapture ? false : frame.colorClusters ?? look.colorClusters;
      if (clusterDefinition !== undefined && clusterDefinition !== false && !clusterAPI)
        throw new Error('Color clusters need the Ink Studio KIT');
      const clusters = clusterAPI?.prepare(clusterDefinition);
      const boundaryDefinition = raw || reflectionCapture ? false : frame.boundaryCels ?? look.boundaryCels;
      if (boundaryDefinition !== undefined && boundaryDefinition !== false && !boundaryAPI)
        throw new Error('Boundary cels need the Ink Studio KIT');
      const boundaries = boundaryAPI?.prepare(boundaryDefinition, { scenePalette: world.pal, paletteRoles: world.paletteRoles });
      const correctionView = !reflectionCapture && plane === zero;
      const correctionDefinition = correctionView && (frame.shotCorrections ?? look.shotCorrections);
      const captureCorrection = correctionView && frame.captureShotCorrectionSurface === true;
      const correctionAPI = toolkit.KIT?.shotCorrections;
      if ((correctionDefinition !== undefined && correctionDefinition !== false || captureCorrection) && !correctionAPI)
        throw new Error('Shot corrections need the Ink Studio KIT');
      const corrections = correctionAPI?.prepare(correctionDefinition);
      const finishSceneArt = clusters || boundaries, finishShotArt = corrections || captureCorrection;
      const finishArt = finishSceneArt || finishShotArt;
      const coherenceHistory = plane === zero ? frame.colorClusterHistory ?? viewHistories.clusters : viewHistories.clusters;
      if (!reflectionCapture && !clusters?.coherent) coherenceHistory?.clear();
      const clock = contract?.clock(world, frame);
      const t = clock?.time ?? frame.time ?? world.lookSnapshot?.time ?? 0;
      const time = clock?.scene ?? (frame.stepAnimation !== false && look.pixel ? Math.floor(t * 10) / 10 : t);
      if (frame.strictContract) {
        if (!contract) throw new Error('Strict frame validation needs the Ink Studio KIT');
        contract.requirePortable(world, frame);
      }
      if (!raw && world.pal.length > scenePaletteSize && !warnedPalettes.has(world.pal)) {
        warnedPalettes.add(world.pal);
        console.warn('Ink Studio WebGL resolves the first ' + scenePaletteSize + ' of ' + world.pal.length +
          ' scene swatches. Use raw: true to draw baked colors without palette resolve.');
      }
      const canStage = contract?.hasStages(world, frame) && !contract.compatibility(world, frame).length;
      const qualityDefinition = frame.passQuality ?? look.passQuality;
      if (qualityDefinition && !qualityAPI) throw new Error('Pass quality needs the Ink Studio KIT');
      const priorityDefinition=frame.jointPriority ?? look.jointPriority;
      const priorityAPI=toolkit.KIT?.jointPriority;
      if(priorityDefinition && !priorityAPI)throw new Error('Joint priorities need the Ink Studio KIT');
      const priority=priorityAPI?.prepare(priorityDefinition);
      const priorityWindow=priority && priorityAPI.at(priority.windows,t);
      let quality=qualityAPI?.prepare(qualityDefinition);
      if(priorityWindow)quality=qualityAPI.prioritize(quality,priorityWindow.picture);
      const qualityKey=quality ? JSON.stringify([qualityDefinition,frame.silhouetteProtection ?? look.silhouetteProtection,
        priorityWindow?.picture]) : '';
      const cutawayDefinition = frame.cutaway ?? look.cutaway;
      if (cutawayDefinition && !cutawayAPI) throw new Error('Cutaways need the Ink Studio KIT');
      const cutaway = cutawayAPI?.prepare(cutawayDefinition);
      const section = cutaway && cutawayAPI.uniforms(cutaway,camera.setback);
      if (!reflectionCapture && plane === zero) {
        const signature = cutaway ? JSON.stringify(cutaway) : null;
        if (signature !== previousSection) historyValid = false;
        previousSection = signature;
      }
      const detailDefinition = frame.stableDetail ?? look.stableDetail;
      if (detailDefinition && !stableDetailAPI) throw new Error('Stable detail needs the Ink Studio KIT');
      const stableDetail = stableDetailAPI?.prepare(detailDefinition);
      const detailTexture = stableDetail?.texture ?? [0,0], detailGeometry = stableDetail?.geometry ?? [0,0];
      const detailRanges = [...detailTexture,...detailGeometry];
      if (!sameValues(previousDetailRanges,detailRanges)) historyValid = false;
      previousDetailRanges = detailRanges;
      const tolerance = transparencyAPI?.transparency(frame.stableTransparency ?? look.stableTransparency)?.depthTolerance ?? 0;
      const budget = raw ? null : detailAPI?.prepare(look.detailBudgets);
      const coverage = !raw && (frame.coverageAware ?? look.coverageAware) === true;
      const toneDefinition = raw || coverage ? undefined : frame.tone ?? look.tone;
      const frameTone = prepareTone(toneDefinition);
      const globalCleanup = !raw && (frame.clean ?? (!!look.clean && !frameTone?.preserveDots));
      const needsDetailFlags = budget?.affectsCleanup && (globalCleanup || budget.enablesCleanup || budget.protectsGaps);
      const constraintDefinition = frame.paletteConstraints ?? look.paletteConstraints;
      const wantsEmitters=frame.areaLights!==false&&world.areaLights?.length||Array.isArray(frame.areaLights)&&frame.areaLights.length||(frame.volumeLighting??world.volumeLighting)?.spots?.length;
      const wantsRelight = lighting && (frame.relighting ?? world.relighting);
      const wantsShot = frame.shotLighting ?? world.shotLighting;
      if (wantsShot && !shotLighting) throw new Error('Shot lighting needs the Ink Studio KIT');
      const filterDefinition = frame.coverageFilter ?? look.coverageFilter;
      if (filterDefinition !== lastFilterDefinition) {
        filterSettings = coverageFilter.prepare(filterDefinition);
        lastFilterDefinition = filterDefinition;
      }
      let group = frame.instances;
      const afterimageDefinition = frame.afterimages ?? look.afterimages;
      const trails = !reflectionCapture && !frame.order && afterimageDefinition !== undefined && afterimageDefinition !== false;
      if (trails && !afterimagesAPI) throw new Error('Afterimages need the Ink Studio KIT');
      const rules = trails ? afterimagesAPI.prepare(afterimageDefinition) : [];
      const normalCaptures = simpleOnly || rules.length ? null : captureWater(world, frame, camera, width, height, 0, quality);
      // Per-pixel material, palette, and detail rules index the expanded world.
      if (group && (pbr || materialControls.has(world) || canStage || quality?.rows || rules.length || clusters || liveBoundarySettings || frame[graphOwner] || frame.order || tolerance || wantsRelight || wantsShot || wantsEmitters || stable?.anchorSpaces?.size || world.engraving || world.splatFootprints || instancePlan(group).hasSplatFootprints || world.opticalSplats || instancePlan(group).hasOpticalSplats || !raw && (constraintDefinition || look.pixel && world.pix ||
          needsDetailFlags || budget?.fineOn || budget?.fineOff || filterSettings?.varied))) {
        expandInstances ??= createExpandedInstances();
        const expanded = expandInstances(world,group,frame.changedRows);
        world = expanded.world;
        frame = { ...frame, changedRows:expanded.changedRows };
        group = undefined;
      }
      const trailHistory = frame.afterimageHistory ?? viewHistories.afterimages;
      let trailStart = reflectedTrailStart;
      if (!reflectionCapture && (frame.resetAfterimages || !rules.length)) trailHistory?.clear();
      if (rules.length) {
        const rows = trailHistory.frame(world, t, rules, { source, layout: reconstruction?.layout ?? frame.instances, poseTime: time, bornAt: frame.bornAt,
          withdrawal: frame.sampleWithdrawal ?? world.sampleWithdrawal ?? 0 });
        if (rows.length) {
          trailStart = world.count;
          world = afterimagesAPI.append(world, rows);
        }
      }
      const captures = !simpleOnly && rules.length ?
        captureWater(world, { ...frame, instances: group }, camera, width, height, trailStart, quality) : normalCaptures;
      const portableStages = canStage && contract.stagePlan(world, frame, time, { skipMasks: frame.lookMasks === false,
        blueNoise: settings => toolkit.KIT.procedural.materials.blueNoise(settings),
        query: () => toolkit.KIT.world.query.fromScene(world, time) });
      const selection = prepareQualityRows(world, frame, quality, qualityKey, trailStart || world.count);
      const constraints = !raw && prepareConstraints(world, constraintDefinition, frame.lookMasks === false);
      const spatialMarks=!raw && !portableStages && !coverage && !!(
        stable?.anchorSpaces?.size || constraints?.palettes.some(row=>row.anchor && row.anchor.dither!=='screen'));
      const masked = Boolean(constraints?.masked);
      if (masked && !coverage) maskOwner ??= target(2);
      const celView = !reflectionCapture && plane === zero;
      const retainedDefinition = (frame.protectedPixels ?? look.protectedPixels);
      if(celView && retainedDefinition && !retainedAPI)
        throw new Error('Protected pixels need the Ink Studio KIT');
      const retained = celView && retainedDefinition && retainedAPI.prepare(
        look.protectedPixels, frame.protectedPixels, {scenePalette:world.pal,paletteRoles:world.paletteRoles});
      const highlightDefinition = celView && (frame.highlightCels ?? world.highlightCels);
      const foamDefinition = celView && !coverage && (frame.shoreFoam ?? world.shoreFoam);
      const celClip = cutaway && (at => cutawayAPI.margin(cutaway, ...at,
        -(camera.view[2]*at[0] + camera.view[6]*at[1] + camera.view[10]*at[2] + camera.view[14]) - camera.setback) >= 0);
      const projectedMarks = retained ? retainedAPI.project(retained,world,camera,{w:width,h:height,time,clip:celClip}) : null;
      const foamRows = foamDefinition && foamAPI?.frame(foamDefinition, world, camera,
        { w: width, h: height, time: t, clip: celClip });
      const highlightPlan = highlightDefinition && highlightAPI?.prepare(highlightDefinition, {
        scenePalette: world.pal, paletteRoles: world.paletteRoles
      });
      const finiteCurves = finiteCurveMode(world,frame);
      if (finiteCurves && !toolkit.KIT?.curveSplats?.renderKernel)
        throw new Error('Finite curves need the Ink Studio KIT');
      const hybridDepthCapture = typeof frame.hybridDepthFrame === 'function';
      const needsPresentationDepth = Boolean(projectedMarks?.coordinates.size || foamRows?.length || highlightPlan?.length || frame[graphOwner]?.depth || finishShotArt);
      const sourceWater = frame.water !== false && world.water?.count ? world.water : null;
      const pixel = toolkit.KIT?.pixel;
      const patterned = !raw && !!(look.pixel && world.pix && pixel?.gpu);
      const glass = raw ? null : world.glass;
      const filter = !raw && !coverage && !portableStages && !sourceWater && !glass &&
        !cover && plane === zero && !reflectionCapture && !proxyPlan && colorMode === 'baked' ? filterSettings : null;
      const historyView = plane === zero && !cover && !reflectionCapture && !raw && !portableStages && !coverage &&
        !masked && !sourceWater && !glass && !proxyPlan && colorMode === 'baked' && !group;
      // These combinations use distinct visibility/resolve contracts. Leave
      // their color and depth intact if a live boundary look is also authored.
      let liveBoundaryRegion, liveBoundaryPass;
      if (liveBoundarySettings) {
        if (coverage || masked || filter || sourceWater || glass || world.opticalSplats ||
            finiteCurves || proxyPlan || colorMode !== 'baked' || localOrder) {
          frame.livePixelBoundaryFrame?.({w:width,h:height,skipped:'incompatible-visibility'});
        } else {
          liveBoundaryRegion=liveBoundaryAPI.region(liveBoundarySettings,width,height);
          if (width*2>Math.min(maxSize,viewportLimit[0]) || height*2>Math.min(maxSize,viewportLimit[1]))
            throw new RangeError('Live pixel boundary samples exceed the WebGL device limit');
          liveBoundaries ??= createLivePixelBoundaries(gl,{api:liveBoundaryAPI,kit:toolkit.KIT,
            target,table,program,textureUnit,fullscreen,maxSize,resources});
          if (liveBoundaries.prepare(world,liveBoundarySettings)) liveBoundaryPass=liveBoundaries;
          else frame.livePixelBoundaryFrame?.({w:width,h:height,skipped:'no-matching-features'});
        }
      }
      if (plane===zero&&!reflectionCapture&&!liveBoundaryPass) liveBoundaries?.clear();
      const temporal = !pbr && !liveBoundaryPass && historyView && surfaceHistory.prepare(frame.surfaceHistory ?? look.surfaceHistory);
      const dampingAllowed = !pbr && !liveBoundaryPass && historyView && !filter && !temporal && !spatialMarks && !frameTone && !constraints &&
        !!look.pixel && frame.damp !== false;
      const packedDepth = needsPresentationDepth || hybridDepthCapture || finiteCurves || !!temporal || dampingAllowed || !!liveBoundaryPass;
      let owner = packedDepth ? (graphOwnership ??= target(3)) : masked && !coverage ? maskOwner : plainOwner;
      const ao = world.ao ? world.ao.k ?? 0.5 : 0;
      const paletteFrame = toolkit.PixelLook.framePalette?.(world, frame, { grading: !raw && !portableStages, tone: !raw && !coverage }) ??
        { grade: frame.ignoreLookGrade ? undefined : world.anim?.grade };
      const needsGrade = !raw && !portableStages && !reflectionCapture && !frameTone && !!paletteFrame.grade;
      if (simpleOnly) {
        const unsupported = Object.entries({
          'per-pixel materials': patterned, 'glass beams': glass, 'ambient occlusion': !raw && !portableStages && ao > 0,
          'tone mapping': frameTone, 'palette grading': needsGrade, 'palette constraints': constraints, 'spatial dither':spatialMarks, 'coverage filtering':filter,
          'coverage-aware resolve': coverage, water: sourceWater
        }).filter(([, active]) => active).map(([feature]) => feature);
        if (unsupported.length)
          throw new Error(`Ink Studio WebGL simple resolve does not support ${unsupported.join(', ')}. Disable these features or use resolve: 'auto'.`);
      }
      const drivenOptics = Array.isArray(look.optics) && look.optics.some(rule => rule.driver);
      const inputTime = drivenOptics ? frame.time ?? world.lookSnapshot?.time ?? 0 : undefined;
      if (!raw && sourceWater && (look.optics !== opticsDefinition || world !== opticsWorld || inputTime !== opticsTime)) {
        opticsDefinition = look.optics;
        opticsWorld = world;
        opticsTime = inputTime;
        opticsPlan = toolkit.RenderStages?.optics(look.optics, { palette: world.pal, roles: world.paletteRoles,
          time: inputTime, query: () => world.lookSnapshot?.query ?? world.surfaceEffects?.query ??
            toolkit.KIT?.world?.query.fromScene(world, inputTime),
          blueNoise: toolkit.KIT?.procedural?.materials?.blueNoise });
      }
      const optical = raw ? null : opticsPlan;
      if (sourceWater && (sourceWater !== waterLookSource || constraints !== waterLookPlan || optical !== waterOpticalPlan)) {
        waterLookSource = sourceWater;
        waterLookPlan = constraints;
        waterOpticalPlan = optical;
        mappedWater = (constraints || optical) && sourceWater?.mapColors ?
          sourceWater.mapColors((medium, mediumKind) => optical?.transform({ kind: 'medium', medium, mediumKind }) ??
            toolkit.PixelLook.mediumTransform(constraints, { medium, mediumKind })) : sourceWater;
      }
      const waterScene = sourceWater ? mappedWater : null;
      if (waterScene) {
        wetGeometry ??= target(3);
        wetContours ??= target(3);
      }
      let layeredGlass = false;
      if (coverage && (world.refract ?? 6)>0 &&
          maxBuffers >= (waterScene ? 5 : 4)) {
        if (world!==glassWorld || world.data!==glassData || frame.changedRows?.length!==0) {
          const indices=[];
          for(let i=0;i<world.count;i++)if(Math.round(world.data[i*16+11])&256)indices.push(i);
          if(!sameValues(glassIndices,indices))glassIndices=Int32Array.from(indices);
          glassWorld=world;glassData=world.data;
        }
        layeredGlass=Boolean(glassIndices?.length);
      }
      if(layeredGlass){
        glassBoundary ??= target(1,{depth:true});
        glassBoundary.resize(width,height);
        if(waterScene)wetGlassGeometry ??= target(5);
        else dryGlassGeometry ??= target(4);
      }
      let geometry = waterScene ? (layeredGlass ? wetGlassGeometry : wetGeometry) :
        (layeredGlass ? dryGlassGeometry : dryGeometry), contours = waterScene ? wetContours : dryContours;
      const worldIdentity = viewPoses?.source(world) ?? world;
      const isNew = worldIdentity !== lastWorld || world.data !== lastData || world.count !== lastCount || group !== lastInstances;
      const materialStyles = !!(world.materialPixelStyles || group && instancePlan(group).hasMaterialPixelStyles);
      const coverageSetting = frame.splatCoverage ?? look.splatCoverage ?? 'legacy';
      const splatCoverage = coverageRules?.coverageMode?.(coverageSetting) ?? ['legacy', 'continuous', 'soft'].indexOf(coverageSetting);
      if (splatCoverage < 0) throw new RangeError('Splat coverage must be legacy, continuous or soft');
      const voteDefinition = frame.coverageVote ?? look.coverageVote;
      let vote;
      if (voteDefinition && !filter && !raw && !reflectionCapture && !portableStages && !frameTone && !patterned &&
          !constraints && !spatialMarks && !waterScene && !glass && !group && !proxyPlan && plane === zero && !cover) {
        coverageVotePass ??= createCoverageVote(gl, {
          samples:target(2), source:target(), resources, table, program, textureUnit, fullscreen,
          maxSize, paletteSize:scenePaletteSize
        });
        if (coverageVotePass.prepare(world, voteDefinition, Math.min(world.pal.length, scenePaletteSize))) {
          vote = coverageVotePass;
          if (width * 2 > Math.min(maxSize, viewportLimit[0]) || height * 2 > Math.min(maxSize, viewportLimit[1]))
            throw new RangeError('Coverage vote samples exceed the device viewport or texture limit');
          vote.samples.resize(width * 2, height * 2);
          vote.source.resize(width, height);
        }
      }
      const colored = !temporal && !filter && !spatialMarks && !raw && !finiteCurves && !portableStages && !frameTone && !patterned && !coverage && (frame.colored ?? true);
      const detail = prepareDetail(world, budget);
      const scopedCleanup = !!needsDetailFlags && !!detail?.flags;
      const protectGaps = scopedCleanup && budget.protectsGaps;
      const protectResolve = protectGaps && !colored;
      const cleaning = globalCleanup || scopedCleanup && budget.enablesCleanup;
      const damping = dampingAllowed && !colored;
      const primary = plane === zero && !reflectionCapture;
      if (primary && !temporal) lightingHistory?.clear();
      if (primary && !damping) paletteHistory?.clear();
      const historyResources={target,program,textureUnit,fullscreen};
      if(temporal)lightingHistory??=createSurfaceHistory(gl,historyResources);
      if(damping)paletteHistory??=createSurfaceHistory(gl,historyResources);
      const History = temporal ? lightingHistory : damping ? paletteHistory : null;
      const historyOwner=History?.labels(world.count);
      const historyFrame={w:width,h:height,time:t,camera,layout:world.count,
        source:frame.surfaceHistorySource??frame.renderHistoryKey??source};
      const historyCut=frame.resetRenderHistory===true || frame.resetSurfaceHistory===true;
      const samples = colored ? 2 : 1, rw = width * samples, rh = height * samples;
      if (rw > maxSize || rh > maxSize ||
          rw > viewportLimit[0] || rh > viewportLimit[1])
        throw new RangeError('Ink Studio frame exceeds the WebGL viewport or texture size limit');
      if (!output) {
        if (gl.canvas.width !== width) gl.canvas.width = width;
        if (gl.canvas.height !== height) gl.canvas.height = height;
      }
      const resized = width !== finished.width || height !== finished.height;
      if (isNew || qualityKey !== lastQualityKey || resized || output !== lastOutput || detail !== lastDetail || !damping || waterScene !== lastWater ||
          (waterScene && frame.time !== lastWaterTime)) historyValid = false;
      if (gaussianProfile?.compensated) {
        halfColor ??= Boolean(gl.getExtension('EXT_color_buffer_float') || gl.getExtension('EXT_color_buffer_half_float'));
        if (halfColor) {
          function filteredTarget(source) {
            if (!filteredTargets.has(source)) filteredTargets.set(source, target(source.textures.length, { halfColor: true }));
            return filteredTargets.get(source);
          }
          geometry = filteredTarget(geometry);
          if (colored) contours = filteredTarget(contours);
        }
      }
      geometry.resize(rw, rh);
      contours.resize(colored ? width : 1, colored ? height : 1);
      if (liveBoundaryPass) {
        if (!colored) liveBoundaryPass.sampleColor.resize(width*2,height*2);
        liveBoundaryPass.sampleOwner.resize(width*2,height*2);
      }
      const ownsPixels = !!liveBoundaryPass || needsPresentationDepth || !!historyOwner || !!clusters || !!boundaries || !!frame[graphOwner] || patterned || !!constraints && !coverage || scopedCleanup;
      const drawOwnership = ownsPixels || hybridDepthCapture || finiteCurves;
      owner.resize(drawOwnership ? width : 1, drawOwnership ? height : 1);
      const captureNormals = harmonics && frame[gaussianNormalFrame];
      if (captureNormals) {
        normalOwner ??= target();
        normalOwner.resize(width, height);
      }
      if (scopedCleanup) detailTable ??= table(512, 7, maxSize);
      if (materialStyles) materialStyleTable ??= table(512, 16, maxSize);
      key.resize(patterned ? width : 1, patterned ? height : 1);
      if (cover && patterned && frame.edges !== false) portalSpriteLayer ??= target(2, { depth: true });
      if (cleaning) cleaned.resize(width, height);
      const materialPalette = gaussianMaterial && colorMode === 'palette';
      if (materialPalette) {
        gaussianMaterialPalette ??= createGaussianMaterialPalette({ target, program, textureUnit, fullscreen });
        gaussianMaterialPalette.resize(width, height);
        historyValid = false;
      }
      const splatPaint = !raw && splatPaintStyle(world, frame.splatPaint ?? world.splatPaint);
      if (splatPaint) {
        splatPaintPass ??= createSplatPaintPass({ table, target, maxSize, program, textureUnit, fullscreen });
        splatPaintPass.resize(width, height);
        historyValid = false;
      }
      if (damping) finished.resize(width,height);

      const paint = world.paintCorrections;
      let geometryData = world.data, paintActive = false;
      if (paint && frame.paintCorrections !== false) {
        if (paintData?.length !== world.count * 16) paintData = new Float32Array(world.count * 16);
        paintData.set(world.data.subarray(0, world.count * 16));
        const { view: m, ortho, eye } = camera;
        paintActive = paint.paint(paintData, world.pal,
          { direction: [m[2], m[6], m[10]], ...(ortho ? {} : { eye }) });
        geometryData = paintData;
      }
      const paintView = paintActive ? camera.view : undefined;
      if (portableStages) geometryData = contract.materialData(portableStages, world, geometryData);
      const diffuseRamps = colorMode === 'baked' && applyDiffuseRamps(toolkit, world, frame, camera, t, time, geometryData, patterned);
      if (diffuseRamps) geometryData = diffuseRamps.data;
      if (diffuseRamps || lastDiffuseRamps) historyValid = false;
      if ((paintView || lastPaintView) && (!paintView || !lastPaintView || !sameValues(paintView, lastPaintView)))
        historyValid = false;
      lastPaintView = paintView;
      const paintOnPatterns = patterned && (paintActive || !!trailStart || !!world.opticalSplats);
      if (paintOnPatterns) {
        paintOverlay ??= target(2);
        paintOverlay.resize(width, height);
      }
      const clip = plane === zero ? zero : planeInView(plane, camera);
      const direction = clock?.direction ?? motionAPI?.prepare(look.artDirection,frame.artDirection);
      const motion = clock ?? motionAPI?.motion(direction,t,frame.skyTime ?? t,time);
      const rainClocks = motion ? [motion.falling, motion.rain] : [time,time];
      const hasCadence = rainClocks.some(value => value !== time) ||
        ['foliage','cloth','rain','water','emissive'].some(key => direction?.cadence[key] !== undefined);
      const windResponse = direction?.windResponse;
      const useWind = climateAPI && airflow && (hasCadence || windResponse?.water || windResponse?.emissive) &&
        (frame.windField ?? world.windField ?? world.weatherBinding);
      let flex = () => [0,0], rainWind = [0,0];
      if (useWind) {
        const sampled = climateAPI.timeline(world,frame,t,direction,toolkit.KIT?.weather?.tracks);
        const weather = climateAPI.opticalWeather(world,frame,sampled,frame.weatherFronts !== false);
        const air = climateAPI.wind(world,frame,{ ...sampled, weather });
        const velocity = (channel) => air.windAt?.(direction?.cadence[channel] === undefined ? sampled.skyTime :
          frame.skyTime ?? motion[channel]) ?? air.wind.velocity;
        flex = channel => airflow.frame(velocity(channel)).flex;
        if (direction?.cadence.rain !== undefined) rainWind = velocity('rain');
      }
      const water = waterUniforms(waterScene, camera, motion?.water ?? time, plane, captures, width, height,
        frame.reflectionDetail!==false,frame);
      if (water) water.u_waterWindGain = (windResponse?.water ?? 0) * Math.hypot(...flex('water'));
      const env = environment.prepare(world,frame,camera,t,width,height,reflectionCapture,plane);
      if(water?.u_waterReflectionProxyBase){
        const base=rgb(world.pal[0]??0), [ground,horizon,sky]=env?.reflectionColors??[base,base,base];
        Object.assign(water,{u_waterProxyGround:ground,u_waterProxyHorizon:horizon,u_waterProxySky:sky});
      }
      if (env || lastEnvironment) historyValid = false;
      lastEnvironment = !!env;
      const wet = water && wetVariant(water,!!env?.media);
      engraved=frame.engraving!==false&&!!world.engraving;
      if(engraved)historyValid=false;
      const footprints = (frame.splatFootprints ?? look.splatFootprints) !== false && !!world.splatFootprints?.styles.length;
      let footprintUniforms;
      if (footprints) {
        if (!footprintAPI) throw new Error('Splat footprints need the Ink Studio KIT');
        footprintBinding ??= createFootprintBinding(gl,footprintAPI,resources,targets,maxSize);
        footprintUniforms=footprintBinding(world);
      }
      const opticalMaterials = !!world.opticalSplats;
      if (opticalMaterials) {
        const K = toolkit.KIT;
        opticalRows = K.optical.frame(world, camera, K.splatFrames.create(world, time, K.M), null, opticalRows);
        opticalData ??= table(1536, opticalUnit, maxSize);
        opticalData.upload(opticalRows);
      }
      if (opticalMaterials || lastOptical) historyValid = false;
      lastOptical = opticalMaterials;
      const K = toolkit.KIT;
      const sample = (wantsRelight || wantsShot) && K.splatFrames.create(world, time, K.M);
      const pbrTargets = pbr && K.pbr.splatBindings(world, pbr);
      let lightRows;
      if (wantsRelight) {
        const timeline = K.environmentFrame.timeline(world, frame, t,
          K.artDirection.prepare(look.artDirection, frame.artDirection), K.weather?.tracks);
        lightRows = lighting.rows(world, lighting.frame(world, frame, timeline), sample, null, relightRows);
        if (lightRows) relightRows = lightRows;
      }
      if (wantsShot) {
        const rig = shotLighting.frame(world.shotLighting, frame.shotLighting, t);
        const joined = shotLighting.rows(world, rig, sample, lightRows, shotRows);
        if (joined && joined !== lightRows) shotRows = joined;
        lightRows = joined;
      }
      let relighting = !!lightRows;
      if (relighting) {
        relightData ??= table(512, relightUnit, maxSize);
        relightData.upload(lightRows);
      }
      const brickAPI=K?.procedural?.brickMeshes;
      const brickFrames=world.brickMeshes ? brickAPI.frame(world,frame.brickMeshes) : [];
      const brickCasters=brickFrames.flatMap(({assets})=>assets.shadow ? [assets.shadow] : []);
      const emitterFrame=emitterLighting.prepare(world,frame,t,reflectionCapture,camera,brickCasters);
      const highlightRows = highlightPlan?.length && highlightAPI.frame(highlightPlan, world, camera, {
        w: width, h: height, time: t, lights: emitterFrame?.areas ?? [], cutoutShadows: frame.cutoutShadows, shadowShells: frame.shadowShells,
        spriteSurfaces: frame.spriteSurfaces === false ? false : world.spriteSurfaces,
        spriteAtlases: frame.spriteAtlases === true ? world.spriteAtlases : frame.spriteAtlases,
        spriteAtlasMode: frame.spriteAtlasMode,
        clip: celClip
      });
      relighting=emitterLighting.compose(emitterFrame,world,time,lightRows,relighting);
      emitterLighting.fog(emitterFrame,world,frame);
      if(emitterFrame){historyValid=false;if(env){env.emitters=emitterFrame;env.uniforms={...env.uniforms,...emitterFrame.uniforms};}}
      if(env)env.lit=!!(env.lit&&emitterFrame);
      if (relighting || lastRelighting) historyValid = false;
      lastRelighting = relighting;
      const fineTiles = detail?.fine ?? world.fine;
      const hasFine = !raw && !!fineTiles;
      const pal = palette(world.pal);
      const palCount = Math.min(world.pal.length, scenePaletteSize);
      const fog = world.fog ?? {};
      // An editor may provide a row diff against the exact bake currently on the GPU.
      const incremental = frame.previousWorld && frame.previousWorld === lastWorld &&
        lastWorld.data === lastData && world.count === lastCount && !group && !lastInstances &&
        !paint && !lastWorld.paintCorrections && !world.anim && !lastWorld.anim;
      const dirty = !isNew || incremental ? frame.changedRows : undefined;
      const geometryDirty = portableStages || lastStaged || diffuseRamps || lastDiffuseRamps ? undefined : dirty && paint ?
        [...new Set([...dirty, ...paint.dirtyRows.filter(row => row * 512 < world.count)])] : dirty;
      let plan;
      if (group) {
        instanceDraw ??= createInstanceDraw(table,data,maxSize);
        plan = instanceDraw.bind(group, world, geometryDirty, lastInstances !== group, geometryData);
      } else if(engraved){
        if(!engravingAPI)throw new Error('Engraving needs the Ink Studio KIT');
        engravingBinding??=createEngravingBinding(engravingAPI);
        const attached=engravingBinding(world,geometryData,geometryDirty,frame.engravingLight??[-.45,.8,.55]);
        data.upload(attached.data,attached.data.length,attached.changedRows);engravingUniforms=attached.uniforms;
      }else data.upload(geometryData, world.count * 16, geometryDirty);
      let facing;
      const disabledViewColors = Number(frame.facingColors === false) | (Number(frame.directionalColors === false) << 1);
      if (!raw && disabledViewColors !== 3 && facingAPI && world.surfaceData) {
        if (!facingPlans.has(world)) facingPlans.set(world, facingAPI.table(world,
          (settings, ramp) => facingSurface.prepareFacing(settings, world.pal, world.paletteRoles, ramp),
          settings => facingSurface.prepareDirection(settings, world.pal, world.paletteRoles)));
        facing = facingPlans.get(world);
        if (facing) {
          facingTable ??= table(facingAPI.width, facingUnit, maxSize);
          const posed = facingAPI.pose(facing, world, time, (source, t) =>
            toolkit.KIT.splatFrames.create(source, t, toolkit.KIT.M));
          if (facing !== uploadedFacing || posed) facingTable.upload(facing.data);
          else facingTable.bind();
          uploadedFacing = facing;
        }
      }
      if (materialStyles) bindMaterialStyles(world, group, plan, dirty);
      if (hasFine && (isNew || fineValues !== fineTiles)) {
        const fineDirty = incremental && !detail && fineValues === lastWorld.fine ? frame.changedFineRows : undefined;
        fine.upload(fineTiles, world.count * 12, fineDirty);
        fineValues = fineTiles;
      } else fine.bind();
      uploadBones(world.anim?.bones);
      relighting=materialControls.compose(world,frame,camera,time,relighting);
      if(materialControls.has(world))historyValid=false;
      const back = [camera.view[2],camera.view[6],camera.view[10]];
      if (source.spriteReconstruction && !spriteLOD) throw new Error('Sprite reconstruction needs the Ink Studio KIT');
      const spritePlan = source.spriteReconstruction && spriteLOD.resolve(source, {
        eye:camera.eye,back,ortho:camera.ortho,focal:camera.projection[5]*height*.5,
        setback:camera.setback,time:t,mode:frame.spriteReconstruction,impostorBlend:frame.impostorBlend,protectedLOD:frame.protectedLOD
      });
      const spriteMask = spritePlan?.mask, spriteRows = spritePlan?.rows;
      if (spriteMask) {
        spriteLodTable ??= table(128,spriteLodUnit,maxSize);
        if (uploadedSpriteMask !== spriteMask) spriteLodTable.upload(Float32Array.from(spriteMask.codes));
        else spriteLodTable.bind();
        if (spriteMask !== uploadedSpriteMask) historyValid=false;
      } else if (uploadedSpriteMask) historyValid=false;
      uploadedSpriteMask=spriteMask;
      if(source.spriteReconstruction&&!reflectionCapture&&plane===zero)
        frame.spriteReconstructionFrame?.(spritePlan??{geometryRows:source.count,states:[]});
      const withdrawal = sampleCoverage?.amount(frame.sampleWithdrawal ?? world.sampleWithdrawal ?? 0) ?? 0;
      const reliefMeshes=world.reliefLOD!==undefined ? toolkit.KIT.procedural.lod.relief.frame(world,frame,{
        view:camera.view,width,height,focal:camera.projection[5]*height*.5,ortho:camera.ortho,
        near:camera.clipNear,far:camera.clipFar,setback:camera.setback
      },!reflectionCapture&&plane===zero?frame.reliefLODFrame:undefined) : [];
      if(!reflectionCapture&&plane===zero){
        if(reliefMeshes.length!==lastReliefMeshes.length || reliefMeshes.some((mesh,index)=>mesh!==lastReliefMeshes[index]))
          historyValid=false;
        lastReliefMeshes=reliefMeshes;
      }
      const meshLayers = frame.meshes===false || (world.meshes===undefined&&!reliefMeshes.length) ? [] :
        toolkit.KIT.sceneMeshes.layers(world,frame.meshes,reliefMeshes,frame.scanPaint)
          .filter(atlas=>!spriteLOD?.hidden(spriteMask?.meshCodes.get(atlas.sceneMeshId)));
      const brickLayers=brickFrames.length ? brickAPI.layers(brickFrames) : [];
      const spriteSurfaces = frame.spriteSurfaces === false ? null : world.spriteSurfaces;
      const decals=frame.surfaceDecals !== false && world.surfaceDecals?.sample(t,frame).layers;
      const atlases=[spriteSurfaces].flat().concat(meshLayers,brickLayers,decals || []);
      const simulationLayers=frame.simulationMaterialLayers===false?undefined:toolkit.KIT?.sceneMeshes?.simulationFrame(atlases,{
        time:t,query:()=>world.lookSnapshot?.query??world.surfaceEffects?.query??toolkit.KIT.world.query.fromScene(world,t),
        report:!reflectionCapture&&plane===zero?frame.simulationMaterialLayersFrame:undefined
      });
      if(!reflectionCapture&&plane===zero){
        const changes=simulationLayers?.size?JSON.stringify([...simulationLayers.values()].map(row=>row.amounts)):undefined;
        if(changes!==lastSimulationMaterials)historyValid=false;
        lastSimulationMaterials=changes;
      }
      const pbrFrames=pbr && (sample || K.splatFrames.create(world,time,K.M));
      if(pbr)pbr.shadows=K.pbr.shadows.prepare(pbr,world,pbrFrames,{
        layers:atlases,enabled:frame.pbrShadows!==false,curveMode:finiteCurves?'curves':'splats',shadowArt:frame.shadowArt,
        footprint:2*(camera.distance-(camera.setback??0))*camera.tangent/height,
        surfaceOptions:{time:t,interfaceTime:frame.interfaceTime,enabled:frame.cutoutShadows!==false,shadowShells:frame.shadowShells,
          surfaces:spriteSurfaces,sprites:frame.spriteAtlases===true?world.spriteAtlases:frame.spriteAtlases??world.spriteAtlases,
          atlasMode:frame.spriteAtlasMode}
      });
      const pbrRamps = pbr && K.pbr.rampTable(pbr);
      const pbrUseAlbedo = !portableStages && !paintActive;
      const pbrSource = pbr && [world, world.data, world.count, world.pix, world.renderMetadata, world.surfaceData,
        world.gaussianHarmonics, world.splatRoughness, world.materialAlbedo, pbrDefinition, time, pbrUseAlbedo,
        world.pal,world.paletteRoles,pbrRampOverrides,frame.shadowArt,world.look?.shadowArt,!!pbr.shadows,
        world.rest,world.curveRest,pbr.hasBands];
      const pbrSameRows = pbr && frame.changedRows?.length === 0 && !world.anim?.bones && !world.anim?.rig &&
        !world.reconstructionPose && pbrRowSource && pbrSource.every((value,i)=>value===pbrRowSource[i]) &&
        sameValues(pbrRamps.rows,uploadedPbrRamps);
      if (pbr && !pbrSameRows) pbrRest=K.pbr.restTable(world,pbr,pbrFrames,pbrTargets);
      const pbrRestOffset = pbr && world.count + pbrRamps.rows.length/4;
      const pbrShadowOffset = pbr && pbrRestOffset+(pbrRest.offsets?.length??0)/4+pbrRest.transforms.length/4;
      const pbrShadowData = pbr && K.pbr.shadows.pack(pbr.shadows,pbrShadowOffset);
      const pbrLength = pbr && (pbrShadowOffset*4+pbrShadowData.rows.length);
      // The shared shaders read ramps, rest transforms and shadow maps after the vertex rows.
      const pbrUniforms = pbr && {
        ...K.pbr.uniforms(pbr,pbrRamps),...pbrShadowData.uniforms,u_pbrRows:pbrUnit,u_pbrRampOffset:world.count,
        u_pbrRestOffset:pbrRest.offsets ? pbrRestOffset : -1
      };
      const pbrRetained = pbrSameRows && pbrRows?.length===pbrLength;
      if (pbr && !pbrRetained) {
        if (!pbrRows || pbrRows.length !== pbrLength) pbrRows = new Float32Array(pbrLength);
        for (let i = 0; i < world.count; i++) {
          const id = pbrTargets(i), at = i * 4;
          const receive=pbr.shadows?.receives?.(i)!==false;
          pbrRows[at + 3] = id ? (id+.5*(world.splatRoughness?.[i]??.45))*(receive?1:-1) : 0;
          if (id) {
            const normal = K.pbr.splatNormal(world, i, pbrFrames(i), K.M);
            pbrRows.set(K.pbr.encodeNormal(normal), at);
            pbrRows[at + 2] = pbrUseAlbedo ? world.materialAlbedo?.[i] ?? 0 : 0;
          }
        }
        pbrRows.set(pbrRamps.rows,world.count*4);
        if (pbrRest.offsets) {
          pbrRows.set(pbrRest.offsets,pbrRestOffset*4);
          pbrRows.set(pbrRest.transforms,pbrRestOffset*4+pbrRest.offsets.length);
        }
        uploadedPbrRamps = pbrRamps.rows;
        pbrRowSource = pbrSource;
      }
      if(pbr && (!pbrRetained || pbr.shadows)){
        pbrRows.set(pbrShadowData.rows,pbrShadowOffset*4);
        pbrTable ??= table(512,pbrUnit,maxSize);
        pbrTable.upload(pbrRows);
      }
      let marks, engraving, markOutput, filterBuffers;
      if (filter) {
        if (width*2 > Math.min(maxSize,viewportLimit[0]) || height*2 > Math.min(maxSize,viewportLimit[1]))
          throw new RangeError('Coverage filtering exceeds the device viewport or texture limit');
        coverageResolve ??= createCoverageFilter({target,table,maxSize,program,textureUnit,fullscreen});
        filterBuffers = coverageResolve.prepare(world,filter,width,height,finiteCurves);
      }
      if(spatialMarks){
        if(!stability)throw new Error('Spatial dither needs the Ink Studio KIT');
        restBinding??=createRestMarks(stability,table,maxSize);
        marks=restBinding(world,stable||undefined,frame.changedRows,camera.view);
        restTarget??=target(2,{floating:true});
        restTarget.resize(width,height);
      }
      if(frame.engraving!==false && (finiteCurves && world.engraving || atlases.some(atlas=>
          atlas?.surfaceCharts?.some(chart=>chart.engraving || atlas.sceneMeshId && world.look?.engraving)))){
        surfaceEngraving??=createSurfaceEngraving(engravingAPI,table,maxSize,toolkit.KIT?.curveSplats?.renderKernel);
        engraving=surfaceEngraving(world,atlases,frame,finiteCurves);
      }
      if(marks || engraving || filter){
        bones.bind();
        data.bind();
      }
      const curveSettings = finiteCurves && {
        pbrTargets,pbrUseAlbedo,pbrReceives:pbr?.shadows?.receives,
        time,restMarks:marks,engraving,filter, drawOrder:frame.order, planes:[clip,cover?.plane],
        geometryCode:spriteMask && (i=>spriteMask.codes[i]),
        curveSubdivision:cutaway ? false : frame.curveSubdivision ?? look.curveSubdivision,
        strandLod:(frame.curveLod ?? look.curveLod) !== false && !cutaway && !cover && !reflectionCapture && !clip.slice(0,3).some(Boolean),
        protectedLOD:frame.protectedLOD,strandLODFrame:!reflectionCapture&&plane===zero ? frame.strandLODFrame : undefined,
        strandForbidden:opticalMaterials ? i=>opticalRows[i*toolkit.KIT.optical.stride+3] : undefined,
        keep:i=>(!selection || selection.keep[i]) && !spriteLOD?.hidden(spriteMask?.codes[i]),
        visible:sampleCoverage && (flags=>sampleCoverage.visible(flags,withdrawal))
      };
      const curvePlan = curveSettings && prepareFiniteCurves(toolkit.KIT,world,geometryData,camera,width,height,curveSettings);
      orderFor(world, frame, camera, time, isNew, plan, width, height, scale, tolerance, histories, source, rainClocks, rainWind,
        selection, spriteMask, curvePlan?.rows);
      if (patterned) {
        const pixelRows=()=>{
          const source=diffuseRamps?.pixels ?? world.pix;
          if (!pbr) return source;
          const rows=source.slice();
          for (let i=0;i<world.count;i++) if(Math.abs(pbrRows[i*4+3])>=1)rows[i*8]=0;
          return rows;
        };
        const changedPbrPatterns = !!pbr !== materials?.pbr || pbr && !pbrRetained;
        if (isNew || materials?.world !== world || materials.values !== world.pix ||
            materials.ramps !== world.ramps || materials.rampOff !== world.rampOff || materials.patterns !== world.pats || changedPbrPatterns) {
          const patternDirty = !pbr && !changedPbrPatterns && !diffuseRamps && !lastDiffuseRamps && incremental && materials?.world === lastWorld &&
            materials.values === lastWorld.pix ? frame.changedPatternRows : undefined;
          const rows=pixelRows();
          pattern.upload(rows, world.count * 8, patternDirty);
          const gpu = materialData(pixel, world);
          updateSprites(pbr ? {...world,pix:rows} : world, pixel);
          materials = { world, values: world.pix, ramps: world.ramps, rampOff: world.rampOff, patterns: world.pats, pbr:!!pbr, gpu };
        } else if (diffuseRamps || lastDiffuseRamps || world.anim?.dynamicPatternNormals) {
          const patternDirty = !pbr && !diffuseRamps && !lastDiffuseRamps && dirty ? [...new Set(dirty.map(row => Math.floor(row / 2)))] : undefined;
          pattern.upload(pixelRows(), world.count * 8, patternDirty);
        } else pattern.bind();
      }

      const { colorGrade } = paletteFrame;
      let grade = null;
      if (needsGrade) {
        grade = pal.slice();
        for (let i = 0, n = Math.min(palCount * 3, paletteFrame.grade.length); i < n; i++)
          grade[i] = paletteFrame.grade[i];
      }
      if (masked) bindMaskSites(world,dirty);
      if (constraints?.textured) textureUnit(14,maskTexture);
      if (constraints && (coverage || masked)) {
        constraintSlots.bind(9);
        textureUnit(8,constraintRows);
      }
      gl.disable(gl.SCISSOR_TEST);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.CULL_FACE);
      gl.disable(gl.STENCIL_TEST);
      gl.colorMask(true, true, true, true);
      gl.depthMask(true);
      gl.blendEquation(gl.FUNC_ADD);
      if (proxyPlan) gaussianProxies ??= createGaussianProxyRenderer(gl, resources, table, maxSize);
      const gaussianUniforms = gaussianRows.bind(harmonics, colorMode !== 'baked');
      if (gaussianMaterial) gaussianMaterialBinding ??= createGaussianMaterialBinding(gl, table, maxSize);
      const materialUniforms = gaussianMaterial &&
        gaussianMaterialBinding.bind(gaussianMaterial, frame.gaussianLight ?? world.gaussianLight);
      const proxy = proxyPlan && { width, height, scale: 1, camera, clip, plane, gaussian: gaussianUniforms,
        degree: harmonics.degree, range: [...camera.depthRange, fog.start ?? 1, fog.max ?? 0],
        fog: rgb(world.pal[fog.c ?? 0] ?? 0) };
      const proxyUniforms = proxy && gaussianProxies.prepare(proxyPlan, proxy);
      if (proxy) historyValid = false;
      if (brickLayers.length!==lastBrickLayers.length || brickLayers.some((layer,index)=>layer!==lastBrickLayers[index]))
        historyValid=false;
      lastBrickLayers=brickLayers;
      if (brickPipelines.size && !reflectionCapture && plane===zero) {
        const retained=new Set(brickLayers.map(atlas=>atlas.sceneMeshId));
        for (const [id,draw] of brickPipelines) if (!retained.has(id)) {
          draw.dispose();
          brickPipelines.delete(id);
        }
      }
      const surfaces=meshLayers.length||brickLayers.length||decals?.length ? atlases : spriteSurfaces;
      const depthProxy = K?.sceneMeshes?.proxyPlan(world,frame.splatDepthProxies,frame.splatProxyTolerance,brickFrames.flatMap(row=>row.assets.depthProxies));
      if (depthProxy || lastDepthProxy) historyValid=false;
      lastDepthProxy=!!depthProxy;
      const preview = frame.spriteSurfaces !== false && frame.scalePreviews !== false && world.scalePreviews;
      let ordering, orderReport;
      if (orderMode === 'selective' && !reflectionCapture)
        orderReport = { mode:'center', reason:'selective-backend', splats:pipeline.count, passes:0 };
      if (localOrder && !reflectionCapture) {
        const { layers, maxSplats, maxPixels, region } = localOrder;
        const count = pipeline.count;
        let reason;
        if (plane !== zero || cover || group || waterScene || env || layeredGlass || ownsPixels || finiteCurves || materialPalette ||
            splatPaint || vote || marks || filter || portableStages || depthProxy || surfaces || preview || spriteMask || spriteRows?.length || emitterFrame?.display ||
            proxy || footprints && footprintBinding.hasPlanes || section) reason = 'composition';
        else if (count < 2) reason = 'splat-count';
        else if (count > maxSplats) reason = 'splat-budget';
        else if (rw * rh + (colored ? width * height : 0) > maxPixels) reason = 'pixel-budget';
        else {
          surfaceOrdering ??= createSurfaceOrder(gl, { target, table, maxSize, textureUnit, fullscreen, program });
          if (!surfaceOrdering.prepare(world)) reason = 'no-normals';
        }
        if (!reason) {
          ordering = { layers: Math.min(count, layers), overflow: count > layers, region };
          historyValid = false;
        }
        orderReport = { mode: reason ? 'center' : 'surface', reason, splats: count,
          layers: ordering?.layers ?? 0, passes: ordering ? (ordering.layers + Number(ordering.overflow)) * (colored ? 2 : 1) : 0 };
      }
      const shader = pipeline.programFor({
        pbr: !!pbr,
        pbrSpecularSampling:pbr?.specularSampling, pbrBands:pbr?.hasBands,
        water: !!wet, airPockets: wet?.airPockets, layered: wet?.layered, atlas: wet?.atlas, portal: plane !== zero || !!cover,
        gaussianDegree: gaussianUniforms ? harmonics.degree : undefined, gaussianFilter:!!gaussianProfile, splatPaint: !!splatPaint, gaussianProxy:!!proxy, depthMeshProxy:!!depthProxy, gaussianMaterials:!!gaussianMaterial, coverageVote:!!vote, coverageFilter:!!filterBuffers?.centerMatte, surfaceOrder:!!ordering,
        instances: !!plan, contributions:coverage && !!constraints, masked, textured:constraints?.textured, layeredGlass,
        engraving:engraved, marks:!!marks, footprints, materialStyles, facingColors:facing && facingAPI, environment:env?.media,emitterFog:!!env?.lit, relighting, optical:opticalMaterials, cutaway:!!section, spriteReconstruction:!!spriteMask
      });
      shader.bindFilterRows = filterBuffers?.centerMatte ? filterBuffers.bindRows : undefined;
      shader.footprints=footprints;
      shader.gaussianRows = !!gaussianUniforms;
      shader.gaussianMaterials = !!gaussianMaterial;
      shader.pbr = !!pbr;
      gl.useProgram(shader.p);
      const growth = world.grow ?? {};
      shader.set({
        ...(pbr&&{...pbrUniforms,u_pbrRowCount:world.count,u_pbrView:camera.view,u_pbrReach:camera.radius}),
        u_data: 0, u_bones: 4, u_fine: 7, u_filterRows:filterUnit,
        ...(gaussianUniforms && { ...gaussianUniforms, u_eye: camera.eye }),
        ...proxyUniforms,
        ...materialUniforms,
        ...(ordering && surfaceOrdering.uniforms),
        ...(facing && { u_facingRows: facingUnit, u_facingCount: facing.count, u_disabledViewColors: disabledViewColors, u_facingPattern: patterned }), ...(spriteMask && {u_spriteLod:spriteLodUnit,u_spriteLodCount:spriteMask.codes.length}), ...(plan && { u_instances: 15 }), u_fineOn: +hasFine, u_fpal: pal, u_view: camera.view,
        ...(materialStyles && { u_materialStyles: 16, u_materialPatterned: patterned, u_materialStylesOn: !raw }),
        u_proj: camera.projection, u_time: time, u_scale: scale, u_ortho: +camera.ortho,
        u_motionCadence: hasCadence, u_rainWind:rainWind,
        u_motionWind: !!useWind && (direction?.cadence.foliage !== undefined || direction?.cadence.cloth !== undefined),
        ...(hasCadence && { u_effectTimes:[motion.foliage,motion.cloth,rainClocks[0],motion.emissive],
          u_waterAndRainTime:[motion.water,rainClocks[1]] }),
        ...(useWind && { u_effectWind:[...flex('foliage'),...flex('cloth')] }),
        u_emissiveWind:(windResponse?.emissive ?? 0) * Math.hypot(...flex('emissive')),
        ...(constraints && (coverage || masked) && { u_constraintSlots:9,u_constraintRows:8,
          u_constraintDefault:constraints.defaultSlot,u_constraintMap:constraintMap,u_constraintMask:maskAt,u_constraintLab:constraintLab,u_constraintPairs:constraintPairs,
          u_lookSites:10,u_maskTextures:14,u_maskTime:t,u_lookProjection:camera.projection,u_lookRadius:camera.radius,u_lookOrtho:+camera.ortho,
          u_pal:pal,u_n:palCount,u_grade:+Boolean(grade),u_out:grade??pal,u_toneOn:0,u_lookGrid:[width,height] }),
        u_arrivalPlane: plane, u_splatPlane: clip, ...(layeredGlass&&{u_glassBoundary:13,u_glassTolerance:tolerance}),
        u_sampleWithdrawal: withdrawal,
        u_born: [frame.bornAt ?? 0, growth.rate ?? 1, growth.fade ?? 0.5, growth.loop ?? 0],
        u_detailTexture: detailTexture, u_detailGeometry: detailGeometry,
        ...(gaussianProfile && { u_gaussianFilter: [
          gaussianProfile.variance, +gaussianProfile.compensated + 2*gaussianProfile.ray, +!!world.gaussianCloud, camera.clipFar
        ] }),
        u_preserveSceneY: !!world.gltfParts, u_crisp: +!raw, u_splatCoverage: splatCoverage, u_dilate: raw ? 0.3 : 0.15, u_near: camera.clipNear,
        u_paintMarker: paint?.marker ?? 0, u_afterimageStart: trailStart,
        ...footprintUniforms,...engravingUniforms,...marks?.uniforms,
        u_screenPrintOn:frame.screenPrint!==false,
        u_brushworkOn:frame.brushwork!==false,
        ...(opticalMaterials && { u_optical: opticalUnit }),
        ...(relighting && { u_relight: relightUnit }),
        u_fog: rgb(world.pal[fog.c ?? 0] ?? 0),
        u_fogp: [fog.start ?? 1, fog.max ?? 0, ...camera.depthRange], ...water, ...env?.uniforms, ...section
      });
      const captureSplatCoverage = world.gaussianHarmonics && frame[gaussianCoverageFrame];
      const background = [...rgb(world.pal[0]), captureSplatCoverage ? 0 : 1], tangent = camera.tangent;
      const previewAPI = toolkit.KIT?.procedural?.lod?.preview;
      if (preview && !previewAPI) throw new TypeError('Scale previews need the Ink Studio KIT');
      const scaleSurfaces = preview ? previewAPI.surfaces(preview, {
        view: camera.view, width, height, focal: height / (2 * tangent), ortho: camera.ortho,
        near: camera.clipNear, far: camera.clipFar
      }, reflectionCapture || plane !== zero ? undefined : frame.scalePreviewFrame) : undefined;
      const surfaceFrame={depthProxy,meshCodes:spriteMask?.meshCodes,lighting:emitterFrame,environment:env,view:camera.view,projection:camera.projection,range:[...camera.depthRange,fog.start??1,fog.max??0],
        pbr,pbrUniforms,bindPbr:pbr ? pbrTable.bind : undefined,
        fog:rgb(world.pal[fog.c??0]??0),arrivalPlane:plane,clip,section,paletteSampling:frame.paletteSampling,
        surfaceNormals:frame.surfaceNormals,semanticMipmaps:frame.semanticMipmaps,
        scanFrame:toolkit.KIT?.sceneMeshes?.scanFrame(frame.scanDaylight,frame.scanAppearance),simulationLayers,anchors:marks,engraving,eye:camera.eye,back,ortho:camera.ortho,
        sceneryBlockTextures:frame.sceneryBlockTextures,
        sceneryDeliveryFrame:reflectionCapture || plane!==zero ? undefined : frame.sceneryDeliveryFrame,
        onModelingWork:frame.onModelingWork,
        sun:spriteRows?.length || meshLayers.some(layer=>layer.semanticMipmaps?.some(row=>row.objectTexels||row.liveLight)) ?
          spriteLOD.light(world.liveSun,frame.liveSun,t) : undefined};
      if(layeredGlass){
        textureUnit(13,null);
        const indices = selection ? glassIndices.filter(i => selection.keep[i]) : glassIndices;
        drawGeometry(shader, glassBoundary, width, height, { u_idx: 3, u_snap: 1, u_hard: 0, tangent,
          indices, surfaceFrame: { ...surfaceFrame, lighting: undefined } });
        textureUnit(13,glassBoundary.textures[0]);
      }
      drawGeometry(shader, geometry, rw, rh, { u_idx: 0, u_snap: +(!raw && !colored), u_hard: 0,
        u_materialGridScale: samples, background, tangent, proxy, lookGrid:coverage&&[width,height],layeredGlass,surfaces,scaleSurfaces,spriteRows,spriteGrid:[width/rw,height/rh],surfaceFrame }, water, wet, env);
      if (colored)
        drawGeometry(shader, contours, width, height, { u_idx: 0, u_snap: 1, u_hard: 0, background, tangent,proxy,surfaces,scaleSurfaces,spriteRows,surfaceFrame }, water, wet, env);
      if (ordering) {
        function sortedGeometry(center, index, snap, scale) {
          const rect = surfaceOrderRegion(ordering.region, center.width, center.height);
          return surfaceOrdering.draw(center, camera, background, { ...ordering, rect }, index, uniforms => {
            data.bind();
            if (shader.gaussianRows) gaussianRows.rebind();
            if (shader.footprints) footprintBinding.rebind();
            if (shader.gaussianMaterials) gaussianMaterialBinding.rebind();
            gl.useProgram(shader.p);
            shader.set({
              ...uniforms, u_snap: snap, u_materialGridScale: scale, u_facingGridScale: scale,
              u_viewport: [center.width, center.height], u_focal: center.height / (2 * tangent)
            });
            pipeline.draw();
          });
        }
        geometry = sortedGeometry(geometry, 0, +(!raw && !colored), samples);
        if (colored) contours = sortedGeometry(contours, 1, 1, 1);
      }
      if (orderReport) frame.surfaceOrderReport?.(orderReport);
      if (captureSplatCoverage) captureSplatCoverage((colored ? contours : geometry).framebuffer);
      if (captureNormals) {
        drawGeometry(shader, normalOwner, width, height, { u_idx: 1, u_snap: 1, u_hard: 0,
          u_gaussianNormalCutoff: captureNormals.cutoff, tangent, proxy, surfaces, scaleSurfaces, spriteRows,
          surfaceFrame: { ...surfaceFrame, index: true } }, water, wet, env);
        captureNormals.read(normalOwner.framebuffer);
      }
      if (materialPalette) {
        drawGeometry(shader, gaussianMaterialPalette.coverage, width, height, {
          u_idx: 0, u_snap: 1, u_hard: 0, u_gaussianMaterialPass: true,
          tangent, proxy, surfaces, scaleSurfaces, spriteRows, surfaceFrame
        }, water, wet, env);
      }
      if (splatPaint) {
        shader.set({ u_splatPaintRows: splatPaintPass.bind(splatPaint.mask), u_splatPaintCount: splatPaint.mask.count });
        drawGeometry(shader, splatPaintPass.mask, width, height, {
          u_idx: 0, u_snap: 1, u_hard: 0, u_paintPass: true, u_splatPaintPass: true,
          tangent, proxy, surfaces, scaleSurfaces, spriteRows, surfaceFrame
        }, water, wet, env);
      }
      if (paintOnPatterns) {
        drawGeometry(shader, paintOverlay, width, height, { u_idx: 0, u_snap: 1, u_hard: 0, u_paintPass: true,
          tangent, lookGrid: coverage && [width, height] }, water, wet, env);
      }
      if (drawOwnership)
        drawGeometry(shader, owner, width, height, { u_idx: 1, u_snap: +(!liveBoundaryPass || !raw), u_hard: 0, tangent,proxy,surfaces,scaleSurfaces,spriteRows,historyOwner,surfaceFrame:{...surfaceFrame,index:true,graphDepth:packedDepth} });
      if(marks){
        marks.bindSites();
        drawGeometry(shader,restTarget,width,height,{u_idx:1,u_markPass:true,u_snap:1,u_hard:0,tangent,proxy,
          surfaces,scaleSurfaces,spriteRows,surfaceFrame:{...surfaceFrame,index:true}});
        markOutput=restTarget;
      }
      if (cover) {
        cover.environment = env;
        const foreground = { u_snap: +(!raw && !colored), u_hard: 0,
          u_portalCoverPlane: cover.plane, tangent, proxy:proxy && {...proxy,clip:cover.plane},
          surfaces, scaleSurfaces, spriteRows, surfaceFrame: { ...surfaceFrame, clip: cover.plane } };
        drawGeometry(shader, cover.buffer, width, height, { ...foreground, u_idx: 2,
          surfaceFrame: { ...foreground.surfaceFrame, index: true, portalCover: true } });
        drawGeometry(shader, cover.foreground, width, height, { ...foreground, u_idx: 0, portalForeground: true },
          water, wet, env);
      }
      if (vote) {
        drawGeometry(shader, vote.samples, width * 2, height * 2, { u_snap: 0, u_hard: 0,
          u_coverageVotePass: true, u_materialGridScale: 2, tangent, surfaces, scaleSurfaces, spriteRows,
          spriteGrid:[.5,.5], surfaceFrame });
      }
      const baseOwner = owner;
      const geometryOpts = { u_idx:0,u_snap:1,u_hard:0,tangent,proxy,surfaces,scaleSurfaces,spriteRows,surfaceFrame };
      if (filter) {
        const sampleOpts = { ...geometryOpts, u_snap:0, u_materialGridScale:2, spriteGrid:[.5,.5] };
        drawGeometry(shader,coverageResolve.taps,width*2,height*2,{...sampleOpts,background},water,wet,env);
        if (filterBuffers.sampleDepth) drawGeometry(shader,filterBuffers.sampleDepth,width*2,height*2,{
          ...sampleOpts,u_idx:1,surfaceFrame:{...surfaceFrame,index:true,graphDepth:true}
        });
        if (filterBuffers.centerMatte) {
          const matte = { coverageFilter:filter, background:[1,1,1,1] };
          drawGeometry(shader,filterBuffers.centerMatte,width,height,{...geometryOpts,...matte});
          drawGeometry(shader,filterBuffers.tapMatte,width*2,height*2,{...sampleOpts,...matte});
        }
      }
      if (liveBoundaryPass) {
        const samples = { ...geometryOpts,u_snap:0,u_materialGridScale:2,spriteGrid:[.5,.5],
          surfaceFrame:{...surfaceFrame,sceneryDeliveryFrame:undefined} };
        if (!colored) drawGeometry(shader,liveBoundaryPass.sampleColor,width*2,height*2,{...samples,background},water,wet,env);
        drawGeometry(shader,liveBoundaryPass.sampleOwner,width*2,height*2,{
          ...samples,u_idx:1,surfaceFrame:{...surfaceFrame,index:true,graphDepth:true}
        });
        const boundaryFrame={...historyFrame,source:frame.livePixelBoundaryKey??historyFrame.source,
          layout:world.count,output};
        const resolved=liveBoundaryPass.resolve(colored?contours:geometry,owner,world,boundaryFrame,liveBoundarySettings,
          liveBoundaryRegion,inverseView(camera.view,camera.eye),time,{
            cut:frame.resetRenderHistory===true||frame.resetLivePixelBoundaries===true,
            locks:projectedMarks?.coordinates,capture:frame.livePixelBoundaryFrame,highColor:colored?geometry:undefined
          });
        ({geometry,owner}=resolved);
        if (colored) contours=resolved.contours;
        data.bind();
      }
      if (curvePlan?.count) {
        finiteCurvePass ??= createFiniteCurvePass(gl,{
          target,table,program,maxSize,textureUnit,fullscreen,cutaway:cutawayAPI?.glsl,spriteLOD:spriteLOD?.glsl,
          strandLOD:toolkit.KIT.curveSplats.strandLod.glsl,print:engravingAPI?.print?.glsl,pbrGLSL:K.pbr?.glsl
        });
        const curveOptions = { clip, section, wet, water, env, layeredGlass,engraving,
          pbrUniforms,bindPbr:surfaceFrame.bindPbr,pbrSpecularSampling:pbr?.specularSampling,pbrBands:pbr?.hasBands,
          glass:layeredGlass ? glassBoundary.textures[0] : undefined,historyOwner,
          fog:{ color:surfaceFrame.fog,range:[fog.start ?? 1,fog.max ?? 0] },
          proxy:depthProxy && depthMeshPass.prepare(surfaceFrame,width,height) };
        finiteCurvePass.prepare(curvePlan);
        finiteCurvePass.draw(curvePlan,geometry,baseOwner,camera,curveOptions);
        if (cover) {
          const foreground = { ...curveOptions, cover:cover.plane, layeredGlass:false, glass:undefined };
          finiteCurvePass.draw(curvePlan,cover.buffer,baseOwner,camera,{ ...foreground, mode:'mask' });
          finiteCurvePass.draw(curvePlan,cover.foreground,baseOwner,camera,foreground);
        }
        if(markOutput)markOutput=finiteCurvePass.anchors(curvePlan,baseOwner,camera,
          {...curveOptions,wet:undefined,water:undefined,env:undefined,layeredGlass:false,glass:undefined},markOutput);
        if (ownsPixels || hybridDepthCapture)
          owner = finiteCurvePass.owner(curvePlan,baseOwner,camera,curveOptions);
        if (filterBuffers?.centerMatte)
          finiteCurvePass.draw(curvePlan,filterBuffers.centerMatte,baseOwner,camera,{...curveOptions,filterMatte:true});
        if (filter) {
          const tapCurves = prepareFiniteCurves(toolkit.KIT,world,geometryData,camera,width*2,height*2,
            {...curveSettings,restMarks:undefined,strandLODFrame:undefined,artScale:2});
          if (tapCurves.count) {
            const options={...curveOptions,artScale:2,
              proxy:depthProxy && depthMeshPass.prepare(surfaceFrame,width*2,height*2)};
            finiteCurvePass.prepare(tapCurves);
            finiteCurvePass.draw(tapCurves,coverageResolve.taps,filterBuffers.sampleDepth,camera,options);
            if (filterBuffers.tapMatte)
              finiteCurvePass.draw(tapCurves,filterBuffers.tapMatte,filterBuffers.sampleDepth,camera,{...options,filterMatte:true});
          }
        }
        data.bind();
        fine.bind();
        if (constraints && (coverage || masked)) {
          constraintSlots.bind(9);
          textureUnit(8,constraintRows);
        }
      }
      if (curvePlan && typeof frame.curveFrame === 'function')
        frame.curveFrame({ segments:curvePlan.count, tileReferences:curvePlan.references, peakTile:curvePlan.peakTile,
          ...(curvePlan.subdivision && { subdivision:curvePlan.subdivision }),
          ...(curvePlan.strandLod && { strandLod:curvePlan.strandLod }) });
      frame[graphOwner]?.(world, owner.textures[0], frame[graphOwner].depth ? owner.textures[2] : undefined, camera);
      if (hybridDepthCapture) {
        const depth = readHybridDepth(gl, owner.framebuffer, width, height, camera.depthRange);
        const depthInPortal = frame[portalDepth] ?? 0, reflection = frame[hybridReflection];
        frame.hybridDepthFrame({ view: depthInPortal ? 'portal' : reflection ? 'reflection' : 'camera',
          reflection, portalDepth: depthInPortal, width, height, depth });
      }
      gl.disable(gl.BLEND);
      if (filter) coverageResolve.finish(geometry.textures[0],filterBuffers);
      let sceneTexture=filter ? coverageResolve.filtered.textures[0] : geometry.textures[0];
      if(temporal){
        const result=lightingHistory.resolve(sceneTexture,owner,historyFrame,temporal,inverseView(camera.view,camera.eye),
          {cut:historyCut,capture:typeof frame.surfaceHistoryFrame==='function'});
        sceneTexture=result.texture;
        if(result.report)frame.surfaceHistoryFrame(result.report);
      }
      textureUnit(1,sceneTexture);
      textureUnit(2, geometry.textures[1]);
      textureUnit(5, paintOnPatterns ? paintOverlay.textures[0] : contours.textures[0]);
      textureUnit(6, paintOnPatterns ? paintOverlay.textures[1] : contours.textures[1]);
      textureUnit(9, owner.textures[0]);
      const pixelOutline = look.outline === true ? {} : look.outline;
      const mat = materials?.gpu;
      const patternedUniforms = patterned ? {
        u_idxTex: 9, u_pixTex: 10, u_dat: 0, u_depth: 2, u_tanF: tangent,
        u_portho: +camera.ortho, u_asp: camera.aspect, u_pview: camera.view, u_peye: camera.eye,
        u_fpG: camera.footprint, u_ptime: time, u_shape: +(frame.lightShape !== false),
        u_ol: +!!pixelOutline, u_olK: pixelOutline?.k ?? 2,
        u_olEps: pixelOutline ? (pixelOutline.eps ?? 0.6) / (world.detail || 12) / (2 * camera.radius) : 0,
        u_olGap: pixelOutline ? (pixelOutline.gap ?? 0.3) / (2 * camera.radius) : 1,
        u_patK4: mat.kinds, u_patP: mat.params, u_rmp4: mat.ramps, u_rmpN4: mat.lens, u_detail: world.detail || 12
      } : {};
      if (patterned) {
        keyProgram ??= program(sources.KEY_FS);
        textureUnit(15, null);
        pattern.bind();
        fullscreen(keyProgram, key.framebuffer, width, height, { u_pixOn: 1, ...patternedUniforms });
      }
      textureUnit(15, key.textures[0]);
      const sprites = patterned && anchorCount > 0 && frame.edges !== false;
      const spriteCover = sprites && drawSprites(camera, world, width, height, time, pixelOutline, water, pal, wet, clip, cover?.plane, env,
        relighting, section);
      const lines = outline && look.lines;
      const lineIndex = world.outline?.c ?? 0;
      const glassUniforms = {};
      if (glass) {
        const gaps = new Float32Array(24 * 4), count = Math.min(24, glass.gaps?.length ?? 0);
        if (count) for (let i = 0; i < count; i++) gaps.set(glass.gaps[i], i * 4);
        Object.assign(glassUniforms, {
          u_gview: camera.view, u_geye: camera.eye, u_gd: glass.dir,
          u_gw: [...glass.at, glass.r], u_gcam: [tangent * camera.aspect, tangent, ...camera.depthRange],
          u_gortho: +camera.ortho, u_gtime: t, u_gc: palette(glass.cols.map(c => world.pal[c]), 6),
          u_gm: +(glass.mode === 'gaps'), u_gq: glass.steps ?? 0,
          u_gh: gaps, u_ghn: count, u_gfl: glass.floor ?? -50
        });
      }
      if (finishArt) {
        copy ??= program(sources.COPY_FS);
        artPixelFinish ??= createArtPixelFinish(gl, { target, textureUnit, fullscreen, copy });
        artPixelFinish.buffer.resize(width, height);
      }
      const presentation = finishArt ? artPixelFinish.buffer.framebuffer : output;
      const destination = damping ? finished.framebuffer : presentation;
      if (scopedCleanup) {
        if (uploadedDetailFlags !== detail.flags) {
          detailTable.upload(detail.flags);
          uploadedDetailFlags = detail.flags;
        } else detailTable.bind();
        if (protectResolve) detailTable.bind(detailResolveUnit);
      }
      let resolve;
      const styledPattern = patterned && materialStyles, facingPattern = patterned && facing;
      const relitPattern = patterned && relighting;
      if (simpleOnly) {
        resolve = protectResolve ? (simpleGaps ??= program(simpleResolve(pixelSources({ protectGaps: true }).POST_FS))) : simple;
      } else if (markOutput || protectResolve || coverage || constraints || styledPattern || facingPattern || relitPattern || env?.media && patterned) {
        const cache = coverage ? contributionPost : constrainedPosts;
        const mode = (wet ? 1+Number(wet.airPockets)+2*Number(wet.layered)+4*Number(wet.atlas) : 0) +
          16*Number(masked) + 32*Number(layeredGlass) + 64*Number(styledPattern) + 128*Number(!!constraints) +
          256*Number(!!env?.media) + 512*Number(relitPattern)+1024*Number(!!env?.lit)+2048*Number(!!facingPattern) +
          4096*Number(protectResolve) + 8192*Number(patterned) + 16384*Number(colored) +
          32768*Number(paintOnPatterns)+65536*Number(!!markOutput);
        if (!cache.has(mode)) cache.set(mode,program(pixelSources({
          water:!!wet,airPockets:wet?.airPockets,layered:wet?.layered,atlas:wet?.atlas,
          paletteConstraints:!!constraints && !coverage,masked,contributions:coverage,layeredGlass,materialStyles:styledPattern, facingColors:facingPattern && facingAPI,
          environment:!!env?.media,emitterFog:!!env?.lit,relighting:relitPattern,protectGaps:protectResolve,
          marks:!!markOutput,patternedResolve:patterned,coloredResolve:colored,paintResolve:paintOnPatterns
        }).POST_FS));
        resolve = cache.get(mode);
        if (constraints && !coverage) {
          constraintSlots.bind();
          textureUnit(8,constraintRows);
          if (masked) textureUnit(7, owner.textures[1]);
        }
      } else resolve = wet ? (wet.post ??= program(wet.sources.POST_FS)) :
        resolveProgram(patterned || glass || world.ao || frameTone || grade);
      if (water) textureUnit(11, (colored ? contours : geometry).textures[2]);
      if (layeredGlass) {
        const at = geometry.textures.length - 2;
        textureUnit(4, geometry.textures[at]);
        textureUnit(7, geometry.textures[at + 1]);
        textureUnit(8, glassBoundary.textures[0]);
      }
      if (facingPattern) bones.bind(facingBoneUnit);
      if(markOutput){
        marks.bindRules();
        textureUnit(25,markOutput.textures[0]);
        textureUnit(30,markOutput.textures[1]);
      }
      const resolveOutput = cleaning ? cleaned.framebuffer : destination;
      const paintOutput = splatPaint ? splatPaintPass.source.framebuffer : resolveOutput;
      const paletteOutput = materialPalette ? gaussianMaterialPalette.source.framebuffer : paintOutput;
      fullscreen(resolve, vote ? vote.source.framebuffer : paletteOutput, width, height, {
        ...patternedUniforms, ...glassUniforms, ...bindTone(frameTone), ...water, ...env?.uniforms, u_waterMask: 11,
        ...(layeredGlass&&{u_glassBoundary:8,u_glassBack:4,u_glassThrough:7,
          u_glassRange:camera.depthRange,u_glassProjection:camera.projection}),
        ...(constraints ? { u_constraintSlots: 4, u_constraintRows: 8, u_constraintDefault: constraints.defaultSlot,
          u_constraintMap: constraintMap, u_constraintMask: maskAt, u_constraintLab: constraintLab, u_constraintPairs: constraintPairs,u_constraintAnchor:constraintAnchor,u_maskOwner: 7 } : {}),
        ...(markOutput && {...marks.uniforms,u_markWorld:25,u_markRest:30}),
        u_scene: 1, u_depth: 2, u_crispScene: 5, u_crispDepth: 6, u_keyTex: 15,
        u_pixOn: +patterned, u_keyOn: +patterned, u_sprOn: +sprites,
        u_paintOn: paintOnPatterns,
        ...(styledPattern && { u_materialStyles: 16 }),
        ...(facingPattern && { u_facingRows: facingUnit, u_facingCount: facing.count, u_disabledViewColors: disabledViewColors, u_facingBones: facingBoneUnit }),
        ...(relitPattern && { u_relight: relightUnit, u_relightFog: rgb(world.pal[fog.c ?? 0] ?? 0),
          u_relightFogRange: [fog.start ?? 1, fog.max ?? 0] }),
        u_idxTex: 9,
        ...(protectResolve && { u_detailFlags: detailResolveUnit }),
        u_sprLayer: 13, u_markA: +(cleaning || !!clusters), u_pal: pal, u_n: palCount, u_luma: 0,
        u_bypass: +(raw || !!portableStages), u_colored: +colored, u_dither: dither * 0.3 * (world.dither ?? 1),
        u_line: rgb(world.pal[lineIndex] ?? 0),
        u_remapLine: colorGrade && Number.isInteger(lineIndex) && lineIndex >= 0 && lineIndex < palCount ? lineIndex + 1 : 0,
        u_edge: outline ? (world.outline?.edge ?? 0) / (2 * camera.radius) : 0,
        u_lines: +!!lines, u_lineGap: lines ? (lines.gap ?? 0.3) / (2 * camera.radius) : 1,
        u_lineK: lines ? (lines.k ?? 0.55) : 1, u_lineLit: lines ? (lines.lit ?? 1.15) : 1,
        u_refract: world.refract ?? 6, u_ao: ao,
        u_aor: (world.ao?.range ?? 0.4) / (2 * camera.radius),
        u_gk: glass ? (glass.k || 0.14) : 0, u_bars: Math.min(0.45, world.anim?.bars ?? 0),
        u_grade: +!!grade, ...(grade ? { u_out: grade, u_outl: grade } : {})
      });
      if (vote) vote.finish(paletteOutput, width, height, pal, grade);
      if (materialPalette)
        gaussianMaterialPalette.finish(gaussianMaterial, (colored ? contours : geometry).textures[0], paintOutput, width, height);
      if (splatPaint)
        splatPaintPass.finish(splatPaint, (colored ? contours : geometry).textures[0], resolveOutput, width, height);
      if (cleaning) {
        if (scopedCleanup) detailTable.bind();
        textureUnit(8, cleaned.textures[0]);
        textureUnit(6, colored ? contours.textures[1] : geometry.textures[1]);
        fullscreen(clean, destination, width, height, {
          u_look: 8, u_dep: 6, u_keep: look.clean?.keep ?? 0.3,
          u_scopedOnly: !globalCleanup, u_hasDetailFlags: scopedCleanup,
          u_protectGaps: protectGaps,
          u_detailOwners: scopedCleanup ? 9 : 8, u_detailFlags: scopedCleanup ? 7 : 8,
          u_detailWater: scopedCleanup && !!water, u_waterMask: scopedCleanup && water ? 11 : 8,
          u_lineClean: +(frame.lineClean !== false && !!look.pixel)
        });
      }
      if(damping){
        copy ??= program(sources.COPY_FS);
        const activePalette=(grade??pal).subarray(0,palCount*3);
        if(!sameValues(previousPalette,activePalette))historyValid=false;
        previousPalette=activePalette;
        const vp=viewProjection(camera.projection,camera.view);
        const still=previousVP && sameValues(previousVP,vp);
        const result=paletteHistory.resolve(finished.textures[0],owner,historyFrame,holdSettings,
          inverseView(camera.view,camera.eye),{cut:historyCut||!historyValid||!still,hold:true});
        textureUnit(8,result.texture);
        fullscreen(copy,presentation,width,height,{u_src:8});
        historyValid=true;
        previousVP=vp;
      }
      const nativeRows = [...(foamRows || []), ...(highlightRows || [])];
      let protectedRows = [];
      if (projectedMarks) {
        let visible;
        if(projectedMarks.coordinates.size){
          probeNativeDepth ??= createNativeDepthProbe(gl,resources,target,textureUnit);
          visible=probeNativeDepth(projectedMarks.coordinates,owner.textures[2],camera.depthRange);
        }
        const result=retainedAPI.resolve(projectedMarks,key=>visible?.get(key));
        protectedRows=result.pixels;
        frame.protectedPixelsFrame?.(result.report);
      }

      function drawCels(to) {
        if (!nativeRows.length && !protectedRows.length) return;
        drawNativeCels ??= createNativeCelDraw(gl, resources, textureUnit, nativeCelAPI.pixels);
        drawNativeCels(nativeRows, width, height, to, owner.textures[2], camera.depthRange, protectedRows);
      }
      if (finishArt) {
        let project;
        function artProject() {
          if (!project) {
            const sampler = toolkit.KIT.splatFrames.create(world, time, toolkit.KIT.M);
            project = clusterAPI.projector(camera, i => sampler(i).position());
          }
          return project;
        }
        const continuity = clusters?.coherent && { history:coherenceHistory,
          source:frame.colorClusterSource ?? frame.renderHistoryKey ?? source,
          time:t, project:artProject(), projection:camera.ortho,
          reset:frame.resetRenderHistory || frame.resetColorClusters, captureLabels:frame.captureColorClusterLabels,
          palette:Array.from((grade ?? pal).subarray(0,palCount*3)) };
        const detail = clusters && (cleaning ? cleaned : damping ? finished : artPixelFinish.buffer);
        function sceneArt(art, passes) {
          if (clusters) {
            const { rgb, ...report } = clusterAPI.apply(art, passes, clusters, world, continuity);
            frame.colorClustersFrame?.(report);
            if (rgb !== art.rgb) art = { ...art, rgb };
          }
          if (!boundaries) return art;
          const result = boundaryAPI.apply(art, passes, boundaries, world, {
            time: t, captureSites: frame.captureBoundarySites
          });
          frame.boundaryCelsFrame?.(result.report);
          return { ...art, rgb: result.rgb };
        }
        function shotArt(art, passes) {
          if (captureCorrection) frame.shotCorrectionSurfaceFrame?.(
            correctionAPI.capture(art, passes, world, artProject()));
          if (!corrections) return art.rgb;
          const { edits, ...report } = correctionAPI.resolve(art, passes, corrections, world,
            { time:t, shotCorrectionFrame:frame.shotCorrectionFrame, project:artProject(),
              shotCorrectionHits:frame.shotCorrectionHits });
          frame.shotCorrectionsFrame?.(report);
          return correctionAPI.apply(art.rgb, edits);
        }
        if (finishShotArt && (nativeRows.length || protectedRows.length)) {
          let source = artPixelFinish.buffer;
          if (finishSceneArt) {
            correctionStage ??= target();
            correctionStage.resize(width, height);
            artPixelFinish.finish(owner, detail, correctionStage.framebuffer, width, height,
              (art, passes) => sceneArt(art, passes).rgb);
            source = correctionStage;
          }
          drawCels(source.framebuffer);
          artPixelFinish.finish(owner, null, output, width, height, shotArt, { source, depthRange:camera.depthRange });
        } else {
          artPixelFinish.finish(owner, detail, output, width, height, (art, passes) => {
            const resolved = sceneArt(art, passes);
            return finishShotArt ? shotArt(resolved, passes) : resolved.rgb;
          }, { depthRange:finishShotArt ? camera.depthRange : undefined });
          drawCels(output);
        }
      } else drawCels(output);
      gl.activeTexture(gl.TEXTURE0);
      lastWorld = worldIdentity;
      lastInstances = group;
      lastData = world.data;
      lastStaged = Boolean(portableStages);
      lastDiffuseRamps = Boolean(diffuseRamps);
      lastCount = world.count;
      lastOutput = output;
      lastDetail = detail;
      lastQualityKey = qualityKey;
      lastWater = waterScene;
      lastWaterTime = frame.time;
      if (!reflectionCapture) {
        frame[frameCamera]?.(camera);
        frame[inspectionDepth]?.(world, geometry.textures[1], camera);
      }
      return spriteCover;
    }

    function bindPortalAperture(mask, depth, fallback) {
      if (!mask) {
        textureUnit(4, fallback);
        return;
      }
      const { width, height, data } = mask;
      if (width > maxSize || height > maxSize) throw new RangeError('Portal aperture exceeds the WebGL texture size limit');
      gl.activeTexture(gl.TEXTURE4);
      const slot = portalApertures[depth] ??= { texture: resources.texture() };
      textureUnit(4, slot.texture);
      if (slot.mask === mask) return;
      const alignment = gl.getParameter(gl.UNPACK_ALIGNMENT);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, Uint8Array.from(data));
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, alignment);
      slot.mask = mask;
    }

    function present(world, frame, width, height, output, camera, entryPlane = zero, depth = 0,
      reflection = false, trailStart) {
      const portal = depth < portalLimit ? frame.portal : undefined, fade = depth === 0 ? frame.fade ?? 0 : 0;
      if (!portal && !fade) return render(world, frame, width, height, output, camera, entryPlane, undefined, reflection, trailStart);
      camera ??= makeCamera(world, frame, width, height);
      const opening = portal?.opening;
      const front = opening && planeInView(anchorPlane(opening.from), camera);
      const visible = front && (camera.ortho ? front[2] > 1e-7 : front[3] > 1e-6);
      if (!visible && !fade) return render(world, frame, width, height, output, camera, entryPlane, undefined, reflection, trailStart);
      if (width > maxSize || height > maxSize || width > viewportLimit[0] || height > viewportLimit[1])
        throw new RangeError('Ink Studio frame exceeds the WebGL viewport or texture size limit');
      if (!output) {
        if (gl.canvas.width !== width) gl.canvas.width = width;
        if (gl.canvas.height !== height) gl.canvas.height = height;
      }
      const buffers = reflection ? reflectedPortalTargets : portalTargets;
      const sourceBuffer = buffers.source ??= target();
      sourceBuffer.resize(width, height);
      let destinationBuffer;
      if (visible) {
        destinationBuffer = buffers.destinations[depth] ??= target();
        destinationBuffer.resize(width, height);
        buffers.mask ??= target();
        buffers.foreground ??= target(2);
        buffers.mask.resize(width, height);
        buffers.foreground.resize(width, height);
        const nextCamera = withCameraPose(camera, opening.camera(cameraPose(camera)), reflection);
        const { raw, outline, dither, scale, clean, lineClean, colored, tone, edges, water, time, skyTime, cameraTime,
          stepAnimation, artDirection, renderStages, strictContract, environment, colorClusters, boundaryCels,
          stableTransparency, resetTransparency, stableDetail, stable, ditherSpace, engraving, engravingLight, screenPrint, brushwork,
          passQuality, silhouetteProtection, jointPriority, scalePreviews } = frame;
        const expand = portalReconstructionInstances[depth] ??= createExpandedInstances();
        const destination = reconstructFrame(portal.destination, { raw, outline, dither, scale, clean, lineClean, colored, tone,
          edges, water, time, skyTime, cameraTime, stepAnimation, artDirection, renderStages, strictContract,
          environment, colorClusters, boundaryCels, stableTransparency, resetTransparency, stableDetail, stable,ditherSpace,
          engraving,engravingLight,screenPrint,brushwork,passQuality,
          silhouetteProtection, jointPriority, scalePreviews, atmosphere: frame.atmosphere, coverageAware: frame.coverageAware,
          reflections: frame.reflections, ...portal.destinationFrame,
          ...(reflection && { water: false, reflections: false, [hybridReflection]: frame[hybridReflection] }),
          curveMode: portal.destinationFrame?.curveMode ?? frame.curveMode,
          curveSplats: portal.destinationFrame?.curveSplats ?? frame.curveSplats,
          curveSubdivision: portal.destinationFrame?.curveSubdivision ?? frame.curveSubdivision,
          curveLod: portal.destinationFrame?.curveLod ?? frame.curveLod,
          hybridDepthFrame: frame.hybridDepthFrame, damp: false, [portalDepth]: depth + 1 }, expand);
        // Resolve the deeper view before reusing the source and coverage targets.
        present(destination.world, destination.frame, width, height, destinationBuffer.framebuffer,
          nextCamera, anchorPlane(opening.to), depth + 1, reflection);
      }
      const cover = visible ? { buffer: buffers.mask, foreground: buffers.foreground, plane: front } : undefined;
      const spriteCover = render(world, visible ? { ...frame, damp: false } : frame, width, height,
        sourceBuffer.framebuffer, camera, entryPlane, cover, reflection, trailStart);
      const env = cover?.environment, key = +!!env?.media + 2 * !!env?.lit;
      if (!portalPrograms.has(key)) portalPrograms.set(key, program(portalComposite(env?.media, env?.lit)));
      const portalProgram = portalPrograms.get(key);
      const source = sourceBuffer.textures[0], mask = visible ? opening.mask : undefined;
      textureUnit(0, source);
      textureUnit(1, destinationBuffer?.textures[0] ?? source);
      textureUnit(2, visible ? buffers.mask.textures[0] : source);
      textureUnit(5, visible ? buffers.foreground.textures[0] : source);
      textureUnit(3, spriteCover || source);
      bindPortalAperture(mask, depth, source);
      fullscreen(portalProgram, output, width, height, { ...env?.uniforms, u_source: 0, u_destination: 1,
        u_cover: 2, u_foreground: 5, u_spriteCover: 3, u_hasSpriteCover: +!!spriteCover,
        u_aperture: 4, u_painted: +!!mask, u_entryPlane: entryPlane,
        u_opening: visible ? (opening.shape === 'ellipse' ? 2 : 1) : 0, u_size: [width, height], u_fade: fade,
        ...(visible ? { u_origin: opening.from.origin, u_right: opening.from.right, u_up: opening.from.up,
          u_normal: opening.from.normal, u_half: [opening.halfWidth, opening.halfHeight],
          u_eye: camera.eye, u_portalView: camera.view, u_far: camera.clipFar,
          u_settings: [camera.tangent * camera.aspect, camera.tangent, +camera.ortho] } : {}) });
      gl.activeTexture(gl.TEXTURE0);
    }
    const viewportLimit = gl.getParameter(gl.MAX_VIEWPORT_DIMS);
    let composite;
    function presentMixed(world, frame, width, height, output) {
      const definition = frame.composite ?? world.look?.composite;
      if (definition === undefined || definition === false) return present(world, frame, width, height, output);
      const api = toolkit.KIT?.frameComposite;
      if (!api) throw new Error('Mixed-resolution frames need the Ink Studio KIT');
      const plan = api.prepare(definition, width, height, world.look?.pixelSize ?? 1);
      if ([plan, plan.world].some(({ w, h }) => w > Math.min(maxSize, viewportLimit[0]) ||
          h > Math.min(maxSize, viewportLimit[1])) || plan.layers.some(({ w, h }) => w > maxSize || h > maxSize))
        throw new RangeError('Composite frame exceeds the WebGL viewport or texture size limit');
      const time = frame.interfaceTime ?? frame.time ?? world.lookSnapshot?.time ?? 0;
      let camera, guide;
      const drawFrame = { ...frame,
        ...(plan.needsCamera && captureFrameCamera(frame, view => { camera = view; })),
        ...(plan.needsDepth && { [graphOwner]: Object.assign((scene, owner, depth, view) => {
          guide = { texture:depth, range:view.depthRange };
          frame[graphOwner]?.(scene, owner, frame[graphOwner].depth ? depth : undefined, view);
        }, { depth:true }) }) };
      const readLayers = () => {
        let index = 0;
        return api.layers(plan, world, time, camera, gl, frame.time, (spec, options) => {
          const pass = mixedLighting[index++] ??= createPixelLightingPass(gl, toolkit.KIT.pixelLighting, resources);
          return pass.draw(spec, options);
        });
      };
      const layers = plan.needsCamera || plan.layers.some(layer => layer.lit) ? readLayers : readLayers();
      composite ??= createFrameComposite(gl, { program, resources, target, textureUnit, fullscreen });
      composite(world, drawFrame, plan, layers, output, present, plan.needsDepth ? () => guide : undefined);
    }
    let gridRenderer;
    function presentGrid(world, frame, width, height, output) {
      const definition = frame.renderGrid ?? world.look?.renderGrid;
      if (definition === undefined || definition === false) return presentGraph(world, frame, width, height, output);
      const api = toolkit.KIT?.renderGrid;
      if (!api) throw new Error('Tessellated grids need the Ink Studio KIT');
      // Immutable prepared plans are recommended for a host's animation loop.
      const plan = api.prepare(definition);
      api.dimensions(plan, width, height);
      if (width > Math.min(maxSize, viewportLimit[0]) || height > Math.min(maxSize, viewportLimit[1]))
        throw new RangeError('Render grid exceeds the WebGL viewport or texture size limit');
      gridRenderer ??= createRenderGrid(gl, { api, program, target, textureUnit, fullscreen });
      gridRenderer(world, frame, plan, width, height, output, presentGraph);
    }
    const mixedLighting = [];
    let graphLighting, renderGraph;
    function presentGraph(world, frame, width, height, output) {
      const definition = frame.renderGraph ?? world.look?.renderGraph;
      const capture = frame.renderGraphCapture;
      if ((definition === false || definition === undefined) && capture === undefined) {
        renderGraph?.reset();
        return presentMixed(world, frame, width, height, output);
      }
      const api = toolkit.KIT?.renderGraph;
      if (!api) throw new Error('Render graphs need the Ink Studio KIT');
      const plan = api.prepare(definition, capture);
      if (!plan || !plan.passes.length && !plan.captures.size) {
        renderGraph?.reset();
        return presentMixed(world, frame, width, height, output);
      }
      if (width > Math.min(maxSize, viewportLimit[0]) || height > Math.min(maxSize, viewportLimit[1]))
        throw new RangeError('Render graph exceeds the WebGL viewport or texture size limit');
      renderGraph ??= createRenderGraph(gl, { api, program, resources, target, textureUnit, fullscreen, maxSize,
        drawLighting(spec, options) {
          graphLighting ??= createPixelLightingPass(gl, toolkit.KIT.pixelLighting, resources);
          return graphLighting.draw(spec, options);
        }
      });
      renderGraph(world, frame, plan, width, height, output, presentMixed);
    }
    let inspection;
    function presentFrame(world, frame, width, height, output) {
      frame = contract?.options(world, frame, toolkit.KIT.timeDomains?.frame) ??
        { ...frame, time: frame.time ?? frame.t };
      const source = frame.renderHistoryKey ?? world;
      ({ world, frame } = reconstructFrame(world, frame, reconstructionInstances));
      if (source !== world) frame = { ...frame, renderHistoryKey: source };
      const presentation = toolkit.KIT?.presentationTimeline?.sample(world, frame.interfaceTime ?? frame.time ?? world.lookSnapshot?.time ?? 0, frame);
      if (presentation) frame = { ...frame, ...presentation };
      if (frame.onCompatibility) {
        const mode = frame.gaussianColor ?? world.gaussianHarmonics?.colorMode ?? 'baked';
        const raw = mode === 'baked' ? frame.raw : mode === 'rgb';
        frame.onCompatibility(contract.review(world, { ...frame, raw }));
      }
      const definition = frame.inspection ?? world.look?.inspection;
      if (definition === undefined || definition === false) return presentGrid(world, frame, width, height, output);
      const api = toolkit.KIT?.inspection;
      if (!api) throw new Error('Target inspection needs the Ink Studio KIT');
      const settings = api.prepare(definition);
      if (!settings.targets.length) return presentGrid(world, frame, width, height, output);
      inspection ??= createInspection(gl, { api, program, resources, target, textureUnit, fullscreen, maxSize, viewportLimit });
      inspection(world, frame, settings, width, height, output, presentGrid);
    }
    return { render: presentFrame, dispose() {
      for (const pipeline of surfacePipelines) pipeline?.dispose();
      for (const pipeline of brickPipelines.values()) pipeline.dispose();
      depthMeshPass?.dispose();
      spriteReconstructionPipeline?.dispose();
      for (const preview of previewPipelines) preview?.dispose();
      emitterSurfacePipeline?.dispose();
      pipeline.dispose();
      resources.dispose();
    } };
  } catch (error) {
    pipeline?.dispose();
    resources.dispose();
    throw error;
  }
}

export function createRenderer(canvas, { toolkit, specialize = true, reuseOrder = true, resolve = 'auto', contextAttributes, input } = {}) {
  if (typeof canvas?.getContext !== 'function')
    throw new TypeError('Ink Studio needs a canvas or OffscreenCanvas');
  if (typeof toolkit?.PixelLook?.glsl !== 'string')
    throw new TypeError('Pass the Ink Studio toolkit, including PixelLook, to createRenderer');
  if (resolve !== 'auto' && resolve !== 'simple')
    throw new RangeError("Ink Studio WebGL resolve must be 'auto' or 'simple'");
  const gl = canvas.getContext('webgl2', {
    ...contextAttributes, alpha: false, antialias: false, depth: false, stencil: false
  });
  if (!gl) throw new Error('Ink Studio GPU rendering requires WebGL 2');
  const backendOptions = { specialize, reuseOrder, simpleOnly: resolve === 'simple' };
  let backend = createBackend(gl, toolkit, backendOptions);
  let controls;
  if(input!==undefined&&input!==false) {
    try { controls=createViewerInput(canvas,{...input,toolkit}); }
    catch(error){ backend.dispose();throw error; }
  }
  let disposed = false, picking, streamAssemblies = new WeakMap();
  const pickInstances = createExpandedInstances();
  const lose = event => {
    event.preventDefault();
    // The restored context cannot delete or reuse the old resources.
    backend = null;
  };
  canvas.addEventListener?.('webglcontextlost', lose);
  function validateWorld(world, frame) {
    if (!(world?.data instanceof Float32Array) || !Number.isSafeInteger(world.count) ||
        world.count < 0 || world.count * 16 > world.data.length || !world.pal?.length)
      throw new TypeError('Render expects a world with packed splats, a count, and a palette');
    if (world.fine && world.fine.length < world.count * 12 || world.pix && world.pix.length < world.count * 8)
      throw new RangeError('Ink Studio material data is shorter than the splat count');
    if (world.materialPixelStyles != null && (!(world.materialPixelStyles instanceof Float32Array) ||
        world.materialPixelStyles.length < world.count * 4))
      throw new RangeError('Ink Studio material pixel styles must cover the splat count');
    if (world.opticalSplats != null && (!(world.opticalSplats instanceof Float32Array) ||
        world.opticalSplats.length !== world.count * 12))
      throw new RangeError('Ink Studio optical materials must match the splat count');
    const plan = frame.instances === undefined ? null : instancePlan(frame.instances);
    if (plan) validateSize(world.count,plan);
    const logicalCount = world.count + (plan?.count ?? 0);
    if (frame.order !== undefined &&
        (!(frame.order instanceof Int32Array) || frame.order.length !== logicalCount ||
         frame.order.some(index => index < 0 || index >= logicalCount)))
      throw new TypeError('Draw order must contain valid Int32 splat indices');
    for (const [key, perRow] of [['changedRows', 512], ['changedFineRows', 512], ['changedPatternRows', 1024]]) {
      const rows = frame[key];
      if (rows !== undefined && (!Array.isArray(rows) || rows.some(row =>
        !Number.isInteger(row) || row < 0 || row >= Math.max(1, Math.ceil(world.count / perRow)))))
        throw new RangeError(key + ' must contain valid ' + perRow + '-splat row indices');
    }
    if (![frame.time ?? frame.t ?? 0, frame.bornAt ?? 0, frame.scale ?? 1, frame.dither ?? 0.35].every(Number.isFinite))
      throw new RangeError('Ink Studio frame parameters must be finite');
  }
  const api = {
    ...(controls&&{input:controls}),
    describeFrame(world, frame = {}) {
      const contract = toolkit.KIT?.renderContract;
      if (!contract) throw new Error('Frame descriptions need the Ink Studio KIT');
      return contract.describe(world, { ...frame, width: frame.width ?? frame.w ?? frame.target?.width ?? canvas.width,
        height: frame.height ?? frame.h ?? frame.target?.height ?? canvas.height }, toolkit.KIT.timeDomains?.frame);
    },
    cameraPose(world, frame = {}) {
      return cameraPose(makeCamera(world, frame, frame.width ?? frame.w ?? frame.target?.width ?? canvas.width,
        frame.height ?? frame.h ?? frame.target?.height ?? canvas.height));
    },
    renderStream(stream, cameraWorld, { place, portals = [], selected, assemble, ...frame } = {}) {
      if (disposed) throw new Error('The Ink Studio renderer has been disposed');
      if (typeof stream?.advance !== 'function' || typeof assemble !== 'function')
        throw new TypeError('Stream rendering needs a scene stream and a world assembler');
      const width = frame.width ?? frame.w ?? frame.target?.width ?? canvas.width;
      const height = frame.height ?? frame.h ?? frame.target?.height ?? canvas.height;
      const { snapshot, camera } = captureStreamingCamera(cameraWorld, frame, width, height);
      const scene = stream.advance({ place, camera, portals, selected, portalLimit: 1 });
      const previous = streamAssemblies.get(stream), current = new Map();
      function worldFor({ id, place, entries }) {
        const saved = previous?.assemble === assemble && previous.worlds.get(id);
        const same = saved?.place === place && entries.length === saved.entries.length &&
          entries.every((row, i) => {
            const old = saved.entries[i];
            return old.key === row.key && old.revision === row.revision && old.value === row.value;
          });
        const world = same ? saved.world : assemble(place, entries);
        if (world?.count > entries.reduce((sum, row) => sum + row.splats, 0))
          throw new RangeError('Assembled scene exceeds its packed splat budget');
        current.set(id, { place, entries, world });
        return world;
      }
      const world = worldFor(scene.views[0]), next = scene.views[1];
      const portal = next?.entries.length ? { opening: next.opening, destination: worldFor(next) } : undefined;
      api.render(world, { ...frame, cameraSnapshot: snapshot, portal });
      streamAssemblies.set(stream, { assemble, worlds: current });
      return { world, portal, scene };
    },
    render(world, frame = {}) {
      if (disposed) throw new Error('The Ink Studio renderer has been disposed');
      if (gl.isContextLost()) throw new Error('The Ink Studio WebGL context is lost');
      validateWorld(world, frame);
      if (frame.onCompatibility !== undefined && (typeof frame.onCompatibility !== 'function' || !toolkit.KIT?.renderContract))
        throw new TypeError('Frame review needs an onCompatibility callback and the Ink Studio KIT');
      let portal = frame.portal, portalWorld = world;
      for (let depth = 0; portal && depth < portalLimit; depth++) {
        const { source, destination, opening, destinationFrame = {} } = portal;
        if ((source && source !== portalWorld) || typeof opening?.camera !== 'function' || !opening.from || !opening.to)
          throw new TypeError('Expected an oriented portal view for the rendered world');
        validateWorld(destination, destinationFrame);
        portalWorld = destination;
        portal = destinationFrame.portal;
      }
      if (frame.fade !== undefined && (!Number.isFinite(frame.fade) || frame.fade < 0 || frame.fade > 1))
        throw new RangeError('Scene fade opacity must be in [0, 1]');
      const target = Array.isArray(frame.target) || ArrayBuffer.isView(frame.target) ? undefined : frame.target;
      const width = frame.width ?? frame.w ?? target?.width ?? canvas.width;
      const height = frame.height ?? frame.h ?? target?.height ?? canvas.height;
      const output = target?.framebuffer ?? null;
      if (target && (width !== target.width || height !== target.height || !output || !gl.isFramebuffer(output)))
        throw new TypeError('Draw target must match the frame size and use the renderer WebGL context');
      if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width < 1 || height < 1)
        throw new RangeError('Ink Studio frame width and height must be positive integers');
      backend ??= createBackend(gl, toolkit, backendOptions);
      let captures, effectPixel;
      const settings = frame.renderGraphCapture === undefined && frame.effectPixel === undefined ? frame : {
        ...frame,
        ...(frame.renderGraphCapture !== undefined && { [graphCapture]: value => { captures = value; } }),
        ...(frame.effectPixel !== undefined && { [graphEffectPixel]: value => { effectPixel = value; } })
      };
      backend.render(world, settings, width, height, output);
      if (picking?.world === world && (frame.changedRows?.length !== 0 || world.anim?.bones?.length)) picking = undefined;
      return frame.effectPixel === undefined ? captures : { effectPixel, ...(captures && { passes: captures }) };
    },
    inspectEffectPixel(world, at, frame = {}) {
      const result = api.render(world, { ...frame, inspection: false, renderGrid: false, effectPixel: at });
      if (!result.effectPixel?.passes.length) throw new RangeError('Effect inspection needs a visible material-response render graph');
      return result.effectPixel;
    },
    captureTurntable(world, options) {
      if (disposed) throw new Error('The Ink Studio renderer has been disposed');
      if (gl.isContextLost()) throw new Error('The Ink Studio WebGL context is lost');
      return captureGaussianTurntable(gl, api.render, world, options);
    },
    pick(world, ray, frame = {}) {
      if (disposed) throw new Error('The Ink Studio renderer has been disposed');
      validateWorld(world,frame);
      const create = toolkit.KIT?.procedural?.contributions?.create;
      if (!create) throw new Error('Contribution picking needs the Ink Studio KIT');
      const viewPoses = toolkit.KIT?.viewPoses;
      if (frame.viewPoses!==false && viewPoses?.forWorld(world)?.active) {
        const camera = makeCamera(world,frame,frame.width ?? frame.target?.width ?? canvas.width,
          frame.height ?? frame.target?.height ?? canvas.height);
        world = viewPoses.frame(world,{...cameraPose(camera),ortho:camera.ortho});
      }
      const group = frame.instances, revision = group && instancePlan(group).revision;
      const time = frame.time ?? frame.t ?? world.lookSnapshot?.time ?? 0;
      const withdrawal = frame.sampleWithdrawal ?? world.sampleWithdrawal ?? 0;
      const splatCoverage = frame.splatCoverage ?? world.look?.splatCoverage;
      const materialStyles = world.materialPixelStyles;
      const footprintEnabled=(frame.splatFootprints??world.look?.splatFootprints)!==false;
      const marks = world.dataMarks?.snapshot();
      const definition = frame.reconstruction ?? world.look?.reconstruction;
      const reconstruction = definition === undefined || definition === false ? undefined : JSON.stringify(definition);
      if (picking?.world !== world || picking.data !== world.data || picking.count !== world.count || picking.opticalSplats !== world.opticalSplats ||
          picking.group !== group || picking.revision !== revision || picking.withdrawal !== withdrawal ||
          picking.footprintEnabled !== footprintEnabled || picking.footprints !== world.splatFootprints || picking.splatCoverage !== splatCoverage || picking.materialStyles !== materialStyles || picking.marks !== marks ||
          picking.reconstruction !== reconstruction || picking.time !== time &&
            (reconstruction !== undefined || world.moving?.length || world.grow || world.anim?.bones)) {
        let posed = group ? pickInstances(world, group, frame.changedRows).world : world;
        if (reconstruction !== undefined) posed = reconstruct(toolkit, posed, frame, definition);
        const picker = create(posed, { time, sampleWithdrawal: withdrawal, splatCoverage, cutaway:false, splatFootprints:footprintEnabled });
        picking = { world, data: world.data, count: world.count, opticalSplats: world.opticalSplats,
          group, revision, time, withdrawal, reconstruction, splatCoverage, materialStyles, marks, footprints:world.splatFootprints, footprintEnabled, picker };
      }
      const section = toolkit.KIT.cutaway?.prepare(ray?.cutaway ?? frame.cutaway ?? world.look?.cutaway);
      let camera = ray?.camera;
      if (section?.range && !camera) {
        const view = makeCamera(world,frame,frame.width ?? frame.target?.width ?? canvas.width,
          frame.height ?? frame.target?.height ?? canvas.height);
        camera = { ...cameraPose(view),setback:view.setback };
      }
      let footprintView=ray?.footprintView;
      if(footprintEnabled&&!footprintView&&(world.splatFootprints||group&&instancePlan(group).hasSplatFootprints)){
        const width=frame.width??frame.target?.width??canvas.width,height=frame.height??frame.target?.height??canvas.height;
        footprintView={...makeCamera(world,frame,width,height),width,height,variance:frame.raw?.3:.15,snap:!frame.raw&&!frame.colored};
      }
      const spriteLOD=toolkit.KIT?.sprite?.reconstruction;
      const pickFrame={water:frame.water!==false,...ray,meshes:frame.meshes!==false&&ray?.meshes!==false,
        brickMeshes:frame.brickMeshes!==false&&ray?.brickMeshes!==false,cutaway:section,camera,footprintView};
      if(world.reliefLOD!==undefined&&pickFrame.meshes){
        const width=frame.width??frame.target?.width??canvas.width,height=frame.height??frame.target?.height??canvas.height;
        const view=makeCamera(world,frame,width,height);
        pickFrame.additionalMeshes=toolkit.KIT.procedural.lod.relief.frame(world,frame,{
          view:view.view,width,height,focal:view.projection[5]*height*.5,ortho:view.ortho,
          near:view.clipNear,far:view.clipFar,setback:view.setback
        });
      }
      if(world.spriteReconstruction&&spriteLOD){
        const width=frame.width??frame.target?.width??canvas.width,height=frame.height??frame.target?.height??canvas.height;
        const view=makeCamera(world,frame,width,height);
        const {eye,back}=cameraPose(view);
        const plan=spriteLOD.resolve(world,{eye,back,ortho:view.ortho,focal:view.projection[5]*height*.5,
          setback:view.setback,time,mode:frame.spriteReconstruction,impostorBlend:frame.impostorBlend,protectedLOD:frame.protectedLOD});
        if(plan?.mask){
          const {projection,view:m}=view;
          const axial=([x,y,z])=>-(m[2]*x+m[6]*y+m[10]*z+m[14]);
          const pixelAt=([x,y,z])=>{
            const w=view.ortho?1:axial([x,y,z]);
            if(w<=0)return null;
            return [(m[0]*x+m[4]*y+m[8]*z+m[12])*projection[0]/w*width*.5+width*.5,
              (m[1]*x+m[5]*y+m[9]*z+m[13])*projection[5]/w*height*.5+height*.5];
          };
          const clip=point=>axial(point)>=view.clipNear&&(!section||
            toolkit.KIT.cutaway.margin(section,...point,axial(point)-view.setback)>=0);
          const accept=ray?.accept;
          pickFrame.accept=hit=>{
            const code=hit.kind==='mesh' ? plan.mask.meshCodes.get(hit.mesh) : plan.mask.codes[hit.index], pixel=code&&pixelAt(hit.point);
            return (!code||pixel&&spriteLOD.geometry(code,...pixel))&&(!accept||accept(hit));
          };
          pickFrame.extraHits=ray=>spriteLOD.hits(plan,{...ray,pixelAt,clip});
        }
      }
      return picking.picker.ray(pickFrame);
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      streamAssemblies = new WeakMap();
      try { controls?.dispose(); }
      finally {
        backend?.dispose();
        canvas.removeEventListener?.('webglcontextlost', lose);
      }
    }
  };
  return api;
}
