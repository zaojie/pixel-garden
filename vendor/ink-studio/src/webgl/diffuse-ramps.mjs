const diffuse = layer => layer?.type === 'diffuse';

// Resolve opt-in diffuse finishes on the posed carriers before normal splat
// compositing. The software renderer uses the same layer at fragment positions.
export function applyDiffuseRamps(toolkit, world, frame, camera, time, poseTime, data, patterned) {
  if (frame.surface === false || frame.diffuseRamps === false || !world.surfaceData || !world.surfaces?.length ||
      !world.surfaces.some(({settings}) => Object.values(settings.layers ?? {}).some(diffuse))) return null;
  const K = toolkit.KIT, surface = toolkit.SurfaceLook ?? K?.surface;
  if (!surface || !K?.splatFrames) throw new Error('Diffuse ramps need the Ink Studio KIT');

  const light = world.surfaceLight && surface.prepareLightTrack(world.surfaceLight)(time);
  const ramps = surface.lightRamps(frame.raw ? false : frame.surfaceLightRamps ?? world.look?.surfaceLightRamps);
  const materials = world.surfaces.map(({settings, light: baked}) => {
    const definition = surface.withLightRamps(settings, ramps);
    const layers = Object.entries(definition.layers ?? {}).filter(([,layer]) => diffuse(layer));
    if (!layers.length) return null;
    const material = surface.prepare({ ...definition, layers:Object.fromEntries(layers), rainRipple:undefined, opacity:undefined },
      world.pal, light || baked, world.surfaceEffects, {
        paletteRoles:world.paletteRoles, paletteConstraints:false, facingColors:false, directionalColors:false,
        skipMasks:frame.lookMasks === false, paintShadowBands:frame.shadowBandPaint !== false,
        shadeNormals:frame.shadeNormals !== false
      });
    return material.diffuseLayers ? material : null;
  });
  const posed = K.splatFrames.create(world, poseTime, K.M);
  const back = [camera.view[2],camera.view[6],camera.view[10]];
  const metadata = world.surfaceData, stride = surface.STRIDE;
  const base = [0,0,0], result = [0,0,0,1,0];
  let colors, pixels;
  for (let i=0; i<world.count; i++) {
    const tag = i*stride, q = i*16;
    const material = materials[metadata[tag]-1];
    if (!material || data[q+3]<=0) continue;
    const pose = posed(i), at = pose.position(), packed = data[q+7];
    const view = camera.ortho ? back : camera.eye.map((value,k) => value-at[k]);
    const normal = metadata.subarray(tag+1,tag+4), tangent = metadata.subarray(tag+5,tag+8);
    const sample = {
      position:pose.rest, worldPosition:at,
      normal:Array.from(normal), tangent:tangent.some(Boolean) ? Array.from(tangent) : undefined,
      view:pose.localOffset(...view), lightDirection:K.V.norm(pose.localOffset(...material.sun)),
      shadeNormalFrame:pose, shadeNormalLight:material.sun,
      shadowBandBias:frame.shadowBandPaint === false ? 0 : world.renderMetadata?.inkControls?.[i*4],
      time, visibility:metadata[tag+4]
    };
    base[0] = (packed & 255)/255;
    base[1] = ((packed >>> 8) & 255)/255;
    base[2] = ((packed >>> 16) & 255)/255;
    surface.shade(material, sample, base, result);
    colors ??= data.slice();
    colors[q+7] = Math.round(result[0]*255)+256*Math.round(result[1]*255)+65536*Math.round(result[2]*255);
    if (patterned) {
      pixels ??= world.pix.slice();
      // Use the chosen diffuse palette in place of the baked pixel ramp.
      pixels[i*8] = 0;
    }
  }
  return colors ? { data:colors, pixels } : null;
}
