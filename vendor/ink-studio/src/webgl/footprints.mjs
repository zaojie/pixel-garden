/* Retained row slots and shared silhouette SDF tiles; uploads only after an authored edit. */
export function createFootprintBinding(gl, api, resources, targets, maxSize) {
  const active = gl.getParameter(gl.ACTIVE_TEXTURE),
    bound = gl.getParameter(gl.TEXTURE_BINDING_2D);
  const rows = targets.table(1024, 2, maxSize),
    params = resources.texture(),
    atlas = resources.texture();
  gl.activeTexture(active);
  gl.bindTexture(gl.TEXTURE_2D, bound);
  let previous, previousCount, plan;
  function rebind() {
    rows.bind();
    targets.textureUnit(1, params);
    targets.textureUnit(6, atlas);
  }
  const bind = world => {
    if (previous !== world.splatFootprints || previousCount !== world.count) {
      const prior = plan;
      plan = api.plan(world);
      hasPlanes=plan.styles.some(p=>p.depth==='plane');
      if (plan.width > maxSize || plan.height > maxSize)
        throw new RangeError('Footprint atlas exceeds the texture limit');
      if (plan.indices !== prior?.indices) rows.upload(plan.indices);
      else rows.bind();
      targets.textureUnit(1, params);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 64, 2, 0, gl.RGBA, gl.FLOAT, plan.params);
      targets.textureUnit(6, atlas);
      if (plan.atlas !== prior?.atlas)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, plan.width, plan.height, 0, gl.RED, gl.FLOAT, plan.atlas);
      previous = world.splatFootprints;
      previousCount = world.count;
    } else rows.bind();
    targets.textureUnit(1, params);
    targets.textureUnit(6, atlas);
    return { u_footprintRows: 2, u_footprintParams: 1, u_footprintAtlas: 6, u_footprintColumns: plan.columns };
  };
  let hasPlanes=false;
  Object.defineProperty(bind,'hasPlanes',{get:()=>hasPlanes});
  bind.rebind = rebind;
  return bind;
}
