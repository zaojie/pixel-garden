/* Retained, row-aligned attachments share the existing geometry sampler. */
export function createAttachedData(stride = 8192) {
  let packed,
    baseRef,
    baseCount = 0,
    prior = [],
    priorOffsets = [];
  return function pack(base, count, attachments, dirtyRows) {
    if (
      !(base instanceof Float32Array) ||
      !Number.isSafeInteger(count) ||
      count < 0 ||
      count > base.length ||
      !Array.isArray(attachments) ||
      attachments.some(a => !(a instanceof Float32Array))
    )
      throw new TypeError('Attached data needs bounded Float32 tables');
    let rows = Math.ceil(count / stride);
    const offsets = attachments.map(a => {
      const at = rows;
      rows += Math.ceil(a.length / stride);
      return at;
    });
    const fresh = !packed || packed.length !== rows * stride || baseCount !== count;
    if (fresh) packed = new Float32Array(rows * stride);
    const dirty = new Set(dirtyRows ?? []),
      full = fresh || dirtyRows === undefined || baseRef !== base;
    if (full) packed.set(base.subarray(0, count));
    else
      for (const row of dirtyRows) {
        const a = row * stride,
          b = Math.min(count, a + stride);
        if (a < b) packed.set(base.subarray(a, b), a);
      }
    attachments.forEach((a, i) => {
      if (fresh || prior[i] !== a || priorOffsets[i] !== offsets[i]) {
        const at = offsets[i] * stride;
        packed.fill(0, at, at + Math.ceil(a.length / stride) * stride);
        packed.set(a, at);
        if (!full) for (let r = offsets[i]; r < offsets[i] + Math.ceil(a.length / stride); r++) dirty.add(r);
      }
    });
    baseRef = base;
    baseCount = count;
    prior = attachments.slice();
    priorOffsets = offsets;
    return { data: packed, offsets, changedRows: full ? undefined : [...dirty] };
  };
}
