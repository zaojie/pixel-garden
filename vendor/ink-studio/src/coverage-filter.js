/* Gather four subpixel colors around the native art sample before palette lookup. */
const CoverageFilter = (() => {
  const record = value => value && Object.getPrototypeOf(value) === Object.prototype;
  function sharpness(value) {
    if (!Number.isFinite(value) || value < 0 || value > 1)
      throw new RangeError('Coverage filter sharpness must be between 0 and 1');
    return 1 - value;
  }

  function prepare(definition) {
    if (definition === undefined || definition === false) return null;
    const settings = definition === true ? {} : definition;
    if (!record(settings) || Object.keys(settings).some(key => key !== 'sharpness' && key !== 'materials'))
      throw new TypeError('Coverage filter needs a boolean or an object with sharpness and materials');
    const { sharpness: amount = 0, materials = {} } = settings;
    if (!record(materials) || Object.keys(materials).length > 256)
      throw new TypeError('Coverage filter needs a map of up to 256 material names');
    const values = new Map(Object.entries(materials).map(([label, value]) => {
      if (!label.trim()) throw new TypeError('Coverage filter material names must be nonempty');
      return [label, sharpness(value)];
    }));
    const fallback = sharpness(amount);
    if (!fallback && ![...values.values()].some(Boolean)) return null;
    return {
      fallback, varied: values.size > 0 || fallback !== 1,
      at: material => material == null ? fallback : values.get(String(material)) ?? fallback
    };
  }

  function filter(color, alpha, samples, width, height, matte) {
    const { color: sampleColor, alpha: sampleAlpha, w: sampleWidth, matte: sampleMatte } = samples;
    for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
      const p = y * width + x, top = (height - 1 - y) * width + x;
      const s = y * 2 * sampleWidth + x * 2, t = (height * 2 - 1 - y * 2) * sampleWidth + x * 2;
      const centerWeight = matte?.[p] ?? 1;
      if (!centerWeight) continue;
      const a = alpha ? alpha[top] / 255 : 1;
      let r = color[p * 3] * a, g = color[p * 3 + 1] * a, b = color[p * 3 + 2] * a;
      let outAlpha = a, replaced = 0;
      for (let i = 0; i < 4; i++) {
        const dx = i & 1, dy = i >> 1, q = s + dx + dy * sampleWidth, tap = t + dx - dy * sampleWidth;
        const weight = Math.min(centerWeight, sampleMatte?.[q] ?? 1) * .25;
        const tapAlpha = sampleAlpha ? sampleAlpha[tap] / 255 : 1;
        const gain = weight * tapAlpha / 255;
        r += sampleColor[q * 3] * gain;
        g += sampleColor[q * 3 + 1] * gain;
        b += sampleColor[q * 3 + 2] * gain;
        outAlpha += weight * (tapAlpha - a);
        replaced += weight;
      }
      const remove = a * replaced;
      const inv = outAlpha > 1e-6 ? 1 / outAlpha : 0;
      color[p * 3] = (r - color[p * 3] * remove) * inv;
      color[p * 3 + 1] = (g - color[p * 3 + 1] * remove) * inv;
      color[p * 3 + 2] = (b - color[p * 3 + 2] * remove) * inv;
      if (alpha) alpha[top] = Math.round(Math.min(1, Math.max(0, outAlpha)) * 255);
    }
  }

  return Object.freeze({ prepare, filter });
})();
if (typeof module !== 'undefined' && module.exports) module.exports = CoverageFilter;
else globalThis[Symbol.for('ink-studio.coverage-filter')] = CoverageFilter;
