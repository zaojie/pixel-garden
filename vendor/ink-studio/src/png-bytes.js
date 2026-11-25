/* PNG chunk framing shared by the Node writer and browser indexed export. */
const PNGBytes = (() => {
  const signature = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[i] = c >>> 0;
  }
  function chunk(type, data) {
    const bytes = new Uint8Array(data.length + 12), view = new DataView(bytes.buffer);
    view.setUint32(0, data.length);
    for (let i = 0; i < 4; i++) bytes[4 + i] = type.charCodeAt(i);
    bytes.set(data, 8);
    let crc = 0xffffffff;
    for (let i = 4; i < bytes.length - 4; i++) crc = crcTable[(crc ^ bytes[i]) & 255] ^ (crc >>> 8);
    view.setUint32(bytes.length - 4, (crc ^ 0xffffffff) >>> 0);
    return bytes;
  }
  function scanlines(width, height, pixels, channels = 1) {
    const stride = width * channels;
    if (pixels.length !== stride * height) throw new Error('PNG pixel count does not match its shape');
    const bytes = new Uint8Array((stride + 1) * height);
    for (let y = 0; y < height; y++)
      bytes.set(pixels.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
    return bytes;
  }
  function indexed(width, height, palette, alpha, deflated, ink) {
    const hd = new Uint8Array(13), view = new DataView(hd.buffer);
    view.setUint32(0, width);
    view.setUint32(4, height);
    hd[8] = 8;
    hd[9] = 3;
    const parts = [signature, chunk('IHDR', hd), chunk('PLTE', palette)];
    if (alpha) parts.push(chunk('tRNS', alpha));
    if (ink) parts.push(chunk('inKD',ink));
    parts.push(chunk('IDAT', deflated), chunk('IEND', new Uint8Array(0)));
    return parts;
  }
  function rgba(width, height, deflated, ink) {
    const hd = new Uint8Array(13), view = new DataView(hd.buffer);
    view.setUint32(0, width);
    view.setUint32(4, height);
    hd[8] = 8;
    hd[9] = 6;
    return [signature, chunk('IHDR', hd), ...(ink ? [chunk('inKD',ink)] : []),
      chunk('IDAT', deflated), chunk('IEND', new Uint8Array(0))];
  }
  return Object.freeze({ chunk, scanlines, indexed, rgba });
})();
if (typeof module !== 'undefined') module.exports = PNGBytes;
else globalThis[Symbol.for('ink-studio.png-bytes')] = PNGBytes;
