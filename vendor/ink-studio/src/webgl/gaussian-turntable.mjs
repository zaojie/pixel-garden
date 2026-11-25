import * as shared from '../png-bytes.js';
import { captureCamera } from './view.mjs';
import { createResources } from './resources.mjs';
import { createTargets } from './targets.mjs';
import { gaussianMaterialPlan } from './gaussian-materials.mjs';

const png = shared.default ?? globalThis[Symbol.for('ink-studio.png-bytes')];
export const gaussianCoverageFrame = Symbol('Gaussian turntable coverage');
export const gaussianNormalFrame = Symbol('Gaussian turntable normal owners');

function validate(world, options) {
  const { angles, width, height, columns, fps, cutoff, pivot, normals } = options;
  const integer = (n, lo, hi) => Number.isInteger(n) && n >= lo && n <= hi;
  if (!world?.gaussianHarmonics) throw new TypeError('A turntable needs a live imported Gaussian cloud');
  if (!Array.isArray(angles) || !angles.length || angles.length > 64 || ![...angles].every(Number.isFinite) ||
      !integer(width, 1, 512) || !integer(height, 1, 512) || !integer(columns, 1, 64) ||
      !Number.isFinite(fps) || fps < 1 || fps > 60 || !Number.isFinite(cutoff) || cutoff <= 0 || cutoff > 1 ||
      !Array.isArray(pivot) || pivot.length !== 3 || ![...pivot].every(Number.isFinite) || typeof normals !== 'boolean')
    throw new RangeError('A turntable needs 1–64 angles, valid tile dimensions, FPS, coverage cutoff and an XYZ pivot');
}

const bytes = color => [color >> 16, color >> 8 & 255, color & 255];

function nearest(colors, packed) {
  const [r, g, b] = bytes(packed);
  let nearest = 0, distance = Infinity;
  for (const [i, [cr, cg, cb]] of colors.entries()) {
    const d = .30 * (r - cr)**2 + .55 * (g - cg)**2 + .15 * (b - cb)**2;
    if (d < distance) { nearest = i; distance = d; }
  }
  return nearest;
}

function writeViewNormal(output, offset, normals, owner, { right, up, back }) {
  const at = owner * 3;
  if (at < 0 || !normals || at + 2 >= normals.length) return false;
  const x = normals[at], y = normals[at + 1], z = normals[at + 2], length = Math.hypot(x, y, z);
  if (!length) return false;
  const project = axis => axis[0] * x + axis[1] * y + axis[2] * z;
  // A covariance axis has an unsigned direction. Orient it toward this tile's camera.
  const facing = project(back), scale = (facing < 0 ? -.5 : .5) / length;
  output[offset] = Math.round(255 * (.5 + scale * project(right)));
  output[offset + 1] = Math.round(255 * (.5 + scale * project(up)));
  output[offset + 2] = Math.round(255 * (.5 + scale * facing));
  output[offset + 3] = 255;
  return true;
}

export function captureGaussianTurntable(gl, render, world, {
  angles = Array.from({ length: 8 }, (_, i) => i * 45), width = 128, height = width,
  columns = 4, fps = 8, cutoff = .25, pivot = world?.cam?.target ?? [0, 0, 0], frame = {}, normals = false
} = {}) {
  validate(world, { angles, width, height, columns, fps, cutoff, pivot, normals });
  const cols = Math.min(columns, angles.length);
  const sheetWidth = cols * width, sheetHeight = Math.ceil(angles.length / cols) * height;
  if (sheetWidth * sheetHeight > 16_000_000) throw new RangeError('Turntable sheet exceeds the pixel budget');
  const splatPaint = frame.splatPaint ?? world.splatPaint;
  const materials = gaussianMaterialPlan(world, frame.gaussianMaterials ?? world.gaussianMaterials);
  const packedColors = [...new Set([...world.pal, ...(splatPaint?.ramp ?? []), ...(materials?.palettes.flat() ?? [])])];
  if (!packedColors.length || packedColors.length > 255)
    throw new RangeError('A transparent turntable needs 1–255 opaque palette entries');
  const colors = packedColors.map(bytes), transparent = colors.length;
  const slots = new Map(packedColors.map((color, i) => [color, i]));
  const indices = new Uint8Array(sheetWidth * sheetHeight).fill(transparent);
  const rgba = new Uint8Array(width * height * 4), coverage = new Uint8Array(rgba.length);
  const normal = normals ? new Uint8Array(sheetWidth * sheetHeight * 4) : null;
  const owners = normals ? new Uint8Array(rgba.length) : null;
  const surfaceNormals = world.gaussianHarmonics.normals;
  const frames = [], tilePivot = { x: width / 2, y: height / 2 };
  const camera = { ...world.cam, ...frame.camera, pose: undefined, target: [...pivot], composition: [.5, .5], ortho: true };
  const resources = createResources(gl);
  try {
    const target = createTargets(gl, resources).target();
    target.resize(width, height);
    function read(framebuffer, buffer) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, framebuffer);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, buffer);
    }
    for (const [n, angle] of angles.entries()) {
      const rawAngle = (angle % 360) * Math.PI / 180;
      const cameraSnapshot = captureCamera(world, { camera: { ...camera, yaw: rawAngle },
        cameraStaging: false, cameraImpulses: false, stable: false }, width, height);
      let covered = false, owned = false;
      render(world, { ...frame, width, height, target, cameraSnapshot, gaussianColor: 'palette', colored: false,
        damp: false, dither: frame.dither ?? 0, changedRows: [], splatPaint: splatPaint ? { ...splatPaint, preview: false } : false,
        [gaussianCoverageFrame]: framebuffer => { read(framebuffer, coverage); covered = true; },
        ...(normals && { [gaussianNormalFrame]: { cutoff: Math.max(.035, cutoff * .3),
          read: framebuffer => { read(framebuffer, owners); owned = true; } } }) });
      if (!covered || normals && !owned) throw new Error('Turntable did not receive its Gaussian capture passes');
      read(target.framebuffer, rgba);
      if (gl.getError() !== gl.NO_ERROR) throw new Error('Could not read the rendered turntable frame');
      const x0 = n % cols * width, y0 = Math.floor(n / cols) * height;
      let pixels = 0, normalPixels = 0, touchesBorder = false;
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
        const src = ((height - 1 - y) * width + x) * 4;
        if (coverage[src + 3] < Math.ceil(cutoff * 255)) continue;
        const packed = rgba[src] << 16 | rgba[src + 1] << 8 | rgba[src + 2];
        if (!slots.has(packed)) slots.set(packed, nearest(colors, packed));
        const dest = (y0 + y) * sheetWidth + x0 + x;
        indices[dest] = slots.get(packed);
        if (normal && owners[src + 3]) {
          const owner = (owners[src] | owners[src + 1] << 8 | owners[src + 2] << 16) - 1;
          if (writeViewNormal(normal, dest * 4, surfaceNormals, owner, cameraSnapshot.pose)) normalPixels++;
        }
        pixels++;
        touchesBorder ||= x === 0 || y === 0 || x === width - 1 || y === height - 1;
      }
      frames.push({ angle, rect: { x: x0, y: y0, width, height }, pivot: { ...tilePivot },
        duration: 1 / fps, pixels, touchesBorder, ...(normals && { normalPixels }) });
    }
  } finally {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    resources.dispose();
  }
  return {
    width: sheetWidth, height: sheetHeight, indices, ...(normal && { normal }),
    metadata: { version: 1, width: sheetWidth, height: sheetHeight, fps, cutoff, transparent,
      palette: [...colors, [0, 0, 0]], projection: 'orthographic', worldPivot: [...pivot],
      camera: { pitchDegrees: (camera.pitch ?? 0) * 180 / Math.PI, distance: camera.dist }, frames,
      ...(normals && { normalSpace: 'view', encodings: { normal: 'XYZ mapped from [-1,1] to RGB, alpha marks a normal' } }) }
  };
}

export async function encodeGaussianTurntablePNG({ width, height, indices, metadata }) {
  const { palette, transparent } = metadata ?? {};
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1 ||
      !(indices instanceof Uint8Array) || !Array.isArray(palette) || palette.length > 256 ||
      !Number.isInteger(transparent) || transparent < 0 || transparent >= palette.length ||
      ![...palette].every(rgb => Array.isArray(rgb) && rgb.length === 3 &&
        [...rgb].every(value => Number.isInteger(value) && value >= 0 && value <= 255)) ||
      indices.some(index => index >= palette.length))
    throw new TypeError('Expected a valid indexed turntable sheet');
  const raw = png.scanlines(width, height, indices);
  const input = new Blob([raw]).stream().pipeThrough(new CompressionStream('deflate'));
  const compressed = new Uint8Array(await new Response(input).arrayBuffer());
  const colors = Uint8Array.from(palette.flat());
  const alpha = new Uint8Array(palette.length).fill(255);
  alpha[transparent] = 0;
  return new Blob(png.indexed(width, height, colors, alpha, compressed), { type: 'image/png' });
}

export async function encodeGaussianTurntableNormalPNG({ width, height, normal }) {
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1 ||
      !(normal instanceof Uint8Array) || normal.length !== width * height * 4)
    throw new TypeError('Expected turntable normal map pixels');
  const input = new Blob([png.scanlines(width, height, normal, 4)]).stream().pipeThrough(new CompressionStream('deflate'));
  const compressed = new Uint8Array(await new Response(input).arrayBuffer());
  return new Blob(png.rgba(width, height, compressed), { type: 'image/png' });
}
