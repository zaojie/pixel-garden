import { layerSamplingGLSL, depthLayerSamplingGLSL, createCompositeDepth, captureFrameCamera } from './frame-composite.mjs';
import { createRenderEffects } from './render-effects.mjs';
import { inspectionDepth } from './inspection.mjs';

export const graphCapture = Symbol('render graph raster capture');
export const graphOwner = Symbol('render graph owner capture');
export const graphEffectPixel = Symbol('render graph effect pixel');

const tileScheduleGLSL = `
bool tileDue(vec2 pixel, ivec3 tiles, ivec4 cycle) {
  if (cycle.w <= 0) return false;
  ivec2 tile = ivec2(pixel) / tiles.xy;
  int index = tile.y * tiles.z + tile.x, period = cycle.x;
  int offset = ((index % period) * cycle.y) % period;
  return (cycle.z + period - offset) % period < cycle.w;
}
`;

function tileUniforms(grid, width) {
  if (!grid) return { cycle: [1, 0, 0, 0], tiles: [1, 1, 1] };
  return {
    cycle: [grid.every, grid.step, (grid.head - grid.phase + grid.every) % grid.every, grid.span],
    tiles: [...grid.size, Math.ceil(width / grid.size[0])]
  };
}

const fragment = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_input, u_mask, u_owner, u_selected, u_depth, u_history, u_auxMask;
uniform highp sampler2D u_source;
uniform highp sampler2D u_rows, u_schedulePalette;
uniform bool u_scheduleIndexed, u_addressLinear;
uniform ivec2 u_addressEdges;
uniform vec4 u_windowCrop, u_windowDest, u_windowLine, u_windowInk;
uniform float u_windowBorder, u_windowLeader;
uniform bool u_windowLinear, u_windowHasSourceMask;
uniform highp sampler3D u_cube;
uniform vec2 u_size, u_passSize, u_offset, u_amplitude;
uniform bool u_packedDepth, u_depthMask, u_depthInvert, u_background, u_displayMask, u_budgetLinear;
uniform vec3 u_depthWindow;
uniform int u_operation;
uniform int u_effectProbe;
uniform vec2 u_effectPoint;
uniform bool u_hasMask, u_hasSelect, u_hasRect, u_invert, u_layer, u_linear, u_add, u_enabled;
uniform float u_opacity, u_feather, u_line, u_frequency, u_phase, u_cubeSize;
uniform bool u_reflectX;
uniform vec4 u_rect, u_layerDest;
uniform ivec4 u_layerRegion;
uniform bool u_layerTopDown;
uniform bool u_layerProjected;
uniform vec3 u_chartOrigin, u_chartU, u_chartV, u_chartNormal;
uniform vec4 u_chartBounds;
uniform float u_chartReach;
uniform vec3 u_volumeAt, u_extent, u_eye;
uniform vec2 u_depthRange, u_projectionScale;
uniform mat4 u_view;
uniform bool u_volume, u_ortho, u_captureHistory, u_hasHistory, u_hold, u_hasAuxMask, u_hasFlow;
uniform float u_volumeFeather, u_decay;
uniform vec2 u_feedbackPivot, u_feedbackTransform, u_threshold;
uniform float u_blockSize, u_gain;
uniform bool u_feedbackLinear, u_hasThreshold;
uniform bool u_scanlines, u_rejectDepth, u_hasDepth, u_isolated, u_over, u_wrap;
uniform bool u_rawMotion, u_motionPerFrame, u_needsRawMotion, u_blockMemory, u_hasBlockHold;
uniform sampler2D u_blockHold;
uniform vec4 u_blockRect;
uniform ivec4 u_blockCycle;
uniform ivec3 u_blockTiles;
uniform float u_depthThreshold;
uniform vec2 u_flowScale, u_flowVelocity, u_flowBlock;
uniform float u_smearElapsed;
out vec4 color;

${layerSamplingGLSL}
${depthLayerSamplingGLSL}
${tileScheduleGLSL}

float depthAt(vec2 pixel) {
  ivec2 size = textureSize(u_depth, 0);
  ivec2 site = min(ivec2(pixel / u_size * vec2(size)), size - 1);
  vec4 value=texelFetch(u_depth, ivec2(site.x, size.y - 1 - site.y), 0);
  return u_packedDepth ? (value.a<.5 ? 1.0 : dot(round(value.rgb*255.0),vec3(1.0,256.0,65536.0))/16777215.0) : value.r;
}

vec3 worldAt(vec2 pixel, float depth) {
  ivec2 size = textureSize(u_depth, 0);
  ivec2 site = min(ivec2(pixel / u_size * vec2(size)), size - 1);
  vec2 uv = (vec2(site) + .5) / vec2(size);
  vec2 plane = (uv * vec2(2.0, -2.0) + vec2(-1.0, 1.0)) * u_projectionScale;
  vec3 local = vec3(plane * (u_ortho ? 1.0 : depth), -depth);
  return u_eye + vec3(dot(u_view[0].xyz, local), dot(u_view[1].xyz, local), dot(u_view[2].xyz, local));
}

int projectedCellAt(vec2 pixel, out ivec2 cell) {
  if (!u_hasDepth) return 0;
  float normalized = depthAt(pixel);
  if (normalized >= 1.0) return 0;
  vec3 offset = worldAt(pixel, mix(u_depthRange.x, u_depthRange.y, normalized)) - u_chartOrigin;
  if (abs(dot(offset, u_chartNormal)) > u_chartReach) return 1;
  vec2 uv = (vec2(dot(offset, u_chartU), dot(offset, u_chartV)) - u_chartBounds.xy) /
    (u_chartBounds.zw - u_chartBounds.xy);
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) return 2;
  cell = min(ivec2(uv * vec2(u_layerRegion.zw)), u_layerRegion.zw - 1);
  return 3;
}

vec4 projectedTexel(ivec2 cell) {
  return layerTexel(u_source, u_layerRegion.xy + cell,
    textureSize(u_source, 0), u_layerTopDown);
}

vec4 projectedLayerAt(vec2 pixel) {
  ivec2 cell;
  return projectedCellAt(pixel, cell) == 3 ? projectedTexel(cell) : vec4(0.0);
}

int addressAxis(int n, int size, int edge) {
  if (edge == 0) return clamp(n, 0, size - 1);
  if (edge == 1) return n >= 0 && n < size ? n : -1;
  int span = edge == 3 ? size * 2 : size;
  int at = n - (n / span) * span;
  if (at < 0) at += span;
  return at < size ? at : span - 1 - at;
}

vec4 addressedColor(ivec2 site, ivec2 size) {
  int x = addressAxis(site.x, size.x, u_addressEdges.x);
  int y = addressAxis(site.y, size.y, u_addressEdges.y);
  return x < 0 || y < 0 ? vec4(0.0) : texelFetch(u_source, ivec2(x, size.y - 1 - y), 0);
}

vec4 textureAddress(vec2 uv) {
  ivec2 size = textureSize(u_source, 0);
  vec2 at = uv * vec2(size);
  if (!u_addressLinear) return addressedColor(ivec2(floor(at)), size);
  at -= .5;
  ivec2 p = ivec2(floor(at));
  vec2 f = fract(at);
  return mix(mix(addressedColor(p, size), addressedColor(p + ivec2(1, 0), size), f.x),
    mix(addressedColor(p + ivec2(0, 1), size), addressedColor(p + ivec2(1), size), f.x), f.y);
}

vec4 windowTap(ivec2 site, ivec2 size) {
  site = clamp(site, ivec2(0), size - 1);
  site.y = size.y - 1 - site.y;
  vec4 sampleColor = texelFetch(u_source, site, 0);
  return u_windowHasSourceMask ? sampleColor * texelFetch(u_auxMask, site, 0).r : sampleColor;
}

vec4 windowPixel(vec2 pixel) {
  vec2 local = pixel - u_windowDest.xy;
  if (all(greaterThanEqual(local, vec2(0.0))) && all(lessThan(local, u_windowDest.zw))) {
    vec2 edges = min(local, u_windowDest.zw - local);
    if (min(edges.x, edges.y) < u_windowBorder) return u_windowInk;
    ivec2 size = textureSize(u_source, 0);
    if (!u_windowLinear) {
      if (any(lessThanEqual(u_windowCrop.xy + u_windowCrop.zw, vec2(0.0))) ||
          any(greaterThanEqual(u_windowCrop.xy, vec2(size)))) return vec4(0.0);
      ivec2 donor = ivec2(u_windowCrop.xy) + (2 * ivec2(local) + 1) * ivec2(u_windowCrop.zw) / (2 * ivec2(u_windowDest.zw));
      return any(lessThan(donor, ivec2(0))) || any(greaterThanEqual(donor, size)) ? vec4(0.0) : windowTap(donor, size);
    }
    vec2 at = u_windowCrop.xy + local * u_windowCrop.zw / u_windowDest.zw;
    if (any(lessThan(at, vec2(0.0))) || any(greaterThanEqual(at, vec2(size)))) return vec4(0.0);
    at -= .5;
    ivec2 p = ivec2(floor(at));
    vec2 f = fract(at);
    return mix(mix(windowTap(p, size), windowTap(p + ivec2(1, 0), size), f.x),
      mix(windowTap(p + ivec2(0, 1), size), windowTap(p + ivec2(1), size), f.x), f.y);
  }
  if (u_windowLeader > 0.0) {
    vec2 line = u_windowLine.zw - u_windowLine.xy;
    vec2 from = pixel - u_windowLine.xy;
    vec2 delta = from - line * clamp(dot(from, line) / dot(line, line), 0.0, 1.0);
    if (dot(delta, delta) <= u_windowLeader * u_windowLeader * .25) return u_windowInk;
  }
  return vec4(0.0);
}

void main() {
  ivec2 pos = ivec2(gl_FragCoord.xy);
  vec2 pixel = vec2(gl_FragCoord.x, u_passSize.y - gl_FragCoord.y) * u_size / u_passSize;
  if (u_effectProbe != 0) pixel = u_effectPoint;
  if (u_effectProbe == 1) { color = layerAt(u_owner, pixel, u_size, false, false); return; }
  vec4 inputColor = layerAt(u_input, pixel, u_size, false, false);
  if (u_operation == 0) { color = inputColor; return; }
  if (u_isolated) inputColor = vec4(0.0);
  float weight = u_hasMask ? layerAt(u_mask, pixel, u_size, false, false).r : 1.0;
  float chosen = 1.0;
  if (u_hasSelect) {
    vec3 encoded = layerAt(u_owner, pixel, u_size, false, false).rgb;
    ivec3 bytes = ivec3(floor(encoded * 255.0 + 0.5));
    int id = bytes.r + bytes.g * 256 + bytes.b * 65536;
    int rowWidth = textureSize(u_selected, 0).x;
    chosen = texelFetch(u_selected, ivec2(id % rowWidth, id / rowWidth), 0).r;
    weight *= chosen;
  }
  if (u_depthMask) {
    float n = u_hasDepth ? depthAt(pixel) : 1.0;
    float z = mix(u_depthRange.x, u_depthRange.y, n);
    float edge = min(z - u_depthWindow.x, u_depthWindow.y - z);
    float a = u_depthWindow.z > 0.0 ? clamp(edge / u_depthWindow.z, 0.0, 1.0) : step(0.0, edge);
    weight *= n >= 1.0 ? (u_background ? 1.0 : 0.0) : (u_depthInvert ? 1.0 - a : a);
  }
  if (u_effectProbe == 2) {
    ivec2 cell = ivec2(0);
    int status = u_layerProjected ? projectedCellAt(pixel, cell) : 0;
    float alpha = u_layerProjected ? (status == 3 ? projectedTexel(cell).a : 0.0) :
      guidedLayerAt(u_source, pixel, u_size, u_layerDest, u_layerRegion, u_layerTopDown, u_linear).a;
    if (!u_layerProjected) cell.x = int(round(weight * u_opacity * 255.0));
    color = vec4(vec2(cell), float(status + (chosen > 0.0 ? 4 : 0)), alpha * weight * u_opacity * 255.0) / 255.0;
    return;
  }
  if (u_operation == 1) {
    if (u_layer) weight *= guidedLayerAt(u_source, pixel, u_size, u_layerDest, u_layerRegion, u_layerTopDown, u_linear).a;
    if (u_volume) {
      float norm = depthAt(pixel);
      if (norm >= 1.0) weight = 0.0;
      else {
        float depth = mix(u_depthRange.x, u_depthRange.y, norm);
        vec3 position = worldAt(pixel, depth);
        float d = length((position - u_volumeAt) / u_extent);
        weight *= u_volumeFeather > 0.0 ? clamp((1.0 - d) / u_volumeFeather, 0.0, 1.0) : step(d, 1.0);
      }
    }
    if (u_hasRect) {
      vec2 edges = min(pixel - u_rect.xy, u_rect.xy + u_rect.zw - pixel);
      float edge = min(edges.x, edges.y);
      weight *= u_feather > 0.0 ? clamp(edge / u_feather, 0.0, 1.0) : step(0.0, edge);
    }
    color = vec4(u_enabled ? (u_invert ? 1.0 - weight : weight) * u_opacity : 0.0);
    return;
  }
  if (u_operation == 17) {
    ivec2 site = ivec2(pixel), size = ivec2(u_size);
    ivec4 row = ivec4(texelFetch(u_rows, ivec2(0, site.y), 0));
    ivec2 state = ivec2(texelFetch(u_rows, ivec2(1, site.y), 0).xy);
    ivec2 local = site - ivec2(0, state.x);
    ivec2 donor = (local / row.zw) * row.zw + ivec2(0, state.x) + row.xy;
    if (u_wrap) donor = ivec2(mod(vec2(donor), u_size));
    vec4 scheduled = vec4(0.0);
    if (all(greaterThanEqual(donor, ivec2(0))) && all(lessThan(donor, size))) {
      ivec2 extent = textureSize(u_source, 0);
      ivec2 at = ((2 * donor + 1) * extent) / (2 * size);
      if (!u_scheduleIndexed) at.y = extent.y - 1 - at.y;
      scheduled = texelFetch(u_source, at, 0);
      if (u_scheduleIndexed) {
        int slot = int(floor(scheduled.r * 255.0 + 0.5));
        int coverage = int(floor(scheduled.a * 255.0 + 0.5));
        ivec4 swatch = ivec4(floor(texelFetch(u_schedulePalette, ivec2(slot, state.y), 0) * 255.0 + 0.5));
        int alpha = (coverage * swatch.a + 127) / 255;
        scheduled = vec4(vec3((swatch.rgb * alpha + 127) / 255), float(alpha)) / 255.0;
      }
    }
    weight *= u_opacity;
    if (u_over) {
      scheduled *= weight;
      color = scheduled + inputColor * (1.0 - scheduled.a);
    } else color = mix(inputColor, scheduled, weight);
    color = floor(color * 255.0 + .50001) / 255.0;
    return;
  }
  if (u_operation == 18) {
    vec3 uv = texelFetch(u_rows, ivec2(pixel), 0).rgb;
    vec4 addressed = uv.z > 0.0 ? textureAddress(uv.xy) * uv.z : vec4(0.0);
    weight *= u_opacity;
    if (u_over) {
      addressed *= weight;
      color = addressed + inputColor * (1.0 - addressed.a);
    } else color = mix(inputColor, addressed, weight);
    color = floor(color * 255.0 + .50001) / 255.0;
    return;
  }
  if (u_operation == 6) {
    vec4 previous = u_hasHistory ? layerAt(u_history, pixel, u_size, false, u_budgetLinear) : vec4(0.0);
    if (u_captureHistory) {
      vec4 fresh = inputColor * weight;
      color = u_hold && previous.a > 0.0 ? previous :
        min(vec4(1.0), fresh + floor(previous * u_decay * (1.0 - fresh.a) * 255.0 + .00001) / 255.0);
    } else {
      vec4 sampleColor = previous * (chosen * u_opacity);
      color = sampleColor + inputColor * (1.0 - sampleColor.a);
    }
    return;
  }
  if (u_operation == 7) {
    if (u_captureHistory) {
      vec4 previous = vec4(0.0);
      if (u_hasHistory) {
        vec2 local = pixel - u_feedbackPivot;
        float c = u_feedbackTransform.x, s = u_feedbackTransform.y;
        vec2 field = u_blockSize > 0.0 ? (floor(pixel / u_blockSize) + .5) * u_blockSize : pixel;
        vec2 sampleAt = u_feedbackPivot + vec2(c * local.x + s * local.y, -s * local.x + c * local.y)
          - u_offset - u_amplitude * sin(field.yx * u_frequency + u_phase);
        if (all(greaterThanEqual(sampleAt, vec2(0.0))) && all(lessThan(sampleAt, u_size)))
          previous = layerAt(u_history, sampleAt, u_size, false, u_feedbackLinear);
      }
      vec4 fresh = u_layer ? guidedLayerAt(u_source, pixel, u_size, u_layerDest, u_layerRegion, u_layerTopDown, u_linear)
        : layerAt(u_source, pixel, u_size, false, false);
      float gain = weight * u_gain;
      if (u_hasThreshold && fresh.a > 0.0) {
        float amount = max(fresh.r, max(fresh.g, fresh.b)) / fresh.a;
        gain *= clamp((amount - u_threshold.x) / (u_threshold.y - u_threshold.x), 0.0, 1.0);
      }
      fresh *= gain;
      color = min(vec4(1.0), fresh + floor(previous * u_decay * (1.0 - fresh.a) * 255.0 + .00001) / 255.0);
    } else {
      vec4 signal = layerAt(u_history, pixel, u_size, false, u_budgetLinear) * u_opacity * (u_displayMask ? weight : 1.0);
      color.a = signal.a + inputColor.a * (1.0 - signal.a);
      color.rgb = min(vec3(color.a), signal.rgb + inputColor.rgb * (u_add ? 1.0 : 1.0 - signal.a));
    }
    return;
  }
  if (u_operation == 14) {
    if (u_captureHistory) {
      color = inputColor * chosen;
    } else {
      vec4 donor = layerAt(u_history, pixel, u_size, false, u_budgetLinear);
      if (u_hasAuxMask) weight *= layerAt(u_auxMask, pixel, u_size, false, false).r;
      color = u_hasSelect && donor.a <= 0.0 ? inputColor : mix(inputColor, donor, weight * u_opacity);
    }
    return;
  }
  if (u_operation == 15) {
    color = inputColor;
    if (!u_hasHistory) return;
    vec2 origin = u_blockMemory ? u_blockRect.xy : vec2(0.0);
    vec2 edge = u_blockMemory ? u_blockRect.zw : u_size;
    if (any(lessThan(pixel, origin)) || any(greaterThanEqual(pixel, edge))) return;
    if (u_blockMemory && tileDue(pixel - origin, u_blockTiles, u_blockCycle)) return;
    vec2 block = origin + floor((pixel - origin) / u_flowBlock) * u_flowBlock;
    vec2 center = block + min(u_flowBlock, edge - block) * .5;
    vec2 maskPixel = u_blockMemory ? center : pixel;
    float keep = u_hasAuxMask ? 1.0 - layerAt(u_auxMask, maskPixel, u_size, false, false).r : 1.0;
    if (keep <= 0.0) return;
    bool held = u_blockMemory && u_hasBlockHold && layerAt(u_blockHold, center, u_size, false, false).r > .5;
    float amount = keep;
    vec2 donor = pixel;
    if (!held) {
      if (u_decay <= 0.0 || (u_needsRawMotion && !u_hasFlow)) return;
      vec2 offset = u_flowVelocity * u_smearElapsed;
      amount *= u_decay;
      if (u_hasFlow) {
        vec4 flow = layerAt(u_source, center, u_size, u_rawMotion, false);
        if (flow.a <= 0.0 || flow.b <= 0.0) return;
        if (u_rawMotion) {
          offset += flow.rg * u_flowScale * (u_motionPerFrame ? 1.0 : u_smearElapsed);
        } else {
          vec2 bytes = floor(flow.rg / flow.a * 255.0 + .5);
          vec2 velocity = u_flowVelocity + max(vec2(-1.0), (bytes - 128.0) / 127.0) * u_flowScale;
          offset = velocity * u_smearElapsed;
        }
        amount *= flow.b;
      }
      donor -= offset;
      if (any(lessThan(donor, origin)) || any(greaterThanEqual(donor, edge))) return;
    }
    vec2 site = u_captureHistory ? donor : pixel;
    if (u_blockMemory) {
      vec2 scale = vec2(textureSize(u_history, 0)) / u_size;
      vec2 first = ceil(origin * scale - .5) + .5;
      vec2 last = ceil(edge * scale - .5) - .5;
      if (any(greaterThan(first, last))) return;
      site = clamp(site * scale, first, last) / scale;
    }
    vec4 old = layerAt(u_history, site, u_size, false, u_budgetLinear);
    if (u_captureHistory) {
      vec4 delta = (old - inputColor) * (amount * 255.0);
      color = inputColor + sign(delta) * floor(abs(delta) + .00001) / 255.0;
    } else color = floor(mix(inputColor, old, weight * u_opacity) * 255.0 + .50001) / 255.0;
    return;
  }
  if (u_operation == 16) {
    vec4 bank = layerAt(u_history, pixel, u_size, false, u_budgetLinear);
    color = floor(mix(inputColor, bank, weight * u_opacity) * 255.0 + .50001) / 255.0;
    return;
  }
  weight *= u_opacity;
  color = inputColor;
  if (weight <= 0.0) return;
  if (u_operation == 19) {
    vec2 center = vec2(pos.x, int(u_passSize.y) - 1 - pos.y) + .5;
    vec4 overlay = windowPixel(center) * weight;
    color = floor((overlay + inputColor * (1.0 - overlay.a)) * 255.0 + .50001) / 255.0;
  } else if (u_operation == 2) {
    if (inputColor.a <= 0.0) return;
    vec3 lookup = (clamp(inputColor.rgb / inputColor.a, 0.0, 1.0) * (u_cubeSize - 1.0) + 0.5) / u_cubeSize;
    color.rgb = mix(inputColor.rgb, texture(u_cube, lookup).rgb * inputColor.a, weight);
  } else if (u_operation == 5) {
    vec4 over = u_layerProjected ? projectedLayerAt(pixel) :
      u_layer ? guidedLayerAt(u_source, pixel, u_size, u_layerDest, u_layerRegion, u_layerTopDown, u_linear) :
        layerAt(u_source, pixel, u_size, false, false);
    over *= weight;
    color.a = over.a + inputColor.a * (1.0 - over.a);
    color.rgb = min(vec3(color.a), over.rgb + inputColor.rgb * (u_add ? 1.0 : 1.0 - over.a));
  } else {
    vec2 sourcePixel = pixel;
    vec3 tint = vec3(0.0);
    if (u_operation == 3) {
      if (u_reflectX) sourcePixel.x = 2.0 * u_line - sourcePixel.x;
      else sourcePixel.y = 2.0 * u_line - sourcePixel.y;
    } else {
      if (u_scanlines) {
        vec4 transform = texelFetch(u_rows, ivec2(0, int(pixel.y)), 0);
        vec4 signal = texelFetch(u_rows, ivec2(1, int(pixel.y)), 0);
        if (signal.a < 0.0) {
          if (!u_over) color = inputColor * (1.0 - weight);
          return;
        }
        if (signal.a > 0.0) sourcePixel.y = signal.a - 0.5;
        vec2 pivot = u_size * 0.5;
        sourcePixel = (sourcePixel - pivot) * transform.zw + pivot + transform.xy;
        tint = signal.rgb / 255.0;
      }
      sourcePixel += u_offset + u_amplitude * sin(pixel.yx * u_frequency + u_phase);
      if (u_wrap) sourcePixel = mod(sourcePixel, u_size);
    }
    if (!(all(greaterThanEqual(sourcePixel, vec2(0.0))) && all(lessThan(sourcePixel, u_size)))) return;
    if (u_rejectDepth) {
      if (!u_hasDepth) return;
      float a = depthAt(pixel), b = depthAt(floor(sourcePixel) + 0.5);
      bool aSky = a >= 1.0, bSky = b >= 1.0;
      if (aSky != bSky || (!aSky && abs(a - b) * (u_depthRange.y - u_depthRange.x) > u_depthThreshold)) return;
    }
    vec4 sampled = layerAt(u_source, sourcePixel, u_size, false, false);
    sampled.rgb = clamp(sampled.rgb + tint * sampled.a, vec3(0.0), vec3(sampled.a));
    if (u_over) {
      sampled *= weight;
      color = sampled + inputColor * (1.0 - sampled.a);
    } else color = mix(inputColor, sampled, weight);
  }
}`;

const bankFragment = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D u_current, u_previous, u_pulse, u_masks[8];
uniform vec2 u_size, u_work;
uniform bool u_first, u_hasPulse;
uniform int u_rules, u_masked[8];
uniform vec4 u_rects[8];
uniform ivec4 u_cycle;
uniform ivec3 u_tiles;
out vec4 color;
${layerSamplingGLSL}
${tileScheduleGLSL}

bool inside(vec2 p, vec4 rect) {
  return all(greaterThanEqual(p, rect.xy)) && all(lessThan(p, rect.xy + rect.zw));
}

void main() {
  vec2 pixel = vec2(gl_FragCoord.x, u_work.y - gl_FragCoord.y) * u_size / u_work;
  vec4 fresh = layerAt(u_current, pixel, u_size, false, false);
  if (u_first) { color = fresh; return; }
  float weight = u_hasPulse ? layerAt(u_pulse, pixel, u_size, false, false).r : 0.0;
  if (tileDue(pixel, u_tiles, u_cycle)) weight = 1.0;
  ${Array.from({ length: 8 }, (_, i) => `
  if (u_rules > ${i} && weight < 1.0 && inside(pixel, u_rects[${i}]))
    weight = max(weight, u_masked[${i}] != 0 ? layerAt(u_masks[${i}], pixel, u_size, false, false).r : 1.0);`).join('')}
  vec4 old = layerAt(u_previous, pixel, u_size, false, false);
  color = floor(mix(old, fresh, weight) * 255.0 + .50001) / 255.0;
}`;

export function createRenderGraph(gl, { api, program, resources, target, textureUnit, fullscreen, maxSize, drawLighting }) {
  const shader = program(fragment),
    scene = target();
  const selected = resources.texture(),
    raster = resources.texture(),
    rows = resources.texture(),
    schedulePalette = resources.texture();
  const cube = resources.allocate('Texture');
  const pool = [],
    historyFree = [];
  const groupsTexture = resources.texture();
  const layerDepth = createCompositeDepth(gl, resources, textureUnit, 6, 11);
  const effects = createRenderEffects(gl, { program, target, textureUnit, fullscreen });
  const history = api.history({ release: buffer => historyFree.push(buffer), releaseAge: buffer => buffer.dispose() });
  let lastCube, ageScratch, bankShader, probeTarget;
  gl.activeTexture(gl.TEXTURE4);
  gl.bindTexture(gl.TEXTURE_3D, cube);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  for (const axis of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R])
    gl.texParameteri(gl.TEXTURE_3D, axis, gl.CLAMP_TO_EDGE);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.RGBA8, 1, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(4));
  gl.activeTexture(gl.TEXTURE0);

  function capture(plan, outputs, width, height) {
    const pixels = width * height, bottomUp = new Uint8Array(pixels * 4);
    return Object.fromEntries([...plan.captures].map(([id, kind]) => {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, outputs.get(id).framebuffer);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, bottomUp);
      const channels = kind === 'mask' ? 1 : 4, data = new Uint8Array(pixels * channels);
      for (let y = 0, p = 0; y < height; y++) for (let x = 0; x < width; x++, p++) {
        const from = ((height - 1 - y) * width + x) * 4;
        if (kind === 'mask') data[p] = bottomUp[from];
        else {
          const alpha = bottomUp[from + 3];
          data[p * 4 + 3] = alpha;
          if (alpha) for (let c = 0; c < 3; c++)
            data[p * 4 + c] = Math.min(255, Math.round(255 * bottomUp[from + c] / alpha));
        }
      }
      return [id, { kind, width, height, channels, data }];
    }));
  }

  function refreshBank(bank, source, outputs, { width, height }, w, h) {
    bankShader ??= program(bankFragment);
    gl.activeTexture(gl.TEXTURE0);
    const stamp = historyFree.pop() ?? target();
    stamp.resize(width, height);
    textureUnit(0, source.textures[0]);
    textureUnit(1, (bank.value ?? source).textures[0]);
    textureUnit(10, (outputs.get(bank.pulse) ?? source).textures[0]);
    const units = new Int32Array(8), masked = new Int32Array(8), rects = new Float32Array(8 * 4);
    for (let i = 0; i < units.length; i++) {
      const rule = bank.regions[i], mask = outputs.get(rule?.mask);
      units[i] = i + 2;
      textureUnit(units[i], (mask ?? source).textures[0]);
      masked[i] = Number(Boolean(mask));
      rects.set(rule?.rect ?? [0, 0, w, h], i * 4);
    }
    const { cycle, tiles } = tileUniforms(bank.grid, w);
    fullscreen(bankShader, stamp.framebuffer, width, height, {
      u_current: 0, u_previous: 1, u_pulse: 10, u_masks: units, u_size: [w, h], u_work: [width, height],
      u_first: bank.first, u_hasPulse: Boolean(bank.pulse), u_rules: bank.regions.length, u_masked: masked, u_rects: rects,
      u_cycle: cycle, u_tiles: tiles
    });
    return stamp;
  }

  function execute(world, frame, plan, w, h, destination, present) {
    if (plan.captures.size * w * h > 4194304) throw new RangeError('WebGL render graph capture exceeds its readback budget');
    const layout = api.resources?.(plan, w, h);
    const time = frame.interfaceTime ?? frame.time ?? world.lookSnapshot?.time ?? 0;
    const at = frame.effectPixel;
    const probe = at && api.effectProbe(world, at, w, h, time, 'webgl');
    const probeBytes = probe && new Uint8Array(4);
    const effectPoint = probe && at.map(n => n + .5);
    if (probe) {
      probeTarget ??= target();
      probeTarget.resize(1, 1);
    }
    function readPixel(buffer, x, y) {
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, buffer);
      gl.readBuffer(gl.COLOR_ATTACHMENT0);
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, probeBytes);
      return Array.from(probeBytes);
    }
    scene.resize(w, h);
    let ownership, camera, depth, depthRange;
    present(
      world,
      {
        ...frame,
        ...(plan.needsCamera &&
          captureFrameCamera(frame, view => {
            camera = view;
          })),
        ...(plan.needsDepth && {
          [inspectionDepth](scene, texture, view) {
            depth = texture;
            depthRange = view.depthRange;
            frame[inspectionDepth]?.(scene, texture, view);
          }
        }),
        [graphOwner]: plan.needsOwner
          ? Object.assign(
              (world, texture, ownerDepth, view) => {
                ownership = { world, texture, depth: ownerDepth, view };
              },
              { depth: plan.needsDepth }
            )
          : undefined
      },
      w,
      h,
      scene.framebuffer
    );
    if (ownership?.depth) {
      depth = ownership.depth;
      depthRange = ownership.view.depthRange;
    }
    if (!destination) {
      if (gl.canvas.width !== w) gl.canvas.width = w;
      if (gl.canvas.height !== h) gl.canvas.height = h;
    }
    for (const test of [gl.BLEND, gl.DEPTH_TEST, gl.CULL_FACE, gl.SCISSOR_TEST, gl.STENCIL_TEST]) gl.disable(test);
    gl.colorMask(true, true, true, true);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_3D, cube);
    const outputs = new Map([['scene', scene]]),
      free = pool.slice(),
      selections = new Map();
    if (probe && ownership) {
      textureUnit(3, ownership.texture);
      fullscreen(shader, probeTarget.framebuffer, 1, 1, {
        u_effectProbe: 1, u_effectPoint: effectPoint, u_owner: 3, u_cube: 4, u_size: [w,h], u_passSize: [1,1]
      });
      const [r,g,b] = readPixel(probeTarget.framebuffer, 0, 0);
      probe.setReceiver(ownership.world, r + 256 * g + 65536 * b);
    }
    const openHistory = history.begin(plan, {
      source: frame.renderHistoryKey ?? world,
      w,
      h,
      time,
      camera,
      reset: frame.resetRenderHistory
    });
    let hasGroups = false;
    if (plan.passes.some(n => n.enabled && n.effect) && ownership) {
      const ids = api.ownerGroups(ownership.world),
        width = Math.min(2048, maxSize),
        height = Math.ceil(ids.length / width),
        data = new Uint8Array(width * height * 4);
      for (let i = 0; i < ids.length; i++) {
        data[i * 4] = ids[i] & 255;
        data[i * 4 + 1] = (ids[i] >>> 8) & 255;
        data[i * 4 + 2] = (ids[i] >>> 16) & 255;
        data[i * 4 + 3] = ids[i] >>> 24;
      }
      textureUnit(9, groupsTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
      hasGroups = true;
    }
    let lastSelect;
    for (const [index, node] of plan.passes.entries()) {
      const input = outputs.get(node.input) ?? scene;
      const effect = probe?.effect(node);
      if (effect && !node.enabled) probe.note(node, effect);
      if ((!node.enabled || (!ownership && (node.type === 'history' || (node.type === 'frameAge' || node.bank) && node.select))) && node.type !== 'mask')
        outputs.set(node.id, input);
      else {
        const out = free.pop() ?? target();
        if (!pool.includes(out)) pool.push(out);
        out.resize(w, h);
        const feedback = node.type === 'feedback';
        const smear = node.type === 'motionSmear';
        const memory = feedback || smear || node.type === 'history' ? openHistory(node) : null;
        const rawMotion = node.motion && api.motionRaster(node, frame.renderGraphInputs, w, h);
        const mask = outputs.get(node.mask),
          table = node.type === 'grade' && api.cube(node);
        const layer =
          node.enabled && node.layer && (!feedback || memory.capture) &&
          api.layer(node, w, h, world, time, camera, gl, frame.time, frame.renderGraphInputs, drawLighting);
        const window = node.window && node.opacity && api.windowFrame(node.window, {
          world, time, sceneTime: frame.time, width: w, height: h, camera
        });
        const sourceMask = window && outputs.get(node.window.sourceMask);
        const scanlines = api.rowTable(node.rows, w, h, time, node.rowEdits);
        const schedule = node.schedule && api.scheduleTable(node.schedule, w, h, time);
        const indexed = node.indices !== undefined && api.indexedSchedule(node, frame.renderGraphInputs, schedule);
        const select = node.enabled && ownership && node.select && JSON.stringify(node.select);
        const opacity = node.opacity * (node.layer && !feedback ? (layer?.opacity ?? 0) : 1);
        if (layer?.gpu) textureUnit(1, layer.gpu.texture);
        else if (layer) {
          if (layer.w > maxSize || layer.h > maxSize)
            throw new RangeError('Render graph layer exceeds the WebGL texture size limit');
          textureUnit(1, raster);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, layer.w, layer.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, layer.data);
        } else if (indexed) {
          if (indexed.w > maxSize || indexed.h > maxSize)
            throw new RangeError('Raster schedule indices exceed the WebGL texture size limit');
          textureUnit(1, raster);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, indexed.w, indexed.h, 0, gl.RGBA, gl.UNSIGNED_BYTE, indexed.data);
          textureUnit(10, schedulePalette);
          const slots = schedule.paletteSize;
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, slots, schedule.paletteData.length / (slots * 4),
            0, gl.RGBA, gl.UNSIGNED_BYTE, schedule.paletteData);
        } else if (rawMotion) {
          textureUnit(1, raster);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, rawMotion);
        } else textureUnit(1, (outputs.get(window?.source ?? node.source ?? node.field) ?? scene).textures[0]);
        if (table && table !== lastCube) {
          gl.activeTexture(gl.TEXTURE4);
          gl.texImage3D(
            gl.TEXTURE_3D,
            0,
            gl.RGBA8,
            table.size,
            table.size,
            table.size,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            table.data
          );
          lastCube = table;
        }
        if (select && select !== lastSelect) {
          if (!selections.has(select)) {
            const rows = api.selection(ownership.world, node.select),
              width = Math.min(2048, maxSize);
            const height = Math.ceil(rows.length / width),
              data = new Uint8Array(width * height);
            data.set(rows);
            selections.set(select, { width, height, data });
          }
          const { width, height, data } = selections.get(select);
          textureUnit(5, selected);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, data);
          lastSelect = select;
        }
        textureUnit(0, input.textures[0]);
        textureUnit(2, (mask ?? scene).textures[0]);
        textureUnit(3, ownership?.texture ?? scene.textures[0]);
        textureUnit(5, select ? selected : scene.textures[0]);
        textureUnit(6, depth ?? scene.textures[0]);
        const addresses = node.address && api.addressRaster(node, frame.renderGraphInputs, w, h);
        const registers = addresses || (schedule?.registers ?? scanlines);
        textureUnit(8, registers ? rows : (sourceMask ?? scene).textures[0]);
        if (registers) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, addresses ? w : 2, h, 0, gl.RGBA, gl.FLOAT, registers);
        const values = {
          ...layerDepth(layer, ownership?.depth && {texture:depth, range:depthRange}, depth ?? scene.textures[0]),
          u_input: 0,
          u_source: 1,
          u_mask: 2,
          u_owner: 3,
          u_cube: 4,
          u_selected: 5,
          u_size: [w, h],
          u_passSize: [w, h],
          u_effectProbe: 0,
          u_operation: node.window && !window ? 0 : node.operation,
          ...(window && {
            u_windowCrop: window.crop,
            u_windowDest: window.dest,
            u_windowLine: window.line,
            u_windowInk: window.ink?.rgba.map(n => n / 255) ?? [0, 0, 0, 0],
            u_windowBorder: window.ink?.border ?? 0,
            u_windowLeader: window.leader,
            u_windowLinear: window.linear,
            u_windowHasSourceMask: Boolean(sourceMask),
            u_auxMask: 8
          }),
          u_opacity: opacity,
          u_enabled: node.enabled,
          u_hasMask: Boolean(mask),
          u_hasSelect: Boolean(select),
          u_hasGroups: hasGroups,
          u_packedDepth: Boolean(ownership?.depth),
          u_depthMask: Boolean(node.depth),
          u_depthInvert: node.depth?.invert ?? false,
          u_background: node.depth?.background ?? false,
          u_depthWindow: node.depth ? [node.depth.near, node.depth.far, node.depth.feather] : [0, 1, 0],
          u_displayMask: node.displayMask ?? false,
          u_budgetLinear: node.budget?.filter === 'linear',
          u_hasRect: Boolean(node.rect),
          ...(node.rect && { u_rect: node.rect, u_feather: node.feather }),
          u_invert: node.invert ?? false,
          u_cubeSize: table?.size ?? 1,
          u_depth: 6,
          u_history: 7,
          u_blockHold: 0,
          u_volume: Boolean(node.volume && depth && camera),
          u_layerProjected: Boolean(node.projection && layer),
          u_rows: 8,
          u_scanlines: Boolean(scanlines),
          u_isolated: node.isolated ?? false,
          u_wrap: node.schedule?.wrap ?? node.wrap ?? false,
          u_over: node.blend === 'over',
          u_addressLinear: node.address?.filter === 'linear',
          u_addressEdges: node.address?.edges ?? [0, 0],
          u_scheduleIndexed: Boolean(indexed),
          u_schedulePalette: indexed ? 10 : 0,
          u_rejectDepth: node.depthThreshold !== undefined,
          u_hasDepth: Boolean(depth && (!node.projection || camera)),
          u_depthThreshold: node.depthThreshold ?? 0,
          u_depthRange: depthRange ?? [0, 1],
          ...((node.volume || node.projection) &&
            depth &&
            camera && {
              u_eye: camera.eye,
              u_view: camera.view,
              u_projectionScale: [camera.tangent * camera.aspect, camera.tangent],
              u_ortho: camera.ortho
            }),
          ...(node.volume && { u_volumeAt: node.volume.at, u_extent: node.volume.extent, u_volumeFeather: node.volume.feather }),
          ...(node.projection && {
            u_chartOrigin: node.projection.origin, u_chartU: node.projection.u, u_chartV: node.projection.v,
            u_chartNormal: node.projection.normal, u_chartBounds: node.projection.bounds, u_chartReach: node.projection.reach
          }),
          u_layer: Boolean(layer),
          ...(layer && {
            u_layerDest: layer.dest,
            u_layerTopDown: !layer.gpu,
            u_layerRegion: layer.gpu?.region ?? [0, 0, layer.w, layer.h]
          }),
          u_linear: layer?.filter === 'linear',
          u_add: (feedback ? node.blend : (layer?.blend ?? node.blend)) === 'add',
          u_reflectX: node.axis === 'x',
          u_line: node.line ?? (node.axis === 'x' ? w : h) * 0.5,
          ...(node.offset && {
            u_offset: node.offset,
            u_amplitude: node.amplitude,
            u_frequency: (2 * Math.PI) / node.period,
            u_phase: 2 * Math.PI * ((node.phase + time * node.rate) % 1)
          })
        };
        if (effect && node.enabled) {
          if (layer) {
            fullscreen(shader, probeTarget.framebuffer, 1, 1, {
              ...values, u_effectProbe: 2, u_effectPoint: effectPoint, u_passSize: [1,1]
            });
            const [x,y,code,alpha] = readPixel(probeTarget.framebuffer, 0, 0);
            probe.note(node, effect, { selected: Boolean(code & 4), status: code & 3, cell: [x,y], alpha,
              gate: effect.kind === 'particles' ? x / 255 : undefined, raster: layer, camera });
          } else probe.note(node, effect);
        }
        if (node.effect) {
          textureUnit(9, hasGroups ? groupsTexture : scene.textures[0]);
          effects(node, layout.passes[index], out, values, w, h, time, input.textures[0]);
        } else if (node.bank) {
          const bank = openHistory(node);
          const stamp = bank.capture ? refreshBank(bank, outputs.get(node.source), outputs, layout.passes[index], w, h) : bank.value;
          bank.save(stamp);
          // The capture program uses units 0-10, so restore the display pass inputs.
          textureUnit(0, input.textures[0]);
          textureUnit(2, (mask ?? scene).textures[0]);
          textureUnit(3, ownership?.texture ?? scene.textures[0]);
          textureUnit(5, select ? selected : scene.textures[0]);
          textureUnit(6, depth ?? scene.textures[0]);
          textureUnit(7, stamp.textures[0]);
          fullscreen(shader, out.framebuffer, w, h, values);
        } else if (node.type === 'frameAge') {
          const age = openHistory(node);
          if (age.capture) {
            const work = layout.passes[index];
            gl.activeTexture(gl.TEXTURE0);
            const stamp = age.spare ?? target();
            stamp.resize(work.width, work.height);
            textureUnit(0, outputs.get(node.source).textures[0]);
            textureUnit(7, scene.textures[0]);
            fullscreen(shader, stamp.framebuffer, work.width, work.height, {
              ...values, u_passSize: [work.width, work.height], u_captureHistory: true,
              u_hasMask: false, u_depthMask: false, u_hasAuxMask: false, u_auxMask: 8
            });
            age.save(stamp);
          }
          const ready = node.regions.flatMap(region => {
            const stamp = age.sample(region);
            return stamp ? [{ stamp, mask: outputs.get(region.mask) }] : [];
          });
          if (ready.length > 1) {
            gl.activeTexture(gl.TEXTURE0);
            ageScratch ??= target();
            ageScratch.resize(w, h);
          }
          let prior = input;
          for (const [i, { stamp, mask }] of ready.entries()) {
            const next = (ready.length - i) % 2 ? out : ageScratch;
            textureUnit(0, prior.textures[0]);
            textureUnit(7, stamp.textures[0]);
            textureUnit(8, (mask ?? scene).textures[0]);
            fullscreen(shader, next.framebuffer, w, h, {
              ...values, u_captureHistory: false, u_hasAuxMask: Boolean(mask), u_auxMask: 8
            });
            prior = next;
          }
          if (!ready.length) {
            textureUnit(0, input.textures[0]);
            textureUnit(7, scene.textures[0]);
            fullscreen(shader, out.framebuffer, w, h, { ...values, u_operation: 0 });
          }
        } else if (memory) {
          const elapsed = smear ? memory.sampleElapsed : 0;
          const smearDecay = elapsed > 0 && (!node.motion || rawMotion) ? 2 ** (-elapsed / node.halfLife) : 0;
          const heldBlocks = smear && outputs.get(node.hold);
          const smearActive = smearDecay > 0 || elapsed > 0 && Boolean(heldBlocks);
          if (smear) {
            const refresh = outputs.get(node.refresh), blocks = memory.blocks;
            const rect = blocks?.rect ?? [0, 0, w, h];
            const { cycle, tiles } = tileUniforms(blocks?.grid, rect[2] - rect[0]);
            textureUnit(8, (refresh ?? scene).textures[0]);
            textureUnit(10, (heldBlocks ?? scene).textures[0]);
            Object.assign(values, {
              u_auxMask: 8, u_hasAuxMask: Boolean(refresh), u_hasFlow: Boolean(node.field || rawMotion),
              u_needsRawMotion: Boolean(node.motion), u_blockMemory: Boolean(blocks),
              u_blockHold: 10, u_hasBlockHold: Boolean(heldBlocks), u_blockRect: rect,
              u_blockCycle: cycle, u_blockTiles: tiles,
              u_rawMotion: Boolean(rawMotion), u_motionPerFrame: node.units === 'pixels/frame',
              u_flowScale: node.direction === 'backward' ? node.fieldScale.map(value => -value) : node.fieldScale,
              u_flowVelocity: node.velocity, u_flowBlock: node.blockSize,
              u_smearElapsed: elapsed, u_decay: smearDecay
            });
          }
          let stamp = memory.value;
          if (memory.capture) {
            // Target allocation binds a texture on the active unit.
            gl.activeTexture(gl.TEXTURE0);
            stamp = historyFree.pop() ?? target();
            const work = layout?.passes[index] ?? { width: w, height: h };
            stamp.resize(work.width, work.height);
            // Resizing rebinds texture unit zero.
            textureUnit(0, input.textures[0]);
            textureUnit(7, memory.value?.textures[0] ?? scene.textures[0]);
            const motion = feedback && api.feedbackMotion(node, time, memory.decay > 0 ? memory.elapsed : 0, w, h);
            fullscreen(shader, stamp.framebuffer, work.width, work.height, {
              ...values,
              u_passSize: [work.width, work.height],
              u_captureHistory: true,
              u_hasHistory: smear ? smearActive : Boolean(memory.value) && (!feedback || memory.decay > 0),
              u_hold: node.mode === 'hold',
              u_decay: smear ? smearDecay : feedback || node.mode === 'persistence' ? memory.decay : 0,
              ...(motion && {
                u_feedbackTransform: motion.transform,
                u_feedbackPivot: motion.pivot,
                u_offset: motion.offset,
                u_amplitude: motion.wave,
                u_frequency: motion.frequency,
                u_phase: motion.theta,
                u_blockSize: motion.blockSize,
                u_feedbackLinear: node.filter === 'linear',
                u_gain: node.gain * (node.layer ? (layer?.opacity ?? 0) : 1),
                u_hasThreshold: Boolean(node.threshold),
                u_threshold: node.threshold ?? [0, 1]
              })
            });
            memory.save(stamp);
          }
          textureUnit(7, stamp.textures[0]);
          fullscreen(shader, out.framebuffer, w, h, {
            ...values, u_captureHistory: false, u_hasHistory: !smear || smearActive
          });
        } else {
          textureUnit(7, scene.textures[0]);
          fullscreen(shader, out.framebuffer, w, h, values);
        }
        outputs.set(node.id, out);
      }
      for (const id of node.dependencies)
        if (plan.lastUse.get(id) === index) {
          const used = outputs.get(id);
          outputs.delete(id);
          if (used !== scene && ![...outputs.values()].includes(used)) free.push(used);
        }
    }
    if (plan.captures.size) frame[graphCapture](capture(plan, outputs, w, h));
    if (probe) frame[graphEffectPixel]?.(probe.finish(
      readPixel(scene.framebuffer, at[0], h - 1 - at[1]),
      readPixel(outputs.get(plan.output).framebuffer, at[0], h - 1 - at[1])
    ));
    textureUnit(0, outputs.get(plan.output).textures[0]);
    fullscreen(shader, destination, w, h, { u_operation: 0, u_effectProbe: 0, u_cube: 4, u_size: [w, h], u_passSize: [w, h] });
  }
  return Object.assign(execute, { reset: history.clear });
}
