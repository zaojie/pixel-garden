const textureWidth = 512;
const maxRows = 0xfffffe;

export function prepareCoverageVote(world, setting, paletteSize) {
  if (!setting) return null;
  const { ramps } = setting;
  if (!ramps || Object.getPrototypeOf(ramps) !== Object.prototype)
    throw new TypeError('Coverage voting needs a ramps object keyed by material');
  const { objectIds, materialIds, materials } = world.renderMetadata ?? {};
  if (!objectIds || !materialIds || !materials || !world.count) return null;
  if (world.count > maxRows) throw new RangeError('Coverage voting needs 24-bit splat IDs');
  const rampByMaterial = new Map(), swatches = [];
  for (const [label, ramp] of Object.entries(ramps)) {
    const material = materials.indexOf(label);
    if (!Array.isArray(ramp) || !ramp.length || ramp.length > paletteSize ||
        Array.from(ramp).some(i => !Number.isInteger(i) || i < 0 || i >= paletteSize))
      throw new RangeError('Each coverage ramp needs valid scene palette indices');
    if (material < 0) continue;
    rampByMaterial.set(material, swatches.length + 1);
    swatches.push(ramp);
  }
  const groups = new Map(), memberships = new Float32Array(world.count * 2);
  for (let i = 0; i < world.count; i++) {
    const material = materialIds[i], ramp = rampByMaterial.get(material) ?? 0;
    const key = objectIds[i] + '/' + material;
    if (!groups.has(key)) groups.set(key, groups.size + 1);
    memberships[i * 2] = groups.get(key);
    memberships[i * 2 + 1] = ramp;
  }
  return swatches.length ? { memberships, swatches } : null;
}

const resolveSource = paletteSize => `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_base,u_sampleColor,u_sampleOwner,u_voteRows,u_voteRamps;
uniform vec3 u_votePalette[${paletteSize}],u_voteOutput[${paletteSize}];
out vec4 color;
vec2 identity(vec4 sampleRow) {
  vec3 bytes = round(sampleRow.rgb*255.0);
  int row = int(dot(bytes,vec3(1,256,65536)))-1;
  if (row<0) return vec2(0);
  vec4 packedRow=texelFetch(u_voteRows,ivec2((row>>1)&511,row>>10),0);
  return (row&1)==0?packedRow.rg:packedRow.ba;
}
void main() {
  ivec2 p=ivec2(gl_FragCoord.xy);
  color=texelFetch(u_base,p,0);
  vec2 ids[4];
  vec4 samples[4];
  for(int i=0;i<4;i++){
    ivec2 tap=p*2+ivec2(i&1,i>>1);
    vec4 owner=texelFetch(u_sampleOwner,tap,0);
    ids[i]=identity(owner);
    samples[i]=texelFetch(u_sampleColor,tap,0);
    if(ids[i].x>0.0) {
      float hi=owner.a;
      samples[i].a=dot(round(vec2(hi,samples[i].a)*255.0),vec2(256,1))/65535.0;
    }
  }
  if(ids[0].x==ids[1].x&&ids[0].x==ids[2].x&&ids[0].x==ids[3].x)return;
  int winner=0, bestCount=0;
  float bestDepth=2.0;
  for(int i=0;i<4;i++){
    int count=0;
    float depth=1.0;
    for(int j=0;j<4;j++)if(ids[i].x==ids[j].x){count++;depth=min(depth,samples[j].a);}
    if(count>bestCount||count==bestCount&&(depth<bestDepth||
        depth==bestDepth&&ids[i].x<ids[winner].x)){
      winner=i;bestCount=count;bestDepth=depth;
    }
  }
  int ramp=int(ids[winner].y)-1;
  if(ramp<0)return;
  vec3 mean=vec3(0);
  for(int i=0;i<4;i++)if(ids[i].x==ids[winner].x)mean+=samples[i].rgb;
  mean/=float(bestCount);
  float best=1.0e8;
  int bestIndex=0;
  for(int i=0;i<${paletteSize};i++){
    int index=int(texelFetch(u_voteRamps,ivec2(i,ramp),0).r)-1;
    if(index<0)break;
    vec3 difference=mean-u_votePalette[index];
    float error=dot(difference*difference,vec3(.30,.55,.15));
    if(error<best){best=error;bestIndex=index;}
  }
  color.rgb=u_voteOutput[bestIndex];
}
`;

export function createCoverageVote(gl, { samples, source, resources, table, program, textureUnit, fullscreen, maxSize, paletteSize }) {
  const rows = table(textureWidth, 12, maxSize);
  const ramps = resources.texture();
  const shader = program(resolveSource(paletteSize));
  let metadata, setting, count, prepared, palSize;
  function prepare(world, definition, paletteCount) {
    if (metadata === world.renderMetadata && setting === definition && count === world.count && palSize === paletteCount)
      return prepared;
    prepared = prepareCoverageVote(world, definition, paletteCount);
    metadata = world.renderMetadata;
    setting = definition;
    count = world.count;
    palSize = paletteCount;
    if (prepared) {
      if (prepared.swatches.length > maxSize) throw new RangeError('Coverage ramp table exceeds the device texture limit');
      rows.upload(prepared.memberships);
      const packed = new Float32Array(paletteSize * prepared.swatches.length);
      for (const [i, ramp] of prepared.swatches.entries())
        packed.set(ramp.map(index => index + 1), i * paletteSize);
      textureUnit(13, ramps);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, paletteSize, prepared.swatches.length, 0, gl.RED, gl.FLOAT, packed);
    }
    return prepared;
  }
  return {
    samples, source, prepare,
    finish(destination, width, height, pal, graded) {
      textureUnit(8, source.textures[0]);
      textureUnit(10, samples.textures[0]);
      textureUnit(11, samples.textures[1]);
      rows.bind();
      textureUnit(13, ramps);
      fullscreen(shader, destination, width, height, {
        u_base:8, u_sampleColor:10, u_sampleOwner:11, u_voteRows:12, u_voteRamps:13,
        u_votePalette:pal, u_voteOutput:graded ?? pal
      });
    }
  };
}
