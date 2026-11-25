import fs from 'node:fs';
import assert from 'node:assert/strict';
import {buildGarden} from './ink-garden.mjs';
const start=performance.now();
const {world,surfaceCount}=buildGarden();
assert(world.count>1000 && world.data.length===world.count*16);
// Packed RGB is bit-cast into float storage; validate only geometry/covariance fields.
for(let i=0;i<world.count;i++) for(const c of [0,1,2,3,4,5,6,8,9,10]) assert(Number.isFinite(world.data[i*16+c]));
const {data,...metadata}=world;
fs.writeFileSync(new URL('./ink-garden.bin',import.meta.url),Buffer.from(data.buffer,data.byteOffset,data.byteLength));
fs.writeFileSync(new URL('./ink-garden.json',import.meta.url),JSON.stringify({...metadata,surfaceCount},(key,value)=>ArrayBuffer.isView(value)?Array.from(value):value));
console.log(JSON.stringify({splats:world.count,surfaceCount,seconds:((performance.now()-start)/1000).toFixed(2),bytes:data.byteLength,metadata:Object.keys(metadata)}));
