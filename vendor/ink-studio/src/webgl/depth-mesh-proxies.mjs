import { createResources } from './resources.mjs';

const vertex = `#version 300 es
precision highp float;
layout(location=0) in vec3 a_position;
uniform mat4 u_view,u_projection;
out vec3 v_world,v_camera;
void main(){
  v_world=a_position;
  vec4 camera=u_view*vec4(a_position,1.0);
  v_camera=camera.xyz;
  gl_Position=u_projection*camera;
}`;

function fragment(cutaway) {
  return `#version 300 es
precision highp float;
in vec3 v_world,v_camera;
uniform mat4 u_projection;
uniform vec4 u_arrivalPlane,u_clip;
uniform float u_tolerance;
${cutaway ?? ''}
void main(){
  if(dot(u_arrivalPlane,vec4(v_world,1.0))<0.0||dot(u_clip,vec4(v_camera,1.0))<0.0)discard;
  ${cutaway ? 'if(sectionMargin(v_world,-v_camera.z)<0.0)discard;' : ''}
  vec4 p=u_projection*vec4(0.0,0.0,v_camera.z-u_tolerance,1.0);
  gl_FragDepth=clamp(p.z/p.w*0.5+0.5,0.0,1.0);
}`;
}

export function createDepthMeshPass(gl, unit, cutaway) {
  const resources = createResources(gl);
  try {
    gl.activeTexture(gl.TEXTURE0+unit);
    const currentBinding=()=>gl.getParameter(gl.TEXTURE_BINDING_2D);
    const bound=currentBinding(), depth=resources.texture(), framebuffer=resources.allocate('Framebuffer');
    gl.bindTexture(gl.TEXTURE_2D,bound);
    const programs=new Map(), slots=[];
    let width=0,height=0,previous;
    function drawMesh(mesh,i) {
      let slot=slots[i];
      if (!slot) {
        slot=slots[i]={vao:resources.allocate('VertexArray'),positions:resources.allocate('Buffer'),
          indices:resources.allocate('Buffer')};
        gl.bindVertexArray(slot.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER,slot.positions);
        gl.vertexAttribPointer(0,3,gl.FLOAT,false,0,0);
        gl.enableVertexAttribArray(0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,slot.indices);
      } else gl.bindVertexArray(slot.vao);
      if (slot.points!==mesh.positions) {
        gl.bindBuffer(gl.ARRAY_BUFFER,slot.positions);
        if (slot.points?.length===mesh.positions.length) gl.bufferSubData(gl.ARRAY_BUFFER,0,mesh.positions);
        else gl.bufferData(gl.ARRAY_BUFFER,mesh.positions,gl.DYNAMIC_DRAW);
        slot.points=mesh.positions;
      }
      if (slot.faces!==mesh.indices) {
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,mesh.indices,gl.STATIC_DRAW);
        slot.faces=mesh.indices;
      }
      gl.drawElements(gl.TRIANGLES,mesh.indices.length,gl.UNSIGNED_INT,0);
    }
    function prepare(frame,w,h) {
      const {depthProxy:plan,view,projection,arrivalPlane,clip,section}=frame;
      if (previous?.plan===plan && previous.clip===clip && w===width && h===height) return depth;
      gl.bindFramebuffer(gl.FRAMEBUFFER,framebuffer);
      if (width!==w || height!==h) {
        gl.activeTexture(gl.TEXTURE0+unit);
        const bound=currentBinding();
        gl.bindTexture(gl.TEXTURE_2D,depth);
        gl.texImage2D(gl.TEXTURE_2D,0,gl.DEPTH_COMPONENT24,w,h,0,gl.DEPTH_COMPONENT,gl.UNSIGNED_INT,null);
        gl.bindTexture(gl.TEXTURE_2D,bound);
        gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,depth,0);
        gl.drawBuffers([gl.NONE]);
        gl.readBuffer(gl.NONE);
        if (gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)
          throw new Error('Splat depth proxy framebuffer is incomplete');
        width=w;
        height=h;
      }
      const sectioned=Boolean(section);
      if (!programs.has(sectioned)) programs.set(sectioned,resources.program(vertex,fragment(sectioned ? cutaway : null)));
      const program=programs.get(sectioned);
      gl.viewport(0,0,w,h);
      gl.disable(gl.BLEND);
      gl.enable(gl.DEPTH_TEST);
      gl.depthFunc(gl.LESS);
      gl.depthMask(true);
      gl.clearBufferfv(gl.DEPTH,0,[1]);
      gl.useProgram(program.p);
      program.set({u_view:view,u_projection:projection,u_arrivalPlane:arrivalPlane,u_clip:clip,
        u_tolerance:plan.tolerance,...section});
      for (const [i,mesh] of plan.meshes.entries()) drawMesh(mesh,i);
      gl.bindVertexArray(null);
      gl.disable(gl.DEPTH_TEST);
      previous={plan,clip};
      return depth;
    }
    return {prepare,dispose:resources.dispose};
  } catch (error) {
    resources.dispose();
    throw error;
  }
}
