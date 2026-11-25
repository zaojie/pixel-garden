import {KIT,PixelLook} from './ink-toolkit.mjs';
import {createRenderer} from './vendor/ink-studio/src/webgl/renderer.mjs';

const $=id=>document.getElementById(id),canvas=$('scene');
let renderer;
try {
  const [info,rows]=await Promise.all([fetch('./ink-garden.json'),fetch('./ink-garden.bin')]);
  if(!info.ok||!rows.ok) throw new Error('场景文件读取失败，请先运行 node bake-ink.mjs');
  const world=await info.json();world.data=new Float32Array(await rows.arrayBuffer());
  world.moving=Int32Array.from(world.moving);world.back=Int32Array.from(world.back);
  if(world.data.length!==world.count*16) throw new Error('高斯数据长度不匹配');
  // Baked colors need only the official lightweight palette resolve shader.
  renderer=createRenderer(canvas,{toolkit:{KIT,PixelLook},resolve:'simple'});
  const initial={...world.cam,target:[...world.cam.target]};
  let camera={...initial},pixel=3,dither=.2,raw=false,paused=false,time=0;
  let auto=!matchMedia('(prefers-reduced-motion: reduce)').matches,drag=null,last=0,lastDraw=0;
  $('rotate').setAttribute('aria-pressed',String(auto));
  $('point-count').textContent=world.count.toLocaleString('en-US');
  canvas.dataset.renderer='Ink Studio WebGL 2';canvas.dataset.splats=String(world.count);
  let width=320,height=240;
  function resize(){const r=canvas.getBoundingClientRect();const ratio=raw?Math.max(1,r.width/1300):pixel;width=Math.max(1,Math.round(r.width/ratio));height=Math.max(1,Math.round(r.height/ratio));}
  new ResizeObserver(resize).observe(canvas);resize();
  function render(now){
    const dt=last?Math.min((now-last)/1000,.05):0;last=now;
    if(!paused){time+=dt;if(auto&&!drag)camera.yaw+=dt*.06;}
    if(now-lastDraw>32){
      try{
        renderer.render(world,{width,height,time,camera,raw,dither:raw?0:dither,colored:!raw,outline:raw?false:world.outline,changedRows:[]});
        $('needle').style.transform=`rotate(${-camera.yaw*180/Math.PI}deg)`;
        $('loading').hidden=true;lastDraw=now;
      }catch(error){showError(error);return;}
    }
    requestAnimationFrame(render);
  }
  function autoRotate(value){auto=value;$('rotate').setAttribute('aria-pressed',String(value));}
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;camera.yaw-=(e.clientX-drag.x)*.007;camera.pitch=Math.max(.1,Math.min(1.2,camera.pitch+(e.clientY-drag.y)*.005));drag={x:e.clientX,y:e.clientY};autoRotate(false);});
  canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
  canvas.addEventListener('wheel',e=>{e.preventDefault();camera.dist=Math.max(8,Math.min(21,camera.dist+e.deltaY*.008));},{passive:false});
  canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-'].includes(e.key))return;e.preventDefault();autoRotate(false);if(e.key==='ArrowLeft')camera.yaw-=.12;if(e.key==='ArrowRight')camera.yaw+=.12;if(e.key==='ArrowUp')camera.pitch=Math.min(1.2,camera.pitch+.08);if(e.key==='ArrowDown')camera.pitch=Math.max(.1,camera.pitch-.08);if(e.key==='+')camera.dist=Math.max(8,camera.dist-.5);if(e.key==='-')camera.dist=Math.min(21,camera.dist+.5);});
  $('pixel').addEventListener('input',e=>{pixel=Number(e.target.value);$('pixel-value').textContent=`${pixel} px`;resize();});
  $('light').addEventListener('input',e=>{dither=Number(e.target.value)/100;$('light-value').textContent=`${e.target.value}%`;});
  function mode(value){raw=value;for(const [id,selected]of [['pixel-mode',!raw],['raw-mode',raw]]){$(id).classList.toggle('selected',selected);$(id).setAttribute('aria-pressed',String(selected));}for(const id of ['pixel','light'])$(id).disabled=raw;resize();}
  $('pixel-mode').addEventListener('click',()=>mode(false));$('raw-mode').addEventListener('click',()=>mode(true));
  $('rotate').addEventListener('click',()=>autoRotate(!auto));
  $('pause').addEventListener('click',()=>{paused=!paused;$('pause').setAttribute('aria-pressed',String(paused));});
  $('reset').addEventListener('click',()=>{camera={...initial};autoRotate(false);});
  window.addEventListener('pagehide',()=>renderer.dispose(),{once:true});
  requestAnimationFrame(render);
}catch(error){showError(error);}
function showError(error){console.error(error);$('loading').hidden=false;$('loading').textContent=`渲染失败：${error.message}`;}
