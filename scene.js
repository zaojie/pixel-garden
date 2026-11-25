const canvas = document.querySelector('#scene');
const ctx = canvas.getContext('2d', { alpha: false });
const $ = (id) => document.getElementById(id);
const points = [];
let seed = 142;
function random() { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; }
const choose = (colors) => colors[Math.floor(random() * colors.length)];
const palettes = {
  stone: ['#25383d','#30484a','#3f5554','#536660','#2c4243'],
  moss: ['#365a49','#496b51','#5c7b57','#789068','#3e604d'],
  bark: ['#493b3b','#674b49','#805c54','#3e3538'],
  pink: ['#b56b85','#d0879b','#e6a2ae','#f4bac1','#9c607d','#d99aaa'],
  roof: ['#344a4d','#446265','#537377','#608482','#283e43'],
  wood: ['#745344','#90654e','#a57a56','#5f483d'],
  water: ['#214e54','#295b60','#326b6b','#457f7a','#55938a'],
};
function point(x,y,z,color,size=0.065,kind=0) { points.push({x,y,z,color,size,kind,phase:random()*Math.PI*2}); }
function box(x,y,z,w,h,d,colors,step=0.095,kind=0) {
  for(let a=-w/2;a<=w/2;a+=step) for(let b=-h/2;b<=h/2;b+=step) {
    point(x+a,y+b,z+d/2,choose(colors),step*.8,kind);
    point(x+a,y+b,z-d/2,choose(colors),step*.8,kind);
  }
  for(let a=-w/2;a<=w/2;a+=step) for(let b=-d/2;b<=d/2;b+=step) point(x+a,y+h/2,z+b,choose(colors),step*.8,kind);
  for(let a=-h/2;a<=h/2;a+=step) for(let b=-d/2;b<=d/2;b+=step) {
    point(x-w/2,y+a,z+b,choose(colors),step*.8,kind);
    point(x+w/2,y+a,z+b,choose(colors),step*.8,kind);
  }
}
function ellipsoid(x,y,z,rx,ry,rz,colors,count=300,size=.085,kind=0) {
  for(let i=0;i<count;i++) {
    const u=random()*Math.PI*2, v=Math.acos(random()*2-1), r=.85+random()*.15;
    point(x+rx*Math.sin(v)*Math.cos(u)*r,y+ry*Math.cos(v)*r,z+rz*Math.sin(v)*Math.sin(u)*r,choose(colors),size,kind);
  }
}
function branch(a,b,r,colors) {
  const dx=b[0]-a[0],dy=b[1]-a[1],dz=b[2]-a[2],length=Math.hypot(dx,dy,dz);
  for(let t=0;t<=1;t+=.06/length) for(let k=0;k<10;k++) {
    const q=k*Math.PI/5;
    point(a[0]+dx*t+Math.cos(q)*r,a[1]+dy*t,a[2]+dz*t+Math.sin(q)*r,choose(colors),.07);
  }
}

// A procedural surface-point renderer; use a covariance-based GPU renderer for full 3DGS asset support.
function build() {
  for(let i=0;i<11500;i++) {
    const theta=random()*Math.PI*2,r=Math.sqrt(random()),x=Math.cos(theta)*r*4.4,z=Math.sin(theta)*r*3.25;
    const pond=(x+.4)**2/6.6+(z-1.1)**2/2.6<1;
    point(x,.02+random()*.08,z,choose(pond?palettes.water:palettes.moss),.075,pond?1:0);
  }
  for(let i=0;i<6200;i++) {
    const a=random()*Math.PI*2,y=-random()*1.65, r=1+y*.12+random()*.035;
    const ridge=1+.025*Math.sin(a*13);
    point(Math.cos(a)*4.4*r*ridge,y,Math.sin(a)*3.25*r*ridge,choose(y>-.25?palettes.moss:palettes.stone),.11);
  }
  // Mossy rocks break up the island silhouette.
  for(let i=0;i<28;i++) {
    const a=random()*Math.PI*2,r=2.9+random()*.45;
    ellipsoid(Math.cos(a)*r*1.27,.13,Math.sin(a)*r*.9,.22+random()*.25,.18+random()*.25,.25,palettes.stone,85,.09);
  }
  // Raised shrine, wood frame, luminous paper screens and curved tiled roof.
  const hx=1.3,hz=-.9;
  box(hx,.25,hz,2.05,.4,1.75,palettes.stone,.11);
  box(hx,.53,hz,2.2,.17,1.85,palettes.wood,.1);
  box(hx,1.25,hz,1.78,1.38,1.5,['#5c4c3b','#795c42','#8d6e4a'],.11);
  box(hx,1.3,hz+.77,1.54,1.12,.035,['#ffc684','#ffd49a','#e9a75e'],.075,2);
  box(hx+.91,1.3,hz,.035,1.12,1.2,['#c48e58','#edb36c','#dba35f'],.085,2);
  for(let x=-.76;x<=.8;x+=.38) box(hx+x,1.3,hz+.8,.045,1.25,.06,palettes.wood,.055);
  for(let y=.85;y<1.9;y+=.35) box(hx,y,hz+.81,1.7,.045,.055,palettes.wood,.06);
  for(const x of [-1,1]) for(const z of [-.8,.8]) box(hx+x,1.24,hz+z,.13,1.55,.13,palettes.wood,.09);
  for(let x=-1.4;x<=1.4;x+=.065) for(let z=-1.12;z<=1.12;z+=.07) {
    const y=2.67-Math.abs(x)*.5+Math.pow(Math.abs(x)/1.4,5)*.21;
    point(hx+x,y,hz+z,choose(palettes.roof),.08);
    if(Math.abs(z)>1.04) point(hx+x,y-.08,hz+z,'#91aaa0',.065);
  }
  branch([hx,2.7,hz-1.2],[hx,2.7,hz+1.2],.075,palettes.roof);
  for(let i=0;i<3;i++) box(hx,.12+i*.12,hz+1.3-i*.18,1.2,.12,.38,palettes.stone,.1);
  // Large flowering tree with a branching, visible silhouette.
  branch([-2.05,.08,-.65],[-2.15,1.65,-.65],.18,palettes.bark);
  branch([-2.15,1.2,-.65],[-1.65,2.5,-.8],.12,palettes.bark);
  branch([-2.15,1.4,-.65],[-3,2.4,-.6],.1,palettes.bark);
  branch([-2.05,1.9,-.7],[-2.3,2.95,-1.15],.09,palettes.bark);
  const crowns=[[-2.9,2.65,-.6,.95,.47,.78],[-1.55,2.9,-.75,1,.5,.9],[-2.35,3.35,-1.05,1.05,.55,.85],[-2.1,2.85,.05,.8,.43,.65],[-3.2,3.05,-1.25,.65,.4,.6]];
  for(const c of crowns) ellipsoid(...c,palettes.pink,1250,.078);
  for(let i=0;i<180;i++) {const a=random()*6.28,r=random()*1.3;point(-2+Math.cos(a)*r,.15,-.6+Math.sin(a)*r,choose(palettes.pink),.06);}
  // A little arched footbridge over the foreground pond.
  for(let i=0;i<25;i++) {
    const x=-1.7+i*.105,y=.24+Math.sin(i/24*Math.PI)*.33;
    box(x,y,1.37,.088,.09,.79,palettes.wood,.065);
    if(i%4===0) for(const z of [.95,1.79]) box(x,y+.25,z,.055,.5,.055,palettes.wood,.06);
    for(const z of [.95,1.79]) point(x,y+.5,z,'#bc926b',.07);
  }
  // Torii gate at the right side of the garden.
  for(const x of [2.45,3.55]) box(x,.85,1.1,.13,1.7,.14,['#9f543d','#bc7051','#ce8a60'],.075);
  box(3,1.56,1.1,1.52,.12,.16,['#a65d46','#d28b61'],.07);
  for(let x=2.05;x<3.96;x+=.065) point(x,1.95+(Math.abs(x-3)**3)*.12,1.1,'#4c6664',.11);
  box(3,1.75,1.1,.12,.3,.12,palettes.wood,.06);
  // Lanterns, reeds and small lily pads.
  for(const [x,z] of [[-2.8,1.5],[1.8,1.75],[3,-1.9],[-.3,-2.4]]) {
    box(x,.37,z,.22,.65,.22,palettes.stone,.09);
    box(x,.82,z,.32,.35,.32,['#ffd893','#f0b76c','#ffcf7f'],.06,2);
    box(x,1.04,z,.47,.1,.47,palettes.roof,.08);
    lights.push({x,y:.82,z});
  }
  for(let i=0;i<35;i++) {
    const a=random()*6.28,x=-.4+Math.cos(a)*(2.3+random()*.4),z=1.1+Math.sin(a)*1.45;
    if(x*x/19+z*z/10>1) continue;
    for(let k=0;k<5;k++) {const h=.15+random()*.4;branch([x+k*.025,.08,z],[x+k*.04,.08+h,z+.06],.015,['#688b62','#98ac72']);}
  }
  for(let i=0;i<24;i++) {
    const x=-2.3+random()*3.5,z=.3+random()*2;
    if(z<1.8&&z>.9&&x<1) continue;
    ellipsoid(x,.13,z,.1+random()*.09,.018,.09,['#61876c','#84a27b','#486f59'],18,.055);
  }
  for(let i=0;i<36;i++) petals.push({x:-3.4+random()*3,y:random()*3.6,z:-1.5+random()*3,phase:random()*6.28});
  for(let i=0;i<180;i++) rain.push({x:random()*10-5,y:random()*6,z:random()*8-4,speed:2+random()*2});
}
const lights=[],petals=[],rain=[];
build();
if(!points.length || points.some(p=>![p.x,p.y,p.z,p.size].every(Number.isFinite))) throw new Error('Invalid scene geometry');
$('point-count').textContent=points.length.toLocaleString('en-US');
let yaw=.5,pitch=.42,zoom=1,pixel=3,brightness=.8,wet=true,paused=false;
let auto=!matchMedia('(prefers-reduced-motion: reduce)').matches,time=0,width=0,height=0,scale=1;
let last=performance.now(),frame=0,drag=null;
$('rotate').setAttribute('aria-pressed',String(auto));
function resize(){const r=canvas.getBoundingClientRect();width=Math.round(r.width/pixel);height=Math.round(r.height/pixel);canvas.width=width;canvas.height=height;scale=Math.min(width/11.9,height/9)*zoom;}
new ResizeObserver(resize).observe(canvas);
function project(x,y,z){const c=Math.cos(yaw),s=Math.sin(yaw),rx=x*c-z*s,rz=x*s+z*c,py=(y-.8)*Math.cos(pitch)-rz*Math.sin(pitch);return {x:width*.53+rx*scale,y:height*.55-py*scale,d:rz*Math.cos(pitch)+(y-.8)*Math.sin(pitch)};}
function glow(x,y,r,color){const g=ctx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,color);g.addColorStop(1,'transparent');ctx.fillStyle=g;ctx.fillRect(x-r,y-r,r*2,r*2);}
function draw(now){
  requestAnimationFrame(draw);
  const delta=Math.min((now-last)/1000,.05);last=now;
  if(!paused){time+=delta;if(auto&&!drag)yaw+=delta*.065;}
  // Keep the preview inexpensive on laptops: the scene renders at 30 fps.
  if(frame++%2) return;
  scale=Math.min(width/11.9,height/9)*zoom;
  ctx.fillStyle='#101619';ctx.fillRect(0,0,width,height);
  glow(width*.54,height*.51,width*.42,'#21363888');
  const shadow=project(0,-1.85,0);
  ctx.save();ctx.translate(shadow.x,shadow.y);ctx.scale(1,.32);glow(0,0,scale*4.5,'#03080cd9');ctx.restore();
  const projected=points.map(p=>{let y=p.y;if(p.kind===1)y+=Math.sin(time*1.1+p.x*2+p.z)*.018;return {...project(p.x,y,p.z),p};});
  projected.sort((a,b)=>a.d-b.d);
  for(const {x,y,p} of projected){
    const size=Math.max(1,Math.ceil(p.size*scale));
    ctx.fillStyle=p.color;
    ctx.globalAlpha=p.kind===2?.5+brightness*.5:1;
    ctx.fillRect(Math.round(x-size/2),Math.round(y-size/2),size,size);
    if(p.kind===1&&Math.sin(p.phase+time*.9)>.993){ctx.fillStyle='#83b9ac';ctx.globalAlpha=.5;ctx.fillRect(Math.round(x),Math.round(y),size*2,1);}
  }
  ctx.globalAlpha=1;
  ctx.globalCompositeOperation='screen';
  for(const l of [...lights,{x:1.3,y:1.25,z:-.1}]) {const p=project(l.x,l.y,l.z);glow(p.x,p.y,scale*.6,`rgba(255,166,65,${brightness*.24})`);}
  ctx.globalCompositeOperation='source-over';
  for(const p of petals){const y=((p.y-time*.18)%3.6+3.6)%3.6,pos=project(p.x+Math.sin(time*.5+p.phase)*.3,y,p.z+Math.sin(time*.3+p.phase)*.2);ctx.fillStyle='#e5acb7';ctx.globalAlpha=.7;ctx.fillRect(Math.round(pos.x),Math.round(pos.y),1,1);}
  if(wet){ctx.strokeStyle='#8cb0c8';ctx.lineWidth=1;ctx.globalAlpha=.22;ctx.beginPath();for(const r of rain){const y=((r.y-time*r.speed)%6+6)%6,a=project(r.x,y,r.z),b=project(r.x-.025,y-.15,r.z);ctx.moveTo(Math.round(a.x),Math.round(a.y));ctx.lineTo(Math.round(b.x),Math.round(b.y));}ctx.stroke();}
  ctx.globalAlpha=1;
  $('needle').style.transform=`rotate(${-yaw*180/Math.PI}deg)`;
}
requestAnimationFrame(draw);
function toggleAuto(value){auto=value;$('rotate').setAttribute('aria-pressed',String(auto));}
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag.x)*.008;pitch=Math.max(.12,Math.min(1.1,pitch+(e.clientY-drag.y)*.005));drag={x:e.clientX,y:e.clientY};toggleAuto(false);});
canvas.addEventListener('pointerup',()=>drag=null);
canvas.addEventListener('pointercancel',()=>drag=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.65,Math.min(1.65,zoom-e.deltaY*.001));},{passive:false});
canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-'].includes(e.key))return;e.preventDefault();toggleAuto(false);if(e.key==='ArrowLeft')yaw-=.12;if(e.key==='ArrowRight')yaw+=.12;if(e.key==='ArrowUp')pitch=Math.min(1.1,pitch+.08);if(e.key==='ArrowDown')pitch=Math.max(.12,pitch-.08);if(e.key==='+')zoom=Math.min(1.65,zoom+.1);if(e.key==='-')zoom=Math.max(.65,zoom-.1);});
$('pixel').addEventListener('input',e=>{pixel=Number(e.target.value);$('pixel-value').textContent=`${pixel} px`;resize();});
$('light').addEventListener('input',e=>{brightness=Number(e.target.value)/100;$('light-value').textContent=`${e.target.value}%`;});
function setWeather(value){wet=value;for(const [id,selected] of [['rain',wet],['clear',!wet]]){$(id).classList.toggle('selected',selected);$(id).setAttribute('aria-pressed',String(selected));}}
$('rain').addEventListener('click',()=>setWeather(true));
$('clear').addEventListener('click',()=>setWeather(false));
$('rotate').addEventListener('click',()=>toggleAuto(!auto));
$('pause').addEventListener('click',()=>{paused=!paused;$('pause').setAttribute('aria-pressed',String(paused));});
$('reset').addEventListener('click',()=>{yaw=.5;pitch=.42;zoom=1;toggleAuto(false);});
