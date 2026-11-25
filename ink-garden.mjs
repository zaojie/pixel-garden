import {makeBuilder, makeSculpt, KIT} from './ink-toolkit.mjs';

export function buildGarden() {
  const palette = [0x101619,0x172a31,0x293f46,0x445e61,0x75918b,
    0x253e37,0x3d6049,0x60835b,0x9aa677,
    0x362c35,0x61434a,0x99655e,0xc18d71,
    0x55384e,0x965a78,0xc47c99,0xe5a2b3,0xffd0cf,
    0x173d48,0x245d66,0x448990,0x88c0b2,
    0x493e38,0x826044,0xb98a5b,0xf0bf7d,0xffe0a1,
    0x234248,0x3b6369,0x64878b,0xa3b5a6,0xaa4e40,0xdf8860];
  const B=makeBuilder(142,palette),S=KIT.sculpt(makeSculpt(B));
  S.res=.075; S.amb=.25; S.sky=.28; S.sunK=.6; S.sun=B.norm([-.5,.85,.5]);
  S.wob=.025; S.fat=.52;
  const stone={r:[1,2,3,4],mod:(x,y,z)=>(B.n3(x*3,y*3,z*3)-.5)*.16};
  const moss={r:[5,6,7,8],mod:(x,y,z)=>(B.n3(x*3,y*3,z*3)-.5)*.2};
  const bark={r:[9,10,11,12]},pink={r:[13,14,15,16,17],leaf:true,still:true};
  const wood={r:[22,23,24,25]},roof={r:[27,28,29,30]},lamp={r:[24,25,26],em:true};
  const lanterns=[[-3,1.2],[1.7,2.1],[3,-1.9],[-.5,-2.6]];
  for(const [x,z] of lanterns) S.lights.push({p:[x,.85,z],r:2.5,s:.8});
  S.lights.push({p:[1.2,1.3,-.1],r:3.1,s:.7});
  // The island and cut-out pond are signed distance solids, not point meshes.
  const island=S.lathe([0,-1.15,0],1.15,t=>3.25+.6*t+.08*Math.sin(t*11),stone,{hold:true,res:.1});
  const basin=S.ball([-.55,.08,1],[2.15,.3,1.3],stone,{hold:true});
  S.join('cut',island,basin,{res:.1});
  const turf=S.lathe([0,-.03,0],.13,3.85,moss,{hold:true,res:.085});
  S.join('cut',turf,basin,{res:.085});
  S.ball([-.55,-.03,1],[2.08,.07,1.23],{r:[18,19,20,21],mod:(x,y,z)=>Math.sin(x*7+z*12)*.1},{res:.075});
  // Shrine: carved SDF woodwork with emissive paper panels.
  const x=1.25,z=-.85;
  S.box([x,.22,z],[1.08,.22,.95],stone,{rd:.05});
  S.box([x,.5,z],[1.16,.08,1],wood);
  S.box([x,1.28,z],[.87,.7,.77],wood);
  S.box([x,1.3,z+.79],[.72,.5,.018],lamp,{res:.045});
  S.box([x+.89,1.3,z],[.018,.5,.62],lamp,{res:.045});
  for(let a=-.75;a<.8;a+=.3) S.box([x+a,1.3,z+.83],[.027,.57,.025],bark,{res:.04});
  for(const y of [.9,1.25,1.6]) S.box([x,y,z+.84],[.78,.025,.026],bark,{res:.045});
  for(let a=-.6;a<.65;a+=.3) S.box([x+.93,1.3,z+a],[.025,.57,.025],bark,{res:.04});
  for(const a of [-1,1]) for(const b of [-.83,.83]) S.box([x+a,1.28,z+b],[.075,.8,.075],wood,{res:.06});
  const roofOutline=[[-1.4,0],[-1.12,.04],[0,.79],[1.12,.04],[1.4,0],[1.4,-.14],[1.1,-.13],[0,.6],[-1.1,-.13],[-1.4,-.14]];
  S.extrude([roofOutline],[x,2.05,z],2.22,roof,{res:.055});
  S.tube([[x,2.85,z-1.18],[x,2.85,z+1.18]],.065,.065,roof,{res:.055});
  for(let a=-1.3;a<=1.3;a+=.22){const h=2.8-Math.abs(a)*.55;S.tube([[x+a,h,z-1.13],[x+a,h,z+1.13]],.022,.022,{r:[28,29,30]},{res:.055});}
  for(let i=0;i<3;i++) S.box([x,.07+i*.11,z+1.38-i*.16],[.57,.06,.2],stone);
  // Branches and clustered blossoms share the SDF union before baking.
  S.tube([[-2,.05,-.7],[-2.12,1.2,-.6],[-1.9,2.2,-.8],[-2.25,3,-.9]],.22,.07,bark);
  S.tube([[-2.1,1.1,-.6],[-2.8,2.1,-.4],[-3.1,2.55,-.35]],.14,.035,bark);
  S.tube([[-2,1.8,-.7],[-1.25,2.35,-.45],[-.8,2.7,-.65]],.1,.03,bark);
  let seed=24;const random=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
  for(const [cx,cy,cz,r] of [[-2.3,3.2,-.9,.8],[-3,2.6,-.4,.65],[-1.25,2.9,-.6,.75],[-2.1,2.65,.05,.6],[-2.85,3,-1.2,.55]]) {
    for(let i=0;i<5;i++) {const a=i*1.256;S.ball([cx+Math.cos(a)*r*.48,cy+(random()-.3)*.3,cz+Math.sin(a)*r*.5],[r*.6,r*.42,r*.58],pink,{res:.07});}
  }
  // A curved bridge is sculpted from planks, posts and two continuous rails.
  for(let i=0;i<22;i++) {const u=i/21,bx=-1.8+u*2.6,by=.2+Math.sin(u*Math.PI)*.35;
    S.box([bx,by,1.15],[.055,.055,.43],wood,{res:.055});
    if(i%4===0) for(const bz of [.7,1.6]) S.box([bx,by+.24,bz],[.035,.25,.035],wood,{res:.05});
  }
  for(const bz of [.7,1.6]) S.tube(Array.from({length:12},(_,i)=>[-1.8+i/11*2.6,.69+Math.sin(i/11*Math.PI)*.35,bz]),.035,.035,wood,{res:.05});
  for(const tx of [2.4,3.35]) S.tube([[tx,.08,1.25],[tx,1.75,1.25]],.085,.07,{r:[22,31,32]}, {res:.06});
  S.box([2.875,1.5,1.25],[.72,.065,.085],{r:[31,32]}, {res:.055});
  S.tube([[1.95,1.92,1.25],[2.3,1.84,1.25],[3.45,1.84,1.25],[3.8,1.92,1.25]],.09,.09,roof,{res:.06});
  for(const [lx,lz] of lanterns){
    S.lathe([lx,.04,lz],.54,[.15,.12],stone);
    S.box([lx,.77,lz],[.16,.2,.16],lamp,{res:.05});
    S.lathe([lx,.97,lz],.16,[.29,.08],roof,{res:.055});
    for(const dx of [-.17,.17]) for(const dz of [-.17,.17]) S.box([lx+dx,.77,lz+dz],[.02,.22,.02],bark,{res:.04});
  }
  for(let i=0;i<24;i++){const a=i/24*Math.PI*2;S.ball([Math.cos(a)*3.57,.12,Math.sin(a)*3.57],[.2+random()*.17,.15+random()*.18,.2],stone,{res:.085});}
  for(let i=0;i<10;i++){const lx=-2.15+random()*3,lz=.25+random()*2;if(lz>.65&&lz<1.7)continue;S.ball([lx,.06,lz],[.16,.025,.12],moss,{res:.045});}
  S.bake();
  const surfaceCount=B.count();
  KIT.weather.fall(B,{kind:'leaf',box:[-3.3,-.7,-1.5,.6],top:3.3,n:35,cols:[15,16,17],speed:.22,size:.035});
  KIT.weather.fall(B,{kind:'rain',rad:4,top:5,n:220,cols:[3,4,20],speed:3.5,size:.007,len:.12});
  const world=B.finish({cam:{target:[0,1,0],dist:13,pitch:.38,yaw:.45,ortho:true},detail:15,dither:.2,radius:5.5,outline:{c:0,edge:.1}});
  return {world,surfaceCount};
}
