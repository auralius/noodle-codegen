/* Noodle Studio R15 — local, dependency-free TFLite TFL3 FlatBuffer inspector.
 * Model import/metadata only; it does NOT claim operator executability on the MCU.
 * Source model bytes are retained separately by Studio for lossless JSON saves.
 */
'use strict';
(function(root){
const MAX_BYTES = 64 * 1024 * 1024;
const OP_NAMES = {0:'ADD',1:'AVERAGE_POOL_2D',2:'CONCATENATION',3:'CONV_2D',4:'DEPTHWISE_CONV_2D',6:'DEQUANTIZE',9:'FULLY_CONNECTED',17:'MAX_POOL_2D',18:'MUL',19:'RELU',21:'RELU6',22:'RESHAPE',25:'SOFTMAX',34:'PAD',39:'TRANSPOSE',40:'MEAN',41:'SUB',53:'CAST',55:'MAXIMUM',57:'MINIMUM',65:'SLICE',67:'TRANSPOSE_CONV',70:'EXPAND_DIMS',83:'PACK',97:'RESIZE_NEAREST_NEIGHBOR',114:'QUANTIZE'};
const DTYPES = {0:'FLOAT32',1:'FLOAT16',2:'INT32',3:'UINT8',4:'INT64',6:'BOOL',7:'INT16',9:'INT8',10:'FLOAT64',15:'UINT32',16:'UINT16',17:'INT4'};
const ACTIVATIONS = ['NONE','RELU','RELU_N1_TO_1','RELU6','TANH','SIGN_BIT'];
class InvalidTFLite extends Error {constructor(message){super(message);this.name='InvalidTFLite';}}
class Flat {
 constructor(src){const b=src instanceof Uint8Array?src:new Uint8Array(src);
   if(b.length<8||b.length>MAX_BYTES)throw new InvalidTFLite('File size outside supported limits (8 bytes to 64 MiB).');
   if(String.fromCharCode(...b.slice(4,8))!=='TFL3')throw new InvalidTFLite('Not a TFLite TFL3 FlatBuffer.');
   this.data=b;this.dv=new DataView(b.buffer,b.byteOffset,b.byteLength);this.length=b.length;
 }
 at(p,n){if(!Number.isSafeInteger(p)||p<0||p+n>this.length)throw new InvalidTFLite(`FlatBuffer read outside file: ${p}, ${n} bytes`);return p;}
 u8(p){this.at(p,1);return this.dv.getUint8(p);}
 i8(p){this.at(p,1);return this.dv.getInt8(p);}
 u16(p){this.at(p,2);return this.dv.getUint16(p,true);}
 u32(p){this.at(p,4);return this.dv.getUint32(p,true);}
 i32(p){this.at(p,4);return this.dv.getInt32(p,true);}
 i64(p){this.at(p,8);const n=this.dv.getBigInt64(p,true);if(n>BigInt(Number.MAX_SAFE_INTEGER)||n<BigInt(Number.MIN_SAFE_INTEGER))throw new InvalidTFLite('Quantization integer exceeds safe JavaScript range');return Number(n);}
 f32(p){this.at(p,4);return this.dv.getFloat32(p,true);}
 field(pos,index){this.at(pos,4);const vt=pos-this.i32(pos);const len=this.u16(vt);if(len<4||vt+len>this.length)throw new InvalidTFLite('Invalid FlatBuffer vtable');const slot=vt+4+2*index;if(slot+2>vt+len)return null;const delta=this.u16(slot);return delta?this.at(pos+delta,1):null;}
 val(pos,index,kind,def=0){const p=this.field(pos,index);return p===null?def:this[kind](p);}
 ref(pos,index){const p=this.field(pos,index);return p===null?null:this.at(p+this.u32(p),4);}
 vec(pos,index,itemBytes=4){const p=this.ref(pos,index);if(p===null)return [];
   const n=this.u32(p);if(n>2000000||n*itemBytes>this.length)throw new InvalidTFLite('Invalid vector length');const a=p+4;this.at(a,n*itemBytes);return Array.from({length:n},(_,i)=>a+i*itemBytes);}
 tabs(pos,index){return this.vec(pos,index).map(p=>this.at(p+this.u32(p),4));}
 ints(pos,index){return this.vec(pos,index).map(p=>this.i32(p));}
 floats(pos,index){return this.vec(pos,index).map(p=>this.f32(p));}
 longs(pos,index){return this.vec(pos,index,8).map(p=>this.i64(p));}
 str(pos,index){const p=this.ref(pos,index);if(p===null)return '';const n=this.u32(p);this.at(p+4,n);return new TextDecoder().decode(this.data.subarray(p+4,p+4+n));}
 bytes(pos,index){const p=this.ref(pos,index);if(p===null)return new Uint8Array();const n=this.u32(p);this.at(p+4,n);return this.data.subarray(p+4,p+4+n);}
}
function inspectTFLite(input){
 const f=new Flat(input);const model=f.u32(0);f.at(model,4);
 const version=f.val(model,0,'u32');if(version!==3)throw new InvalidTFLite('Unsupported TFLite schema version '+version+' (expected 3).');
 const subgraphs=f.tabs(model,2);if(subgraphs.length!==1)throw new InvalidTFLite('This importer currently supports exactly one TFLite subgraph.');
 const codes=f.tabs(model,1).map(p=>{
   const depr=f.val(p,0,'i8');const n=f.val(p,3,'i32',depr);return {id:n,name:OP_NAMES[n]||('OPCODE_'+n),version:f.val(p,2,'i32',1),custom:f.str(p,1)};
 });
 const bufs=f.tabs(model,4).map(p=>f.bytes(p,0));
 const sub=subgraphs[0];const tps=f.tabs(sub,0);
 const tensors=tps.map((p,id)=>{
   const shape=f.ints(p,0),typeId=f.val(p,1,'u8'),buffer=f.val(p,2,'u32'),name=f.str(p,3);if(buffer>=bufs.length)throw new InvalidTFLite(`Tensor ${id}: invalid buffer index`);
   const q=f.ref(p,4),quant=q===null?null:{scales:f.floats(q,2),zero_points:f.longs(q,3),axis:f.val(q,6,'i32'),min:f.floats(q,0),max:f.floats(q,1)};
   return {id,name,shape,dtype:DTYPES[typeId]||('TYPE_'+typeId),buffer,parameter_bytes:bufs[buffer].length,quantization:quant};
 });
 const inputs=f.ints(sub,1),outputs=f.ints(sub,2);
 function tensor(i){if(!Number.isInteger(i)||i<0||i>=tensors.length)throw new InvalidTFLite('Invalid tensor index '+i);return tensors[i];}
 [...inputs,...outputs].forEach(tensor);
 const ops=f.tabs(sub,3).map((p,id)=>{
   const opcode=f.val(p,0,'u32');if(opcode>=codes.length)throw new InvalidTFLite('Invalid operator code index');
   const code=codes[opcode],ins=f.ints(p,1),outs=f.ints(p,2);for(const i of [...ins,...outs])if(i!==-1)tensor(i);
   const o=f.ref(p,4),cfg={};
   if(o!==null && code.name==='FULLY_CONNECTED') {cfg.activation=ACTIVATIONS[f.val(o,0,'u8')]||'UNKNOWN';cfg.weights_format=f.val(o,1,'u8');cfg.keep_num_dims=!!f.val(o,2,'u8');}
   if(o!==null && code.name==='CONV_2D'){cfg.padding=f.val(o,0,'u8');cfg.stride_w=f.val(o,1,'i32',1);cfg.stride_h=f.val(o,2,'i32',1);cfg.activation=ACTIVATIONS[f.val(o,3,'u8')]||'UNKNOWN';cfg.dilation_w=f.val(o,4,'i32',1);cfg.dilation_h=f.val(o,5,'i32',1);}
   if(o!==null && code.name==='DEPTHWISE_CONV_2D'){cfg.padding=f.val(o,0,'u8');cfg.stride_w=f.val(o,1,'i32',1);cfg.stride_h=f.val(o,2,'i32',1);cfg.depth_multiplier=f.val(o,3,'i32',1);cfg.activation=ACTIVATIONS[f.val(o,4,'u8')]||'UNKNOWN';}
   if(o!==null && ['MAX_POOL_2D','AVERAGE_POOL_2D'].includes(code.name)){
     cfg.padding=f.val(o,0,'u8');cfg.stride_w=f.val(o,1,'i32',1);cfg.stride_h=f.val(o,2,'i32',1);
     cfg.filter_w=f.val(o,3,'i32',1);cfg.filter_h=f.val(o,4,'i32',1);cfg.activation=ACTIVATIONS[f.val(o,5,'u8')]||'UNKNOWN';
   }
   if(o!==null && ['SUB','MUL','ADD'].includes(code.name))cfg.activation=ACTIVATIONS[f.val(o,0,'u8')]||'UNKNOWN';
   if(o!==null && code.name==='MEAN')cfg.keep_dims=!!f.val(o,0,'u8');
   if(o!==null && code.name==='SOFTMAX'){cfg.beta=f.val(o,0,'f32',1);}
   if(o!==null && code.name==='CONCATENATION'){cfg.axis=f.val(o,0,'i32');cfg.activation=ACTIVATIONS[f.val(o,1,'u8')]||'UNKNOWN';}
   return {id,op:code.name,opcode:code.id,version:code.version,inputs:ins,outputs:outs,config:cfg,
    parameter_tensors:ins.slice(1).filter(i=>i>=0&&bufs[tensor(i).buffer].length>0)};
 });
 const activeInputDtypes=inputs.map(i=>tensor(i).dtype);
 return {schema_version:version,subgraph_name:f.str(sub,4),description:f.str(model,3),inputs,outputs,tensors,operators:ops,bytes:f.data.length,input_dtypes:activeInputDtypes,operator_codes:codes};
}
function extractTensorData(input, tensorId) {
 const f=new Flat(input); const model=f.u32(0); const sgs=f.tabs(model,2);
 if(sgs.length!==1)throw new InvalidTFLite('Exactly one subgraph required');
 const tps=f.tabs(sgs[0],0);
 if(!Number.isInteger(tensorId)||tensorId<0||tensorId>=tps.length)throw new InvalidTFLite('Invalid tensor ID');
 const idx=f.val(tps[tensorId],2,'u32'); const buffers=f.tabs(model,4);
 if(idx>=buffers.length)throw new InvalidTFLite('Invalid tensor buffer ID');
 return new Uint8Array(f.bytes(buffers[idx],0));
}
function base64Encode(bytes){let parts=[];const chunk=0x6000;for(let i=0;i<bytes.length;i+=chunk){parts.push(String.fromCharCode.apply(null,bytes.subarray(i,i+chunk)));}return btoa(parts.join(''));}
function base64Decode(s){const raw=atob(s);if(raw.length>MAX_BYTES)throw new InvalidTFLite('Embedded model exceeds 64 MiB limit');const out=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)out[i]=raw.charCodeAt(i);return out;}
// Synchronous SHA-256 fallback makes the project checksum available in offline/file:// browsers.
function sha256Fallback(data){
 const K=[0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2];
 const H=[0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19];
 const len=data.length,padded=Math.ceil((len+9)/64)*64;
 const b=new Uint8Array(padded);b.set(data);b[len]=0x80;
 const dv=new DataView(b.buffer);const bits=len*8;
 dv.setUint32(padded-8,Math.floor(bits/4294967296),false);dv.setUint32(padded-4,bits>>>0,false);
 const rotr=(x,n)=>(x>>>n)|(x<<(32-n));const w=new Int32Array(64);
 for(let off=0;off<padded;off+=64){
  for(let i=0;i<16;i++)w[i]=dv.getInt32(off+i*4,false);
  for(let i=16;i<64;i++){
   const a=w[i-15],c=w[i-2],s0=rotr(a,7)^rotr(a,18)^(a>>>3),s1=rotr(c,17)^rotr(c,19)^(c>>>10);
   w[i]=(w[i-16]+s0+w[i-7]+s1)|0;
  }
  let [a,c,d,e,f,g,h,j]=H;
  for(let i=0;i<64;i++){
   const s1=rotr(f,6)^rotr(f,11)^rotr(f,25),choose=(f&g)^(~f&h);
   const t1=(j+s1+choose+K[i]+w[i])|0;
   const s0=rotr(a,2)^rotr(a,13)^rotr(a,22),major=(a&c)^(a&d)^(c&d);
   const t2=(s0+major)|0;
   j=h;h=g;g=f;f=(e+t1)|0;e=d;d=c;c=a;a=(t1+t2)|0;
  }
  [a,c,d,e,f,g,h,j].forEach((v,i)=>{H[i]=(H[i]+v)|0;});
 }
 return H.map(x=>(x>>>0).toString(16).padStart(8,'0')).join('');
}
async function sha256(bytes){if(root.crypto?.subtle){const hash=await root.crypto.subtle.digest('SHA-256',bytes);return Array.from(new Uint8Array(hash)).map(v=>v.toString(16).padStart(2,'0')).join('');}return sha256Fallback(bytes);}

const api={inspectTFLite,extractTensorData,base64Encode,base64Decode,sha256,InvalidTFLite};
if(typeof module!=='undefined'&&module.exports)module.exports=api;
root.NoodleTFLite=api;
})(typeof globalThis!=='undefined'?globalThis:this);
