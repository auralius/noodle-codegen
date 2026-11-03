/* Noodle Studio: MCU/precision-neutral lifetime-aware physical-buffer reuse.
 * Pure offline analysis. Strictly no input/output in-place aliasing: a buffer
 * cannot be recycled until AFTER the previous tensor's last-consuming operator.
 * Only the backend chooses how to allocate each physical buffer.
 */
'use strict';
(function(root){
function check(ok,why){if(!ok)throw Error('Branch buffer planner: '+why);}
function plan(model,graph,storage){
 check(model && Array.isArray(model.inputs) && Array.isArray(model.outputs) && Array.isArray(graph) && graph.length>0,'Invalid graph');
 check(Array.isArray(storage) && storage.length>0,'No tensor storage declarations');
 const entries=new Map();
 for(const [i,s] of storage.entries()){
  check(Number.isInteger(s.tensor_id) && s.tensor_id>=0 && !entries.has(s.tensor_id),'Duplicate or invalid tensor ID');
  const t=model.tensors[s.tensor_id];
  check(t && Array.isArray(t.shape) && t.shape.length>0 && t.shape.every(x=>Number.isSafeInteger(x)&&x>0),'Invalid tensor shape');
  const elementSize={INT8:1,UINT8:1,FLOAT32:4,INT32:4,INT16:2,UINT16:2}[t.dtype];
  check(elementSize,'Unsupported tensor element type '+t.dtype);
  const bytes=t.shape.reduce((v,x)=>v*x,elementSize);
  check(Number.isSafeInteger(bytes)&&bytes>0,'Invalid tensor byte count');
  check(typeof s.backend==='string' && s.backend.length,'Missing memory backend');
  entries.set(s.tensor_id,{id:s.tensor_id,name:s.name,storageIndex:i,backend:s.backend,bytes,start:null,end:null});
 }
 for(const id of model.inputs){check(entries.has(id),'Missing model input '+id);entries.get(id).start=-1;}
 const produced=new Set(model.inputs);
 for(const [i,step] of graph.entries()){
  check(step && Array.isArray(step.inputs) && step.dst && step.inputs.length>0,'Invalid operator '+i);
  const ids=step.inputs.map(x=>x.tensor_id),output=step.dst.tensor_id;
  check(ids.every(id=>produced.has(id)),'Tensor consumed before production at '+i);
  check(!produced.has(output) && entries.has(output),'Duplicate/missing output '+output+' at '+i);
  for(const id of ids){const e=entries.get(id);e.end=Math.max(e.end??-1,i);}
  entries.get(output).start=i;
  produced.add(output);
 }
 for(const id of model.outputs){check(entries.has(id)&&produced.has(id),'Missing model output '+id);const e=entries.get(id);e.end=Math.max(e.end??-1,graph.length);}
 check(produced.size===entries.size,'Unreferenced tensor storage declaration');
 for(const e of entries.values()){
  check(e.start!==null,'Missing tensor producer '+e.id);
  if(e.end===null)e.end=e.start; // unused intermediate is still live for its producing op
 }
 const items=[...entries.values()].sort((a,b)=>a.start-b.start||a.storageIndex-b.storageIndex);
 const logicalBytes=items.reduce((n,e)=>n+e.bytes,0);
 // No allocation can be reused inside the consuming op; end < next start.
 function allocate(strategy){
  const pools=[],assign=new Map();
  for(const e of items){
   const eligible=pools.filter(p=>p.backend===e.backend && p.busyUntil<e.start);
   let target=null;
   const fits=eligible.filter(p=>p.capacity>=e.bytes);
   if(fits.length){
    fits.sort(strategy==='largest-fit'
      ? (a,b)=>b.capacity-a.capacity||a.id-b.id
      : (a,b)=>a.capacity-b.capacity||a.id-b.id);
    target=fits[0];
   } else if(eligible.length){
    eligible.sort((a,b)=>b.capacity-a.capacity||a.id-b.id);
    target=eligible[0]; // reusing + growing costs less than a new allocation
   }
   if(!target){target={id:pools.length,backend:e.backend,capacity:0,busyUntil:-2,members:[]};pools.push(target);}
   target.capacity=Math.max(target.capacity,e.bytes);
   target.busyUntil=e.end;
   target.members.push(e.id);
   assign.set(e.id,target.id);
  }
  return {pools,assign,allocatedBytes:pools.reduce((n,p)=>n+p.capacity,0)};
 }
 // Compare deterministic heuristics and retain the smaller retained footprint.
 const candidates=['best-fit','largest-fit'].map(method=>({...allocate(method),method}));
 candidates.sort((a,b)=>a.allocatedBytes-b.allocatedBytes||a.pools.length-b.pools.length|| (a.method==='best-fit'?-1:1));
 const winner=candidates[0];
 // Independent overlap check (including the operator at which an input dies).
 for(let i=0;i<items.length;i++)for(let j=i+1;j<items.length;j++){
  const a=items[i],b=items[j];
  if(winner.assign.get(a.id)===winner.assign.get(b.id))
   check(a.end<b.start||b.end<a.start,'Unsafe alias '+a.name+' / '+b.name);
 }
 // Read-only birth/death diagnostic; it does NOT influence the established
 // buffer assignment. A day is an operator boundary with inclusive lifetimes:
 // day 0 = model input; day (i+1) = operator i; day (N+1) = retained output.
 // Inputs and outputs of the same operator overlap unless the operator is
 // explicitly modeled as in-place (not supported by this conservative plan).
 const backends=[...new Set(items.map(e=>e.backend))].sort();
 const maximumLivePerBackend=Object.fromEntries(backends.map(b=>[b,0]));
 const timeline=[];
 let maxLiveTensors=0,peakLiveBytes=0;
 for(let step=-1;step<=graph.length;step++){
  const live=items.filter(e=>e.start<=step && e.end>=step);
  const liveBytes=live.reduce((sum,e)=>sum+e.bytes,0);
  const backendCounts=Object.fromEntries(backends.map(b=>[b,live.filter(e=>e.backend===b).length]));
  for(const backend of backends)maximumLivePerBackend[backend]=Math.max(maximumLivePerBackend[backend],backendCounts[backend]);
  maxLiveTensors=Math.max(maxLiveTensors,live.length);
  peakLiveBytes=Math.max(peakLiveBytes,liveBytes);
  timeline.push({day:step+1,operatorIndex:step>=0&&step<graph.length?step:null,
   liveTensorCount:live.length,liveBytes,backendCounts,liveTensorIds:live.map(e=>e.id)});
 }
 const maxLiveDays=timeline.filter(d=>d.liveTensorCount===maxLiveTensors).map(d=>d.day);
 const peakLiveByteDays=timeline.filter(d=>d.liveBytes===peakLiveBytes).map(d=>d.day);
 // With disjoint backend pools, the minimum buffer count is the sum of each
 // backend's own maximum concurrency (not necessarily max global concurrency).
 const minimumPhysicalBuffers=Object.values(maximumLivePerBackend).reduce((a,b)=>a+b,0);
 check(winner.pools.length>=minimumPhysicalBuffers,'Buffer count below per-backend lower bound');
 check(winner.allocatedBytes>=peakLiveBytes,'Allocation below peak-live byte bound');
 return {
  method:'branch-liveness',heuristic:winner.method,
  logicalCount:items.length,logicalBytes,
  physicalCount:winner.pools.length,allocatedBytes:winner.allocatedBytes,
  peakLiveBytes,maxLiveTensors,maxLiveDays,peakLiveByteDays,
  maximumLivePerBackend,minimumPhysicalBuffers,
  dayConvention:'day 0 = input; days 1..N = operators 0..N-1; day N+1 = retained output; birth/death inclusive',
  timeline,
  tensorBufferIds:storage.map(s=>winner.assign.get(s.tensor_id)),
  buffers:winner.pools.map(p=>({id:p.id,backend:p.backend,bytes:p.capacity,tensorIds:p.members})),
  lifetimes:storage.map(s=>{const e=entries.get(s.tensor_id);return {tensorId:e.id,name:e.name,bufferId:winner.assign.get(e.id),backend:e.backend,bytes:e.bytes,born:e.start,lastUsed:e.end,bornDay:e.start+1,dieDay:e.end+1};})
 };
}
const api={plan};root.NoodleBranchBufferPlanner=api;
if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
