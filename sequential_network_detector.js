/* Noodle Studio shared, backend-neutral sequential graph detector.
 * Pure analysis: no platform, precision, memory, or code emission decisions.
 */
'use strict';
(function(root){
 function analyze(model,options={}){
  const fail=(reason,operatorIndex)=>({sequential:false,reason,operatorIndex});
  if(!model || !Array.isArray(model.inputs) || model.inputs.length!==1 ||
     !Array.isArray(model.outputs) || model.outputs.length!==1 ||
     !Array.isArray(model.operators) || model.operators.length===0)
   return fail('Requires one model input, one model output, and a nonempty operator graph',-1);
  const ids=[model.inputs[0]];
  const seen=new Set(ids);
  for(let i=0;i<model.operators.length;i++){
   const op=model.operators[i];
   if(!op || !Array.isArray(op.inputs) || !Array.isArray(op.outputs) || op.outputs.length!==1)
    return fail('Each operator must produce exactly one activation',i);
   const parameters=new Set(op.parameter_tensors||[]);
   const active=op.inputs.filter(id=>Number.isInteger(id)&&id>=0&&!parameters.has(id));
   if(active.length!==1 || active[0]!==ids[ids.length-1])
    return fail('Operator does not exclusively consume the previous activation',i);
   const out=op.outputs[0];
   if(!Number.isInteger(out)||out<0||seen.has(out))
    return fail('Output aliases or reuses an earlier activation tensor',i);
   seen.add(out);ids.push(out);
  }
  if(ids[ids.length-1]!==model.outputs[0])return fail('Final activation is not the model output',model.operators.length);
  if(options.storage){
   if(options.storage.length!==seen.size ||
      options.storage.some(s=>!seen.has(s.tensor_id)) ||
      new Set(options.storage.map(s=>s.tensor_id)).size!==seen.size)
    return fail('Storage declarations must match the activation chain exactly',-1);
  }
  return {sequential:true,tensorIds:ids,operatorCount:model.operators.length};
 }
 const api={analyze};
 root.NoodleSequentialNetwork=api;
 if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof globalThis!=='undefined'?globalThis:this);
