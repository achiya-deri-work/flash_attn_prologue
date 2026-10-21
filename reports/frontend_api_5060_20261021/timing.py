import json,statistics,torch
from flash_attn_prologue import RMSNormRoPEScale
results=[]
for t in (64,6400):
 for backend in ('native','cute'):
  module=RMSNormRoPEScale(4,32,backend=backend,initial_scale=1).cuda()
  fn=torch.compile(module,fullgraph=True,dynamic=False)
  q=torch.randn(t,4,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
  k=torch.randn_like(q,requires_grad=True)
  angles=torch.randn(t,16,device='cuda');sin,cos=angles.sin(),angles.cos()
  dq,dk=torch.randn_like(q),torch.randn_like(k)
  def run(backward):
   module.zero_grad(set_to_none=True);q.grad=None;k.grad=None
   out=fn(q,k,sin,cos)
   if backward:torch.autograd.backward(out,(dq,dk))
  row={'tokens':t,'backend':backend}
  for backward in (False,True):
   for _ in range(5):run(backward)
   torch.cuda.synchronize()
   samples=[]
   for _ in range(30):
    # eager CUDA events; includes dispatch gaps, not CUDA graph replay.
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record();run(backward);end.record();end.synchronize();samples.append(start.elapsed_time(end))
   row['forward_backward_ms' if backward else 'forward_ms']={'median':statistics.median(samples),'min':min(samples),'stdev':statistics.stdev(samples)}
  results.append(row)
print(json.dumps(results,indent=2))
