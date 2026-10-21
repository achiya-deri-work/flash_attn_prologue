import json, math, torch
from flash_attn_prologue import RMSNormRoPEScale
from flash_detr.kernels.window_attention import packed_window_attention

torch.manual_seed(721)
records=[]
for backend in ('native','cute'):
    module=RMSNormRoPEScale(4,32,initial_scale=1.0,backend=backend).cuda().bfloat()
    module.reset_parameters()
    raw=torch.randn(320,3,4,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    sink=torch.randn(4,32,device='cuda',dtype=torch.float32,requires_grad=True)
    shapes=torch.tensor([[8,8],[13,17]],device='cuda',dtype=torch.int32)
    cu=torch.tensor([0,64,285],device='cuda',dtype=torch.int32)
    angles=torch.zeros(320,16,device='cuda',dtype=torch.float32)
    sin,cos=angles.sin(),angles.cos()
    probe=torch.randn(320,4,32,device='cuda',dtype=torch.bfloat16)
    calls=[]
    def operation(raw,sink,cu,shapes,sin,cos):
        q,k,v=raw.unbind(1)
        q,k=module(q,k,sin,cos)
        return packed_window_attention(q,k,v,sink,cu,shapes,shift=4)
    def compiler(gm,example_inputs):
        calls.append(1)
        return torch._inductor.compile(gm,example_inputs)
    compiled=torch.compile(operation,fullgraph=True,dynamic=False,backend=compiler)
    for sizes in (((8,8),(13,17)),((9,9),(12,16))):
        counts=[h*w for h,w in sizes];total=sum(counts)
        shapes.copy_(torch.tensor(sizes,device='cuda',dtype=torch.int32))
        cu.copy_(torch.tensor([0,counts[0],total],device='cuda',dtype=torch.int32))
        position=torch.cat([torch.arange(n,device='cuda') for n in counts]+[torch.zeros(320-total,device='cuda')])
        angles=position[:,None]*torch.exp(-math.log(10000)*torch.arange(16,device='cuda')[None,:]/16)
        sin.copy_(angles.sin());cos.copy_(angles.cos())
        inputs=(raw,sink,*module.parameters())
        expected=operation(raw,sink,cu,shapes,sin,cos)
        expected_grads=torch.autograd.grad(expected,inputs,probe)
        actual=compiled(raw,sink,cu,shapes,sin,cos)
        actual_grads=torch.autograd.grad(actual,inputs,probe)
        metrics={}
        for name,a,r in zip(('output','d_raw_qkv','d_sink','d_norm_weight','d_log_scale'),(actual,*actual_grads),(expected,*expected_grads)):
            assert torch.isfinite(a).all(),name
            diff=a.float()-r.float()
            nrms=(diff.norm()/r.float().norm().clamp_min(1e-8)).item()
            cosine=torch.nn.functional.cosine_similarity(a.float().flatten(),r.float().flatten(),dim=0).item()
            torch.testing.assert_close(a,r,atol=.03,rtol=.03)
            assert nrms<.01,(backend,name,nrms)
            assert cosine>.999,(backend,name,cosine)
            metrics[name]={'max_abs':diff.abs().max().item(),'relative_l2':nrms,'cosine':cosine}
        assert torch.count_nonzero(actual[total:])==0
        assert torch.count_nonzero(actual_grads[0][total:])==0
        records.append({'backend':backend,'shapes':sizes,'metrics':metrics})
    assert len(calls)==1,(backend,len(calls))
print(json.dumps({'status':'passed','compile':'fullgraph=True,dynamic=False; one forward graph per backend across geometry changes','records':records},indent=2))
