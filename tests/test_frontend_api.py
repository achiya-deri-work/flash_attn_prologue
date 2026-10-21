"""Focused frontend qualification; CPU default, set PROLOGUE_TEST_DEVICE=cuda."""
import os
import json
from pathlib import Path
import math
import sys
import torch
import pytest
from flash_attn_prologue import RMSNormRoPEScale, qknorm_rope_scale, prepare_backend
DEVICE=os.environ.get('PROLOGUE_TEST_DEVICE','cpu')


def oracle(q,k,w,s,sin,cos,eps=1e-5):
    # Independent row-wise RMS via vector norm and explicit 2x2 rotation matrices.
    packed=q.ndim==3
    if packed:
        shape=q.shape; q=q.transpose(0,1)[None]; k=k.transpose(0,1)[None]
        if sin.ndim==3: sin=sin[:,0]; cos=cos[:,0]
    if sin.ndim==2: sin=sin[None,None]; cos=cos[None,None]
    elif sin.ndim==3: sin=sin[:,None]; cos=cos[:,None]
    rot=torch.stack((cos,-sin,sin,cos),-1).reshape(*cos.shape,2,2).double()
    outputs=[]
    for i,x in enumerate((q,k)):
        xd=x.double(); rms=torch.sqrt(torch.linalg.vector_norm(xd,dim=-1,keepdim=True)**2 / x.shape[-1]+eps)
        weighted=xd/rms*w[i].double()
        pairs=torch.stack(weighted.chunk(2,-1),-1)
        rotated=torch.matmul(rot,pairs[...,None]).squeeze(-1)
        y=torch.cat((rotated[...,0],rotated[...,1]),-1)
        if i==0:y=y*s.double().exp()[None]
        if packed:y=y[0].transpose(0,1)
        outputs.append(y.to(x.dtype))
    return tuple(outputs)


def inputs(t=65,d=32,packed=True,dtype=torch.bfloat16,noncontig=False,table_grad=False):
    torch.manual_seed(832)
    shape=(t,4,d) if packed else (2,4,t,d)
    def activation():
        x=torch.randn((*shape[:-1],d*2) if noncontig else shape,device=DEVICE,dtype=dtype)
        if noncontig:x=x[...,::2]
        return x.detach().requires_grad_()
    q,k=activation(),activation()
    w=(1+torch.randn(2,1,d,device=DEVICE)*.2).requires_grad_()
    s=(torch.randn(4,1,1,device=DEVICE)*.2-math.log(d)/2).requires_grad_()
    angles=torch.randn((t,d//2) if packed else (2,t,d//2),device=DEVICE)
    sin,cos=angles.sin().requires_grad_(table_grad),angles.cos().requires_grad_(table_grad)
    return q,k,w,s,sin,cos


def compare(args,backend,compiled=False):
    ref_args=tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in args)
    call=lambda *a:qknorm_rope_scale(*a,backend=backend)
    if compiled:call=torch.compile(call,fullgraph=True,dynamic=False)
    out=call(*args); ref=oracle(*ref_args)
    probes=tuple(torch.randn_like(y.transpose(-1,-2)).transpose(-1,-2) for y in out)
    grads=torch.autograd.grad(out,[x for x in args if x.requires_grad],probes)
    rgrads=torch.autograd.grad(ref,[x for x in ref_args if x.requires_grad],probes)
    metrics={}
    for name,a,b in zip(('q','k','dq','dk','dw','ds','dsin','dcos'),out+grads,ref+rgrads):
        assert torch.isfinite(a).all(),name
        # BF16 output/gradient rounding; FP32 params can accumulate rounded upstream activations.
        torch.testing.assert_close(a,b,atol=5e-4 if a.dtype==torch.float32 else .01,
                                   rtol=1e-4 if a.dtype==torch.float32 else .01,
                                   msg=lambda m:name+': '+m)
        delta=(a.float()-b.float()); rms=b.float().square().mean().sqrt()
        if a.numel():
            nrms=(delta.square().mean().sqrt()/rms.clamp_min(1e-8)).item()
            metrics[name]={"max_abs":delta.abs().max().item(),"normalized_rms":nrms}
            if rms>1e-5: assert nrms<.001,name
    if os.environ.get('PROLOGUE_METRICS'):
        with open(os.environ['PROLOGUE_METRICS'],'a') as f:
            f.write(json.dumps({"backend":backend,"compiled":compiled,"shape":list(args[0].shape),"metrics":metrics})+'\n')
    return out


@pytest.mark.parametrize('t,d,packed,noncontig',[(0,32,True,False),(1,32,True,False),(63,32,True,False),(64,32,True,False),(65,32,True,True),(95,64,False,False),(96,64,True,False),(127,32,False,True),(128,64,True,False),(129,32,True,False)])
def test_native(t,d,packed,noncontig):
    compare(inputs(t,d,packed,noncontig=noncontig,table_grad=True),'native')


@pytest.mark.parametrize('backend',['native','cute'])
@pytest.mark.parametrize('t,d,packed',[(1,32,True),(63,32,True),(64,32,True),(65,32,True),(95,64,True),(96,64,False),(127,32,False),(128,64,True),(129,32,True),(6400,32,True)])
def test_cuda(backend,t,d,packed):
    if DEVICE!='cuda':pytest.skip('CUDA qualification only')
    prepare_backend(backend)
    compare(inputs(t,d,packed),backend)


@pytest.mark.parametrize('backend',['native','cute'])
def test_compile(backend):
    if backend=='cute' and DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend(backend)
    compare(inputs(65,32,True,table_grad=backend=='native'),backend,True)


def test_master_precision_and_scale():
    module=RMSNormRoPEScale(4,32,initial_scale=1.0).to(DEVICE)
    with torch.no_grad():module.rmsnorm_weight.uniform_(.5,1.5);module.scales.uniform_(-1,1)
    before=[p.clone() for p in module.parameters()]
    module.bfloat16()
    for p,b in zip(module.parameters(),before): assert p.dtype==torch.float32;torch.testing.assert_close(p,b,rtol=0,atol=0)
    module.bfloat().reset_parameters()
    assert torch.count_nonzero(module.scales)==0
    assert set(module.state_dict())=={'rmsnorm_weight','scales'}


def test_validation():
    a=list(inputs())
    with pytest.raises(ValueError,match='FP32'):qknorm_rope_scale(*a[:2],a[2].bfloat16(),*a[3:])
    with pytest.raises(ValueError,match='half-width'):qknorm_rope_scale(*a[:4],a[4].repeat(1,2),a[5].repeat(1,2))
    with pytest.raises(ValueError,match='eps'):qknorm_rope_scale(*a,eps=0)
    with pytest.raises(ValueError,match='backend'):qknorm_rope_scale(*a,backend='auto')


def test_cpu_import_has_no_cuda_side_effects():
    import subprocess
    subprocess.run([sys.executable,'-c',"import sys,os; before=dict(os.environ); import flash_attn_prologue; assert 'cutlass' not in sys.modules; assert 'quack' not in sys.modules; assert before==dict(os.environ)"],check=True)


def test_optimizer_checkpoint_autocast():
    from torch.utils.checkpoint import checkpoint
    module=RMSNormRoPEScale(4,32,initial_scale=1).to(DEVICE)
    args=inputs(65)
    optim=torch.optim.AdamW(module.parameters(),lr=.001)
    for _ in range(3):
        optim.zero_grad(set_to_none=True)
        with torch.autocast(DEVICE,dtype=torch.bfloat16):
            outputs=checkpoint(module,args[0],args[1],args[4],args[5],use_reentrant=False)
            loss=sum(x.float().square().mean() for x in outputs)
        loss.backward();optim.step()
    clone=RMSNormRoPEScale(4,32,initial_scale=1).to(DEVICE)
    clone.load_state_dict(module.state_dict());clone.bfloat16()
    for a,b in zip(module.parameters(),clone.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_checkpoint_assign_fp32():
    m=RMSNormRoPEScale(4,32)
    state={k:torch.randn_like(v).bfloat16() for k,v in m.state_dict().items()}
    m.load_state_dict(state,assign=True)
    for name,p in m.named_parameters():
        assert p.dtype==torch.float32
        torch.testing.assert_close(p,state[name].float(),rtol=0,atol=0)


@pytest.mark.parametrize('backend',['native','cute'])
def test_noncontiguous_and_offset(backend):
    if backend=='cute' and DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend(backend)
    a=list(inputs(65,noncontig=True))
    for i in range(6):
        original=a[i]
        storage=torch.empty(original.numel()+1,device=DEVICE,dtype=original.dtype)
        value=storage[1:].view_as(original)
        value.copy_(original.detach())
        a[i]=value.requires_grad_(original.requires_grad)
    compare(tuple(a),backend)


@pytest.mark.parametrize('backend',['native','cute'])
def test_single_output_gradient(backend):
    if backend=='cute' and DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend(backend)
    a=inputs(65); out=qknorm_rope_scale(*a,backend=backend)
    out[0].float().sum().backward()
    assert a[0].grad is not None
    assert a[1].grad is None or torch.count_nonzero(a[1].grad)==0


def test_cute_opcheck():
    if DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend('cute')
    a=inputs(65,packed=False)
    result=torch.library.opcheck(torch.ops.flash_prologue.qknorm_rope_scale_cute.default,
        (*a,1e-5),test_utils=('test_schema','test_autograd_registration','test_faketensor','test_aot_dispatch_dynamic'))
    assert set(result.values())=={'SUCCESS'}


@pytest.mark.parametrize('backend',['native','cute'])
def test_compiled_repeated_geometry_and_memory(backend):
    if DEVICE!='cuda':pytest.skip('CUDA qualification only')
    module=RMSNormRoPEScale(4,32,backend=backend,initial_scale=1).to(DEVICE)
    count=[]
    def counter(gm,example_inputs):
        count.append(1)
        return torch._inductor.compile(gm,example_inputs)
    fn=torch.compile(module,fullgraph=True,dynamic=False,backend=counter)
    a=inputs(129)
    def step():
        module.zero_grad(set_to_none=True)
        a[0].grad=None;a[1].grad=None
        # Runtime positions change; the same static buffers and full graph remain valid.
        a[4].copy_(torch.randn_like(a[4]).sin());a[5].copy_(torch.randn_like(a[5]).cos())
        out=fn(a[0],a[1],a[4],a[5])
        sum(x.float().square().mean() for x in out).backward()
    for _ in range(3):step()
    torch.cuda.synchronize();baseline=torch.cuda.memory_allocated()
    for _ in range(10):step()
    torch.cuda.synchronize();end=torch.cuda.memory_allocated()
    assert len(count)==1
    assert end==baseline,(baseline,end)


@pytest.mark.parametrize('layout',['offset_packed','transposed_channels'])
def test_cute_opcheck_adversarial_layout(layout):
    if DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend('cute')
    a=list(inputs(17,packed=False))
    for i in (0,1):
        if layout=='offset_packed':
            storage=torch.randn(a[i].numel()+1,device=DEVICE,dtype=a[i].dtype)
            a[i]=storage[1:].view(2,17,4,32).permute(0,2,1,3).requires_grad_()
        else:
            a[i]=torch.randn(2,4,32,17,device=DEVICE,dtype=a[i].dtype).transpose(-1,-2).requires_grad_()
    compare(tuple(a),'cute')
    if layout=='transposed_channels':
        with pytest.raises(ValueError,match='channel stride'):
            torch.ops.flash_prologue.qknorm_rope_scale_cute(*a,1e-5)
        compare(tuple(a),'cute',compiled=True)
    else:
        result=torch.library.opcheck(torch.ops.flash_prologue.qknorm_rope_scale_cute.default,
            (*a,1e-5),test_utils=('test_schema','test_autograd_registration','test_faketensor','test_aot_dispatch_dynamic'))
        assert set(result.values())=={'SUCCESS'}


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_native_other_dtypes(dtype):
    compare(inputs(32,dtype=dtype,table_grad=True),'native')


def test_native_nondefault_epsilon():
    a=inputs(17,dtype=torch.float32,table_grad=True)
    actual=qknorm_rope_scale(*a,eps=.1)
    expected=oracle(*a,eps=.1)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)


def test_meta_materialization():
    m=RMSNormRoPEScale(4,32).to('meta').to_empty(device='cpu')
    m.reset_parameters()
    for p in m.parameters():assert p.device.type=='cpu' and p.dtype==torch.float32


def test_cute_misaligned_upstream_gradient():
    if DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend('cute')
    a=inputs(65)
    b=tuple(t.detach().clone().requires_grad_(t.requires_grad) for t in a)
    out=qknorm_rope_scale(*a,backend='cute');ref=oracle(*b)
    probes=[]
    for y in out:
        storage=torch.randn(y.numel()+1,device=DEVICE,dtype=y.dtype)
        probes.append(storage[1:].view_as(y))
    grads=torch.autograd.grad(out,a[:4],probes)
    reference=torch.autograd.grad(ref,b[:4],probes)
    for actual,expected in zip(grads,reference):
        torch.testing.assert_close(actual,expected,atol=.04,rtol=.025)


@pytest.mark.parametrize('backend',['native','cute'])
def test_expanded_activations(backend):
    if backend=='cute' and DEVICE!='cuda':pytest.skip('CuTe needs CUDA')
    prepare_backend(backend)
    a=list(inputs(65))
    a[0]=a[0][:,:1].detach().expand(65,4,32).requires_grad_()
    a[1]=a[1][:1].detach().expand(65,4,32).requires_grad_()
    compare(tuple(a),backend,compiled=backend=='cute')
    if backend=='cute':
        with pytest.raises(ValueError,match='expanded activations'):
            torch.ops.flash_prologue.qknorm_rope_scale_cute(
                a[0].permute(1,0,2)[None],a[1].permute(1,0,2)[None],
                a[2],a[3],a[4],a[5],1e-5)
