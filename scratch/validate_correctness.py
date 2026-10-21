import os
import sys
import torch

# Add paths to sys.path
sys.path.append("/home/achiya/vs_code_main/Advantage-First-RL")
sys.path.append("/home/achiya/vs_code_main/flash_attn_prologue")
sys.path.append("/home/achiya/vs_code_main/flash_attn_prologue/temporal_mamba3")

# Import flash_attn_prologue first to initialize environment
import flash_attn_prologue

from temporal_mamba3.mamba3 import Mamba3SingleStepUpdate, init_mamba_state
from temporal_mamba3.cute.cute_mamba3 import cute_mamba3

def validate_fused_epilogue():
    B = 16
    N = 64
    hidden = 256
    num_heads = 4
    head_dim = 64
    device = "cuda"
    dtype = torch.bfloat16

    print("\n" + "="*80)
    print("VALIDATING FUSED QKV + EPILOGUE KERNEL BACKWARD CORRECTNESS")
    print(f"Shapes: B={B}, N={N}, hidden={hidden}, num_heads={num_heads}, head_dim={head_dim}")
    print("="*80)

    # Inputs
    x = torch.randn(B, N, hidden, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(3 * num_heads * head_dim, hidden, device=device, dtype=dtype, requires_grad=True)
    rmsnorm_weight = torch.randn(2, 1, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    scales = torch.randn(num_heads, 1, 1, device=device, dtype=torch.float32, requires_grad=True)

    # Precompute sin/cos frequencies
    position_ids = torch.arange(N, dtype=torch.long, device=device).unsqueeze(0).expand(B, N)
    freqs = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (freqs / head_dim))
    t = position_ids.float()
    freqs_out = torch.outer(t.reshape(-1), inv_freq).reshape(B, N, -1)

    sin_val = torch.sin(freqs_out)
    cos_val = torch.cos(freqs_out)
    sin = sin_val.unsqueeze(1).contiguous()
    cos = cos_val.unsqueeze(1).contiguous()

    # Baseline Sequential
    x_seq = x.clone().detach().requires_grad_(True)
    weight_seq = weight.clone().detach().requires_grad_(True)
    rms_seq = rmsnorm_weight.clone().detach().requires_grad_(True)
    scales_seq = scales.clone().detach().requires_grad_(True)

    q_proj, k_proj, v_proj = torch.ops.flash_prologue.fused_qkv(x_seq, weight_seq)
    q_s, k_s = torch.ops.flash_prologue.qknorm_rope_scale_cute(q_proj, k_proj, rms_seq, scales_seq, sin, cos)

    loss_seq = q_s.float().sum() + k_s.float().sum() + v_proj.float().sum()
    loss_seq.backward()

    # Combined/Fused
    x_comb = x.clone().detach().requires_grad_(True)
    weight_comb = weight.clone().detach().requires_grad_(True)
    rms_comb = rmsnorm_weight.clone().detach().requires_grad_(True)
    scales_comb = scales.clone().detach().requires_grad_(True)

    q_c, k_c, v_c, _, _ = torch.ops.flash_prologue.fused_qkv_epilogue(x_comb, weight_comb, rms_comb, scales_comb, sin, cos)

    loss_comb = q_c.float().sum() + k_c.float().sum() + v_c.float().sum()
    loss_comb.backward()

    # High-precision metrics helper
    def analyze_grad_diff(grad_c, grad_s, name):
        diff = (grad_c.float() - grad_s.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        norm_s = torch.norm(grad_s.float())
        norm_c = torch.norm(grad_c.float())

        if norm_s.item() < 1e-7 and norm_c.item() < 1e-7:
            l2_err = 0.0
            cos_sim = 1.0
        else:
            l2_err = (torch.norm(grad_c.float() - grad_s.float()) / norm_s).item()
            cos_sim = torch.nn.functional.cosine_similarity(grad_c.float().flatten(), grad_s.float().flatten(), dim=0).item()

        print(f"[{name}]")
        print(f"  Max Absolute Diff: {max_abs:.8e}")
        print(f"  Mean Absolute Diff: {mean_abs:.8e}")
        print(f"  L2 Relative Error:  {l2_err:.8e}")
        print(f"  Cosine Similarity:  {cos_sim:.8f}")
        return max_abs, mean_abs, l2_err, cos_sim

    print("\n--- Backward Gradients Precision Report ---")
    x_max, x_mean, x_l2, x_cos = analyze_grad_diff(x_comb.grad, x_seq.grad, "Input Gradient dx")
    w_max, w_mean, w_l2, w_cos = analyze_grad_diff(weight_comb.grad, weight_seq.grad, "Weight Gradient dW")
    rms_max, rms_mean, rms_l2, rms_cos = analyze_grad_diff(rms_comb.grad, rms_seq.grad, "RMSNorm Weight Gradient")
    s_max, s_mean, s_l2, s_cos = analyze_grad_diff(scales_comb.grad, scales_seq.grad, "Scale Factor Gradient")

    # Assert correctness
    assert x_cos > 0.999, f"dx gradient alignment failed: cosine similarity {x_cos:.6f}"
    assert w_cos > 0.999, f"dW gradient alignment failed: cosine similarity {w_cos:.6f}"
    assert rms_cos > 0.999, f"dRMS gradient alignment failed: cosine similarity {rms_cos:.6f}"
    assert s_cos > 0.99, f"dScales gradient alignment failed: cosine similarity {s_cos:.6f}"
    print("\n=> Fused Epilogue Backward Parity Validated Successfully!")

def validate_mamba_correctness():
    B = 16
    S = 64
    NH = 4
    SD = 32
    HD = 64
    MR = 4
    D = NH * HD
    device = "cuda"

    print("\n" + "="*80)
    print("VALIDATING MAMBA-3 CUSTOM KERNEL BACKWARD CORRECTNESS")
    print(f"Shapes: B={B}, S={S}, NH={NH}, SD={SD}, HD={HD}, MR={MR}")
    print("="*80)

    # Initialize model
    model = Mamba3SingleStepUpdate(
        hidden_dim=D,
        num_heads=NH,
        head_dim=HD,
        state_dim=SD,
        mimo_rank=MR,
    ).to(device)

    # FP32 weights
    W_X = model.W_X.clone().detach().float()
    W_Z = model.W_Z.clone().detach().float()
    W_O = model.W_O.clone().detach().float()
    D_skip = model.D_skip.clone().detach().float()

    # Generate random inputs
    x_raw_3d = torch.randn((B, S, NH * HD), device=device, dtype=torch.float32, requires_grad=True)
    z_raw_3d = torch.randn((B, S, NH * HD), device=device, dtype=torch.float32, requires_grad=True)
    x_raw = x_raw_3d.view(B, S, NH, HD)
    z_raw = z_raw_3d.view(B, S, NH, HD)
    B_rot = torch.randn((B, S, NH, SD, MR), device=device, dtype=torch.float32, requires_grad=True)
    C_rot = torch.randn((B, S, NH, SD, MR), device=device, dtype=torch.float32, requires_grad=True)
    dt_raw = torch.randn((B, S, NH), device=device, dtype=torch.float32)
    A_raw = torch.randn((B, S, NH), device=device, dtype=torch.float32)
    lam_raw = torch.randn((B, S, NH), device=device, dtype=torch.float32)

    dt = torch.nn.functional.softplus(dt_raw + model.dt_bias.to(device)).detach().requires_grad_(True)
    A = -torch.clamp(torch.nn.functional.softplus(A_raw), min=model.A_floor).detach().requires_grad_(True)
    lam = torch.sigmoid(lam_raw).detach().requires_grad_(True)
    dt.retain_grad()
    A.retain_grad()
    lam.retain_grad()

    # Initialize states to random values to ensure A gradient path is not zero
    ssm_state = torch.randn((B, S, NH, SD, HD), device=device, dtype=torch.float32, requires_grad=True)
    trap_mem = torch.randn((B, S, NH, SD, HD), device=device, dtype=torch.float32, requires_grad=True)

    ssm_state.retain_grad()
    trap_mem.retain_grad()

    # Naive Eager Forward
    W_X_T = W_X.transpose(-1, -2).unsqueeze(0).unsqueeze(0)
    curr_V_eager = torch.matmul(B_rot, W_X_T) * x_raw.unsqueeze(-2)

    alpha_val = torch.exp(dt * A)
    beta_val = (1.0 - lam) * dt * alpha_val
    gamma_val = lam * dt

    alpha = alpha_val.reshape(B, S, NH, 1, 1)
    beta = beta_val.reshape(B, S, NH, 1, 1)
    gamma = gamma_val.reshape(B, S, NH, 1, 1)

    curr_state_eager = (alpha * ssm_state) + (beta * trap_mem) + (gamma * curr_V_eager)
    Y_t_eager = torch.matmul(curr_state_eager.transpose(-1, -2), C_rot)

    DX_W = (x_raw * D_skip.unsqueeze(0).unsqueeze(0)).unsqueeze(-1) * W_X.unsqueeze(0).unsqueeze(0)
    Y_skip_eager = Y_t_eager + DX_W

    Z_t_eager = z_raw.unsqueeze(-1) * W_Z.unsqueeze(0).unsqueeze(0)
    out_gated_eager = Y_skip_eager * torch.nn.functional.silu(Z_t_eager)

    W_O_5d = W_O.unsqueeze(0).unsqueeze(0)
    out_gated_prime_eager = (out_gated_eager * W_O_5d).sum(dim=-1)
    out_flat_eager = out_gated_prime_eager.to(torch.bfloat16)

    # Custom CuTe Forward
    out_flat_cute_3d, curr_state_cute, curr_V_cute = cute_mamba3(
        x_raw_3d, z_raw_3d, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
        W_X, W_Z, W_O, D_skip
    )
    out_flat_cute = out_flat_cute_3d.view(B, S, NH, HD)

    # Verify Forward Correctness
    print("\n--- Forward Pass Precision Report ---")
    diff_out = (out_flat_cute.float() - out_flat_eager.float()).abs()
    print(f"Output out: max_abs={diff_out.max().item():.8e}, mean_abs={diff_out.mean().item():.8e}")
    diff_state = (curr_state_cute - curr_state_eager).abs()
    print(f"State ssm_state: max_abs={diff_state.max().item():.8e}, mean_abs={diff_state.mean().item():.8e}")

    # Backward Pass Parity
    grad_out = torch.randn_like(out_flat_eager)
    out_flat_eager.backward(grad_out)

    # Save eager gradients
    x_raw_grad_eager = x_raw_3d.grad.clone()
    z_raw_grad_eager = z_raw_3d.grad.clone()
    B_rot_grad_eager = B_rot.grad.clone()
    C_rot_grad_eager = C_rot.grad.clone()
    dt_grad_eager = dt.grad.clone()
    A_grad_eager = A.grad.clone()
    lam_grad_eager = lam.grad.clone()
    ssm_state_grad_eager = ssm_state.grad.clone()
    trap_mem_grad_eager = trap_mem.grad.clone()

    # Zero gradients
    x_raw_3d.grad.zero_()
    z_raw_3d.grad.zero_()
    B_rot.grad.zero_()
    C_rot.grad.zero_()
    dt.grad.zero_()
    A.grad.zero_()
    lam.grad.zero_()
    ssm_state.grad.zero_()
    trap_mem.grad.zero_()

    # Custom CuTe backward
    out_flat_cute_3d.backward(grad_out.view(B, S, NH * HD))

    # Precision analysis helper
    def analyze_grad_diff(grad_c, grad_e, name):
        diff = (grad_c - grad_e).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()

        norm_e = torch.norm(grad_e)
        norm_c = torch.norm(grad_c)

        if norm_e.item() < 1e-7 and norm_c.item() < 1e-7:
            l2_err = 0.0
            cos_sim = 1.0
        else:
            l2_err = (torch.norm(grad_c - grad_e) / norm_e).item()
            cos_sim = torch.nn.functional.cosine_similarity(grad_c.flatten(), grad_e.flatten(), dim=0).item()

        print(f"[{name}]")
        print(f"  Max Absolute Diff: {max_abs:.8e}")
        print(f"  Mean Absolute Diff: {mean_abs:.8e}")
        print(f"  L2 Relative Error:  {l2_err:.8e}")
        print(f"  Cosine Similarity:  {cos_sim:.8f}")
        return max_abs, mean_abs, l2_err, cos_sim

    print("\n--- Backward Gradients Precision Report ---")
    x_max, x_mean, x_l2, x_cos = analyze_grad_diff(x_raw_3d.grad, x_raw_grad_eager, "Input Gradient dx")
    z_max, z_mean, z_l2, z_cos = analyze_grad_diff(z_raw_3d.grad, z_raw_grad_eager, "Gating Gradient dz")
    B_max, B_mean, B_l2, B_cos = analyze_grad_diff(B_rot.grad, B_rot_grad_eager, "SSM Parameter B Gradient")
    C_max, C_mean, C_l2, C_cos = analyze_grad_diff(C_rot.grad, C_rot_grad_eager, "SSM Parameter C Gradient")
    dt_max, dt_mean, dt_l2, dt_cos = analyze_grad_diff(dt.grad, dt_grad_eager, "Step size dt Gradient")
    A_max, A_mean, A_l2, A_cos = analyze_grad_diff(A.grad, A_grad_eager, "Stability A Gradient")
    lam_max, lam_mean, lam_l2, lam_cos = analyze_grad_diff(lam.grad, lam_grad_eager, "Sigmoid lam Gradient")
    ssm_max, ssm_mean, ssm_l2, ssm_cos = analyze_grad_diff(ssm_state.grad, ssm_state_grad_eager, "State ssm_state Gradient")
    trap_max, trap_mean, trap_l2, trap_cos = analyze_grad_diff(trap_mem.grad, trap_mem_grad_eager, "Trap ssm_trap Gradient")

    # Assert correctness
    assert x_cos > 0.999, f"dx gradient alignment failed: cosine similarity {x_cos:.6f}"
    assert z_cos > 0.999, f"dz gradient alignment failed: cosine similarity {z_cos:.6f}"
    assert B_cos > 0.999, f"dB gradient alignment failed: cosine similarity {B_cos:.6f}"
    assert C_cos > 0.999, f"dC gradient alignment failed: cosine similarity {C_cos:.6f}"
    assert dt_cos > 0.999, f"ddt gradient alignment failed: cosine similarity {dt_cos:.6f}"
    assert A_cos > 0.999, f"dA gradient alignment failed: cosine similarity {A_cos:.6f}"
    assert lam_cos > 0.999, f"dlam gradient alignment failed: cosine similarity {lam_cos:.6f}"
    assert ssm_cos > 0.999, f"dssm gradient alignment failed: cosine similarity {ssm_cos:.6f}"
    assert trap_cos > 0.999, f"dtrap gradient alignment failed: cosine similarity {trap_cos:.6f}"
    print("\n=> Mamba-3 Backward Parity Validated Successfully!")

if __name__ == "__main__":
    validate_fused_epilogue()
    validate_mamba_correctness()
