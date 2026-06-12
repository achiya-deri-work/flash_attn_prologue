import flash_attn_prologue
import torch
from mamba3 import Mamba3SingleStepUpdate, init_mamba_state
from cute.cute_mamba3 import cute_mamba3

def test_mamba3_forward_backward_parity():
    # Setup test shapes
    B = 4
    S = 8
    NH = 2
    SD = 64
    HD = 64
    MR = 4
    D = NH * HD
    device = "cuda"

    # Initialize PyTorch module
    model = Mamba3SingleStepUpdate(
        hidden_dim=D,
        num_heads=NH,
        head_dim=HD,
        state_dim=SD,
        mimo_rank=MR,
    ).to(device)
    
    # Extract weights in FP32
    W_X = model.W_X.clone().detach().float()
    W_Z = model.W_Z.clone().detach().float()
    W_O = model.W_O.clone().detach().float()
    D_skip = model.D_skip.clone().detach().float()

    # Generate random inputs in FP32
    x_raw_3d = torch.randn((B, S, NH * HD), device=device, dtype=torch.float32, requires_grad=True)
    z_raw_3d = torch.randn((B, S, NH * HD), device=device, dtype=torch.float32, requires_grad=True)
    x_raw = x_raw_3d.view(B, S, NH, HD)
    z_raw = z_raw_3d.view(B, S, NH, HD)
    B_rot = torch.randn((B, S, NH, SD, MR), device=device, dtype=torch.float32, requires_grad=True)
    C_rot = torch.randn((B, S, NH, SD, MR), device=device, dtype=torch.float32, requires_grad=True)
    dt_raw = torch.randn((B, S, NH), device=device, dtype=torch.float32)
    A_raw = torch.randn((B, S, NH), device=device, dtype=torch.float32)
    lam_raw = torch.randn((B, S, NH), device=device, dtype=torch.float32)
    
    # Process dt, A, lam to match model logic
    dt = torch.nn.functional.softplus(dt_raw + model.dt_bias.to(device)).detach().requires_grad_(True)
    A = -torch.clamp(torch.nn.functional.softplus(A_raw), min=model.A_floor).detach().requires_grad_(True)
    lam = torch.sigmoid(lam_raw).detach().requires_grad_(True)
    dt.retain_grad()
    A.retain_grad()
    lam.retain_grad()

    # Initialize states in FP32
    ssm_state, trap_mem, _ = init_mamba_state(
        batch_size=B,
        seq_len=S,
        num_heads=NH,
        state_dim=SD,
        head_dim=HD,
        device=device,
        dtype=torch.float32
    )
    ssm_state.requires_grad = True
    trap_mem.requires_grad = True

    # --- 1. Naive Eager PyTorch Forward Pass ---
    # Factored curr_V: [B, S, H, N, R] @ [1, 1, H, R, P] -> [B, S, H, N, P]
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
    
    # The output from the rest of the model is in BFloat16, matching user spec
    out_flat_eager = out_gated_prime_eager.to(torch.bfloat16)

    # --- 2. Custom CuTe Kernel Forward Pass ---
    out_flat_cute_3d, curr_state_cute, curr_V_cute = cute_mamba3(
        x_raw_3d, z_raw_3d, B_rot, C_rot, dt, A, lam, ssm_state, trap_mem,
        W_X, W_Z, W_O, D_skip
    )
    out_flat_cute = out_flat_cute_3d.view(B, S, NH, HD)

    # --- 3. Forward Parity Assertions ---
    print("Verifying Forward Parity...")
    print("out_flat_cute nan count:", torch.isnan(out_flat_cute).sum().item())
    print("out_flat_eager nan count:", torch.isnan(out_flat_eager).sum().item())
    print("curr_state_cute nan count:", torch.isnan(curr_state_cute).sum().item())
    print("curr_state_eager nan count:", torch.isnan(curr_state_eager).sum().item())
    try:
        # Tolerances adjusted for bfloat16 output
        torch.testing.assert_close(out_flat_cute, out_flat_eager, rtol=1e-2, atol=1e-2)
        # Tolerances adjusted for TF32 / FP32 state spaces
        torch.testing.assert_close(curr_state_cute, curr_state_eager, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(curr_V_cute, curr_V_eager, rtol=1e-3, atol=1e-3)
    except AssertionError as e:
        print("Mismatched output details:")
        mismatch_mask = ~torch.isclose(out_flat_cute, out_flat_eager, rtol=1e-2, atol=1e-2)
        mismatch_indices = torch.nonzero(mismatch_mask)
        print(f"First 10 mismatches out of {mismatch_indices.shape[0]}:")
        for idx in mismatch_indices[:10]:
            t = tuple(idx.tolist())
            print(f"Index {t}: Cute = {out_flat_cute[t].item()}, Eager = {out_flat_eager[t].item()}")
        raise e
    print("Forward Parity Passed!")

    # --- 4. Backward Pass Parity ---
    print("Verifying Backward Parity...")
    # Loss gradient of BFloat16 output
    grad_out = torch.randn_like(out_flat_eager)
    
    # Eager backward
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
    
    # Zero gradients for custom kernel backward run
    x_raw_3d.grad.zero_()
    z_raw_3d.grad.zero_()
    B_rot.grad.zero_()
    C_rot.grad.zero_()
    dt.grad.zero_()
    A.grad.zero_()
    lam.grad.zero_()
    ssm_state.grad.zero_()
    trap_mem.grad.zero_()

    # CuTe backward
    out_flat_cute_3d.backward(grad_out.view(B, S, NH * HD))
    
    # Verify backward parity
    torch.testing.assert_close(x_raw_3d.grad, x_raw_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(z_raw_3d.grad, z_raw_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(B_rot.grad, B_rot_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(C_rot.grad, C_rot_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(dt.grad, dt_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(A.grad, A_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(lam.grad, lam_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(ssm_state.grad, ssm_state_grad_eager, rtol=2e-2, atol=5e-2)
    torch.testing.assert_close(trap_mem.grad, trap_mem_grad_eager, rtol=2e-2, atol=5e-2)

    print("Backward Parity Passed!")

if __name__ == "__main__":
    test_mamba3_forward_backward_parity()
