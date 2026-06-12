import pytest
import torch
import torch.nn.functional as F
from temporal_mamba3.mamba3 import Mamba3SingleStepUpdate, init_mamba_state

def test_mamba3_compiled_training_loop():
    B = 4
    S = 8
    NH = 2
    SD = 16
    HD = 64
    MR = 4
    D = NH * HD
    device = "cuda"

    # Initialize model with Cute backend
    model_cute = Mamba3SingleStepUpdate(
        hidden_dim=D,
        num_heads=NH,
        head_dim=HD,
        state_dim=SD,
        mimo_rank=MR,
        backend="cute"
    ).to(device).bfloat16()

    # Create optimizer
    optimizer = torch.optim.AdamW(model_cute.parameters(), lr=1e-3)

    # Compile the forward and loss calculation step under torch.compile with fullgraph=True
    @torch.compile(fullgraph=True)
    def train_step_compiled(curr_mems, ssm_state, trap_mem, rope_angles, target):
        out, next_state, next_V, next_angles = model_cute(curr_mems, ssm_state, trap_mem, rope_angles)
        loss = F.mse_loss(out, target)
        return loss, next_state, next_V, next_angles

    # Generate random input data
    curr_mems = torch.randn(B, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    ssm_state, trap_mem, rope_angles = init_mamba_state(
        batch_size=B, seq_len=S, num_heads=NH, state_dim=SD, head_dim=HD, device=device, dtype=torch.float32
    )
    ssm_state.requires_grad = True
    trap_mem.requires_grad = True
    target = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)

    # Warmup compilation
    print("Starting torch.compile warmup...")
    loss, next_state, next_V, next_angles = train_step_compiled(
        curr_mems, ssm_state, trap_mem, rope_angles, target
    )
    loss.backward()
    
    # Check that gradients were populated on model parameters
    for name, param in model_cute.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not accumulate gradients."
        assert not torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients."

    optimizer.step()
    optimizer.zero_grad()
    print("torch.compile warmup completed successfully!")

    # Run a few compiled training steps to observe loss convergence
    initial_loss = loss.item()
    last_loss = initial_loss
    
    # Detach states for next steps
    ssm_state = next_state.detach().requires_grad_(True)
    trap_mem = next_V.detach().requires_grad_(True)
    rope_angles = next_angles.detach()

    print(f"Step 0: Loss = {initial_loss:.6f}")

    for step in range(1, 5):
        loss, next_state, next_V, next_angles = train_step_compiled(
            curr_mems, ssm_state, trap_mem, rope_angles, target
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        current_loss = loss.item()
        print(f"Step {step}: Loss = {current_loss:.6f}")
        
        # Move state forward
        ssm_state = next_state.detach().requires_grad_(True)
        trap_mem = next_V.detach().requires_grad_(True)
        rope_angles = next_angles.detach()
        
        last_loss = current_loss

    # Verify that training is proceeding cleanly
    assert not torch.isnan(torch.tensor(last_loss)), "Training loss is NaN."
    print("Compiled training loop validation passed successfully!")

if __name__ == "__main__":
    test_mamba3_compiled_training_loop()
