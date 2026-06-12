import os
import sys

sys.path.append("/home/achiya/vs_code_main/Advantage-First-RL")
sys.path.append("/home/achiya/vs_code_main/flash_attn_prologue")

# Import flash_attn_prologue first to initialize CUTE_DSL_LIBS
import flash_attn_prologue

import json
import torch
from temporal_mamba3.cute.autotuner import get_autotuned_config, CACHE_FILE as MAMBA3_CACHE_FILE

ADV_FIRST_CACHE_FILE = "/home/achiya/vs_code_main/Advantage-First-RL/advantage_first/layers/cute/autotune_cache.json"

def run():
    device = "cuda"
    
    # Run coordinate descent autotuning for P=32, N=64, R=4
    print("=== Autotuning P=32, N=64, R=4 FWD ===")
    get_autotuned_config(32, 64, 4, "fwd", device)
    print("=== Autotuning P=32, N=64, R=4 BWD ===")
    get_autotuned_config(32, 64, 4, "bwd", device)

    # Run coordinate descent autotuning for P=64, N=32, R=4
    print("=== Autotuning P=64, N=32, R=4 FWD ===")
    get_autotuned_config(64, 32, 4, "fwd", device)
    print("=== Autotuning P=64, N=32, R=4 BWD ===")
    get_autotuned_config(64, 32, 4, "bwd", device)

    # Copy cache to Advantage-First-RL
    if os.path.exists(MAMBA3_CACHE_FILE):
        with open(MAMBA3_CACHE_FILE, "r") as f:
            cache = json.load(f)
        with open(ADV_FIRST_CACHE_FILE, "w") as f:
            json.dump(cache, f, indent=4)
        print(f"Successfully copied cache to {ADV_FIRST_CACHE_FILE}")

if __name__ == "__main__":
    run()
