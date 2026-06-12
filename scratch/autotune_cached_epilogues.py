import json
import os
import copy
from flash_attn_prologue.autotune import get_autotuned_config, load_cache, save_cache

def main():
    cache = load_cache()
    keys = list(cache.keys())
    
    epilogue_keys = [k for k in keys if "epilogue" in k]
    print(f"Found {len(epilogue_keys)} epilogue configurations to tune.")
    
    for key in epilogue_keys:
        print(f"\n==================================================")
        print(f"Re-tuning key: {key}")
        print(f"==================================================")
        
        # Parse the shape: M_N_K_mode
        parts = key.split("_")
        M = int(parts[0])
        N = int(parts[1])
        K = int(parts[2])
        mode = "_".join(parts[3:])
        
        initial_config = copy.deepcopy(cache[key])
        print(f"Starting configuration: {initial_config}")
        
        try:
            # Trigger autotune with initial config.
            # get_autotuned_config will pass initial_config down to profile_shapes,
            # which will run coordinate descent.
            best_config = get_autotuned_config(M, N, K, mode, device="cuda", initial_config=initial_config)
            print(f"Result config for {key}: {best_config}")
        except Exception as e:
            print(f"Error tuning {key}: {e}")
            
    print("\nAll epilogue shape re-tuning runs completed.")

if __name__ == "__main__":
    main()
