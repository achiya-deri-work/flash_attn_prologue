import torch._inductor.config as config

print("==================================================")
print("All Inductor Config Keys:")
print("==================================================")
for key in sorted(dir(config)):
    if not key.startswith("__"):
        val = getattr(config, key)
        # Only print modules, dicts, bools, ints, floats
        if isinstance(val, (bool, int, float, str, list, dict)):
            print(f"{key}: {val}")
print("==================================================")
