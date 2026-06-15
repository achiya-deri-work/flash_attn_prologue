# Copyright (c) 2026 Achiya Deri.
# All rights reserved.

import copy
import math
from typing import Dict, List, Any, Callable, Tuple, Optional

class CoordinateDescentAutotuner:
    """
    A generic, highly Pythonic Coordinate Descent Autotuner.
    
    Optimizes a configuration dictionary by tuning one parameter (coordinate)
    at a time, holding all other parameters constant. The cycle repeats until
    no parameters change during a full sweep (convergence) or the maximum
    number of iterations is reached.
    
    Attributes:
        param_space: A dictionary mapping hyperparameter names to a list of candidate values
                     or a specification dictionary for dynamic stepping.
                     Insertion order of keys dictates the tuning order.
        objective_fn: A callable that accepts a configuration dict and returns a float score
                      representing the performance metric (e.g. latency in milliseconds).
        is_valid_fn: A callable that accepts a configuration dict and returns True if it's
                     physically valid (e.g. fits SMEM, warp layout bounds, etc.), False otherwise.
        maximize: If True, the autotuner aims to maximize the objective function score.
                  If False, it minimizes the score (e.g. minimizing latency).
        max_iterations: Maximum number of coordinate descent cycles to run.
        verbose: If True, prints verbose log messages.
    """
    def __init__(
        self,
        param_space: Dict[str, Any],
        objective_fn: Callable[[Dict[str, Any]], float],
        is_valid_fn: Callable[[Dict[str, Any]], bool] = lambda _: True,
        maximize: bool = False,
        max_iterations: int = 3,
        verbose: bool = True
    ):
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.is_valid_fn = is_valid_fn
        self.direction = -1.0 if maximize else 1.0  # -1.0 for maximizing, 1.0 for minimizing
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Cache to prevent redundant compilation and profiling
        self.eval_cache: Dict[Tuple[Tuple[str, Any], ...], float] = {}

    def _hash_config(self, config: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
        """Convert a configuration dict to a sorted, hashable tuple of key-value pairs, converting lists/dicts to tuples recursively."""
        def make_hashable(val):
            if isinstance(val, list):
                return tuple(make_hashable(x) for x in val)
            if isinstance(val, dict):
                return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
            return val
        return tuple(sorted((k, make_hashable(v)) for k, v in config.items() if k in self.param_space))

    def evaluate(self, config: Dict[str, Any]) -> float:
        """Evaluates a configuration and returns its score, leveraging the cache if possible."""
        key = self._hash_config(config)
        if key in self.eval_cache:
            return self.eval_cache[key]
            
        try:
            score = self.objective_fn(config)
        except Exception as e:
            if self.verbose:
                print(f"[CoordinateDescentAutotuner] Evaluation failed for config {config}: {e}")
            score = float('inf') * self.direction
            
        self.eval_cache[key] = score
        return score

    def _get_candidates(self, param: str, current_config: Dict[str, Any]) -> List[Any]:
        """Generates candidate neighbor values for the parameter dynamically or statically."""
        space = self.param_space.get(param)
        if space is None:
            return []
            
        # 1. Backward compatibility: if it is a list/tuple/set, treat it as static
        if isinstance(space, (list, tuple, set)):
            return list(space)
            
        # 2. Dynamic parameter space description
        if not isinstance(space, dict):
            raise TypeError(f"Parameter space for '{param}' must be a list/tuple or a dict, got {type(space)}")
            
        step_type = space.get("type", "static")
        min_val = space.get("min", 1)
        max_val = space.get("max", float("inf"))
        current_val = current_config[param]
        
        # If static is specified via dict
        if step_type == "static":
            return list(space.get("values", []))
            
        # If custom callback is specified
        if step_type == "custom":
            fn = space.get("fn")
            if fn is None or not callable(fn):
                raise ValueError(f"Custom step type for parameter '{param}' requires a callable 'fn' in the spec")
            candidates = list(fn(current_config))
        else:
            candidates = []
            
            # Generate raw candidate values based on step_type
            if step_type == "power_of_2":
                if isinstance(current_val, (int, float)):
                    candidates.extend([current_val * 2, current_val // 2])
            elif step_type == "step_1":
                if isinstance(current_val, (int, float)):
                    candidates.extend([current_val + 1, current_val - 1])
            elif step_type in ("both", "power_of_2_and_step_1"):
                if isinstance(current_val, (int, float)):
                    candidates.extend([current_val * 2, current_val // 2, current_val + 1, current_val - 1])
            elif step_type == "maxrregcount":
                # Values are of the form 2^p - 1. E.g. 127, 255.
                if isinstance(current_val, (int, float)) and current_val > 0:
                    p = int(round(math.log2(current_val + 1)))
                    candidates.extend([2**(p+1) - 1, 2**(p-1) - 1])
                else:
                    # Fallback if invalid or <= 0
                    candidates.extend([127, 255])
            else:
                raise ValueError(f"Unknown step type '{step_type}' for parameter '{param}'")
            
        # Filter and clamp candidates
        filtered_candidates = []
        for val in candidates:
            # For list/tuples/None/other non-numeric types, skip clamping
            if val is None or not isinstance(val, (int, float)):
                if val != current_val and val not in filtered_candidates:
                    filtered_candidates.append(val)
                continue
                
            # Clamp to min and max bounds
            clamped = val
            if clamped < min_val:
                clamped = min_val
            elif clamped > max_val:
                clamped = max_val
                
            # If current_val is integer, ensure clamped is integer
            if isinstance(current_val, int):
                clamped = int(clamped)
                
            # Ensure candidate is valid and different from current_val
            if clamped != current_val and clamped not in filtered_candidates:
                filtered_candidates.append(clamped)
                
        return filtered_candidates

    def tune(self, initial_config: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """
        Runs the coordinate descent search starting from a valid initial_config.
        
        Args:
            initial_config: A dictionary representing the starting configuration point.
                            Must be valid according to `is_valid_fn`.
                            
        Returns:
            A tuple of (best_config, best_score)
        """
        current_config = copy.deepcopy(initial_config)
        
        if not self.is_valid_fn(current_config):
            raise ValueError(f"Initial configuration is invalid: {current_config}")
            
        best_score = self.evaluate(current_config)
        if self.verbose:
            print(f"[CoordinateDescentAutotuner] Starting tuning...")
            print(f"[CoordinateDescentAutotuner] Initial Configuration: {current_config}")
            print(f"[CoordinateDescentAutotuner] Initial Score: {best_score:.4f}")
            
        for iteration in range(self.max_iterations):
            changed = False
            if self.verbose:
                print(f"\n--- Coordinate Descent Cycle {iteration + 1} ---")
                
            # Optimize each coordinate (parameter) sequentially
            for param in self.param_space.keys():
                local_best_score = best_score
                
                # Test candidate values for this coordinate.
                # Walk greedily along this dimension as long as there is strict improvement.
                improved_local = True
                local_steps = 0
                max_local_steps = 20 # Safety bound to prevent infinite loops
                
                while improved_local and local_steps < max_local_steps:
                    improved_local = False
                    candidates = self._get_candidates(param, current_config)
                    best_val = current_config[param]
                    
                    for val in candidates:
                        if val == current_config[param]:
                            continue
                            
                        candidate = copy.deepcopy(current_config)
                        candidate[param] = val
                        
                        if not self.is_valid_fn(candidate):
                            continue
                            
                        score = self.evaluate(candidate)
                        
                        # Check if this choice improved performance
                        if score * self.direction < local_best_score * self.direction:
                            local_best_score = score
                            best_val = val
                            
                    if best_val != current_config[param]:
                        if self.verbose:
                            print(
                                f"[CoordinateDescentAutotuner] Param '{param}' changed: "
                                f"{current_config[param]} -> {best_val} | "
                                f"Score: {local_best_score:.4f}"
                            )
                        current_config[param] = best_val
                        best_score = local_best_score
                        changed = True
                        improved_local = True
                        local_steps += 1
            
            # If a full cycle of parameters yields no changes, the algorithm has converged
            if not changed:
                if self.verbose:
                    print(f"[CoordinateDescentAutotuner] Converged after {iteration + 1} cycles.")
                break
                
        return current_config, best_score
