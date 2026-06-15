import unittest
import math
from flash_attn_prologue import CoordinateDescentAutotuner

class TestCoordinateDescentAutotuner(unittest.TestCase):
    def test_mock_tuning(self):
        # We want to minimize a mock mathematical loss function:
        # loss = (x - 2)^2 + (y - 6)^2 + (z - 10)^2
        # Global optimum is at x=2, y=6, z=10.
        evals_count = 0
        
        def mock_objective(config):
            nonlocal evals_count
            evals_count += 1
            x = config["x"]
            y = config["y"]
            z = config["z"]
            return float((x - 2) ** 2 + (y - 6) ** 2 + (z - 10) ** 2)

        # Validity check: say we require x + y + z <= 19
        # The global optimum x=2, y=6, z=10 sums to 18 (valid).
        # Any config summing to >19 is invalid.
        def mock_is_valid(config):
            return config["x"] + config["y"] + config["z"] <= 19

        param_space = {
            "x": [0, 1, 2, 3],
            "y": [4, 5, 6, 7],
            "z": [8, 9, 10, 11]
        }

        # Start from a valid suboptimal point: x=0, y=4, z=8
        initial_config = {"x": 0, "y": 4, "z": 8}
        
        autotuner = CoordinateDescentAutotuner(
            param_space=param_space,
            objective_fn=mock_objective,
            is_valid_fn=mock_is_valid,
            maximize=False,
            max_iterations=3,
            verbose=True
        )

        best_config, best_score = autotuner.tune(initial_config)

        # Assert we reached the global minimum
        self.assertEqual(best_config["x"], 2)
        self.assertEqual(best_config["y"], 6)
        self.assertEqual(best_config["z"], 10)
        self.assertEqual(best_score, 0.0)

        # Verify that caching is working (we shouldn't evaluate the same config multiple times)
        unique_cache_keys = len(autotuner.eval_cache)
        self.assertEqual(evals_count, unique_cache_keys)
        
        print("✓ CoordinateDescentAutotuner successfully converged to global optimum!")
        print(f"✓ Total evaluations: {evals_count} (out of {4 * 4 * 4} total grid space)")

    def test_dynamic_tuning(self):
        # We want to minimize a dynamic loss function:
        # loss = (BLOCK_M - 128)^2 + (maxrregcount - 127)^2 + (ab_stage - 5)^2
        # Target: BLOCK_M=128, maxrregcount=127, ab_stage=5
        
        def dynamic_objective(config):
            bm = config["BLOCK_M"]
            reg = config["maxrregcount"]
            stage = config["ab_stage"]
            return float((bm - 128) ** 2 + (reg - 127) ** 2 + (stage - 5) ** 2)

        def dynamic_is_valid(config):
            # Check custom constraints
            bm = config["BLOCK_M"]
            reg = config["maxrregcount"]
            stage = config["ab_stage"]
            # Just some mock checks to ensure constraints are respected
            return bm + stage <= 200 and reg in (31, 63, 127, 255)

        param_space = {
            "BLOCK_M": {"type": "both", "min": 16, "max": 256},
            "maxrregcount": {"type": "maxrregcount", "min": 31, "max": 255},
            "ab_stage": {"type": "step_1", "min": 1, "max": 10}
        }

        # Start value: suboptimal but valid
        initial_config = {
            "BLOCK_M": 64,
            "maxrregcount": 255,
            "ab_stage": 2
        }

        autotuner = CoordinateDescentAutotuner(
            param_space=param_space,
            objective_fn=dynamic_objective,
            is_valid_fn=dynamic_is_valid,
            maximize=False,
            max_iterations=5,
            verbose=True
        )

        best_config, best_score = autotuner.tune(initial_config)

        # Assert convergence to the designated optimum
        self.assertEqual(best_config["BLOCK_M"], 128)
        self.assertEqual(best_config["maxrregcount"], 127)
        self.assertEqual(best_config["ab_stage"], 5)
        self.assertEqual(best_score, 0.0)

        # Verify all evaluated configurations satisfy the specific requirements
        for key in autotuner.eval_cache.keys():
            config = dict(key)
            bm = config.get("BLOCK_M")
            reg = config.get("maxrregcount")
            stage = config.get("ab_stage")

            if bm is not None:
                self.assertTrue(16 <= bm <= 256, f"BLOCK_M {bm} out of bounds")
            if reg is not None:
                self.assertTrue(31 <= reg <= 255, f"maxrregcount {reg} out of bounds")
                # Assert reg is 2^p - 1
                self.assertEqual((reg + 1) & reg, 0, f"maxrregcount {reg} is not of the form 2^p - 1")
            if stage is not None:
                self.assertTrue(1 <= stage <= 10, f"ab_stage {stage} out of bounds")

        print("✓ CoordinateDescentAutotuner dynamic tuning tests passed successfully!")

    def test_unhashable_parameters(self):
        # Test that using lists/dicts inside parameters does not cause TypeError in _hash_config
        def mock_objective(config):
            # Objective: minimize list size and sum of dict values
            l_val = config["list_param"]
            d_val = config["dict_param"]
            return float(len(l_val) + sum(d_val.values()))

        param_space = {
            "list_param": [[1, 2], [1, 2, 3]],
            "dict_param": [{"a": 1}, {"a": 2}]
        }

        initial_config = {
            "list_param": [1, 2, 3],
            "dict_param": {"a": 2}
        }

        autotuner = CoordinateDescentAutotuner(
            param_space=param_space,
            objective_fn=mock_objective,
            maximize=False,
            verbose=False
        )

        best_config, best_score = autotuner.tune(initial_config)
        self.assertEqual(best_config["list_param"], [1, 2])
        self.assertEqual(best_config["dict_param"], {"a": 1})
        self.assertEqual(best_score, 3.0)

if __name__ == "__main__":
    unittest.main()
