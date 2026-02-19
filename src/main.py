"""Main entry point for experiments."""

import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> int:
    """
    Main orchestrator for a single run.
    
    This function:
    1. Loads configuration using Hydra
    2. Applies mode-specific overrides
    3. Invokes the appropriate task (inference for this experiment)
    4. Returns exit code
    """
    print("=" * 60)
    print(f"Run ID: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 60)
    
    # Print config (useful for debugging)
    if cfg.get("verbose", False):
        print("\nConfiguration:")
        print(OmegaConf.to_yaml(cfg))
    
    # This experiment is inference-only (no training)
    # Task type determined by experimental design: prompt tuning / inference
    task_type = "inference"
    
    print(f"\nTask type: {task_type}")
    
    # Run inference
    try:
        exit_code = run_inference(cfg)
        return exit_code
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
