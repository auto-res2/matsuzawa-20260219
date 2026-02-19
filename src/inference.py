"""Inference script for prompt-based LLM experiments."""

import os
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
import wandb

from src.preprocess import (
    load_gsm8k,
    generate_perturbations,
    extract_final_number,
    normalize_number,
    parse_igv_cot_output,
)


def get_openai_client(cfg: DictConfig):
    """Initialize OpenAI client with config settings."""
    from openai import OpenAI
    
    # Get API key from environment
    api_key = os.getenv(cfg.model.api_key_env, "EMPTY")
    
    # Get base URL from config or environment
    base_url = cfg.model.base_url
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    return OpenAI(api_key=api_key, base_url=base_url)


def call_model(
    client,
    prompt: str,
    model_name: str,
    temperature: float = 0.0,
    max_tokens: int = 800,
) -> tuple[str, int]:
    """
    Call LLM with prompt and return response.
    
    Returns:
        (response_text, output_tokens)
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    output_text = response.choices[0].message.content
    output_tokens = response.usage.completion_tokens
    
    return output_text, output_tokens


def run_inference_single_call(
    cfg: DictConfig,
    examples: List[Dict],
    client,
) -> List[Dict]:
    """
    Run inference with single-call method (IGV-CoT).
    
    Each example gets 1 call with all variants bundled.
    """
    results = []
    
    for i, example in enumerate(examples):
        question = example["question"]
        gold = normalize_number(example["gold_numeric"])
        
        # Generate perturbations
        variants = generate_perturbations(
            question, i, cfg.inference.perturbations
        )
        
        # Build prompt with all variants
        prompt = cfg.method.prompt_template.format(
            q=variants["original"],
            v1=variants["v1"],
            v2=variants["v2"],
        )
        
        # Call model once
        output, tokens = call_model(
            client,
            prompt,
            cfg.model.name,
            cfg.model.temperature,
            cfg.model.max_tokens,
        )
        
        # Parse output to extract answers for each variant
        parsed = parse_igv_cot_output(output)
        predictions = {
            k: normalize_number(v) for k, v in parsed.items()
        }
        
        # Check format adherence
        format_ok = "FINAL:" in output.upper()
        
        # Check consistency (all predictions match)
        pred_values = [predictions.get("original"), predictions.get("v1"), predictions.get("v2")]
        consistent = len(set(v for v in pred_values if v is not None)) <= 1
        
        # Compute correctness
        correct_original = predictions.get("original") == gold
        correct_v1 = predictions.get("v1") == gold
        correct_v2 = predictions.get("v2") == gold
        group_correct = correct_original and correct_v1 and correct_v2
        
        result = {
            "index": i,
            "question": question,
            "gold": gold,
            "variants": variants,
            "output": output,
            "predictions": predictions,
            "output_tokens": tokens,
            "format_adherent": format_ok,
            "consistent": consistent,
            "correct_original": correct_original,
            "correct_v1": correct_v1,
            "correct_v2": correct_v2,
            "group_correct": group_correct,
        }
        
        results.append(result)
        
        # Log progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples")
    
    return results


def run_inference_multi_call(
    cfg: DictConfig,
    examples: List[Dict],
    client,
) -> List[Dict]:
    """
    Run inference with multi-call method (Standard CoT).
    
    Each example gets 3 separate calls (original, v1, v2).
    """
    results = []
    
    for i, example in enumerate(examples):
        question = example["question"]
        gold = normalize_number(example["gold_numeric"])
        
        # Generate perturbations
        variants = generate_perturbations(
            question, i, cfg.inference.perturbations
        )
        
        # Call model 3 times (once per variant)
        predictions = {}
        outputs = {}
        total_tokens = 0
        
        for variant_name, variant_text in variants.items():
            prompt = cfg.method.prompt_template.format(q=variant_text)
            output, tokens = call_model(
                client,
                prompt,
                cfg.model.name,
                cfg.model.temperature,
                cfg.model.max_tokens,
            )
            
            outputs[variant_name] = output
            predictions[variant_name] = normalize_number(extract_final_number(output))
            total_tokens += tokens
        
        # Check format adherence (all 3 outputs)
        format_ok = all("FINAL:" in out.upper() for out in outputs.values())
        
        # Check consistency
        pred_values = [predictions.get("original"), predictions.get("v1"), predictions.get("v2")]
        consistent = len(set(v for v in pred_values if v is not None)) <= 1
        
        # Compute correctness
        correct_original = predictions.get("original") == gold
        correct_v1 = predictions.get("v1") == gold
        correct_v2 = predictions.get("v2") == gold
        group_correct = correct_original and correct_v1 and correct_v2
        
        result = {
            "index": i,
            "question": question,
            "gold": gold,
            "variants": variants,
            "outputs": outputs,
            "output": outputs.get("original", ""),  # Store original for compatibility
            "predictions": predictions,
            "output_tokens": total_tokens,
            "format_adherent": format_ok,
            "consistent": consistent,
            "correct_original": correct_original,
            "correct_v1": correct_v1,
            "correct_v2": correct_v2,
            "group_correct": group_correct,
        }
        
        results.append(result)
        
        # Log progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(examples)} examples (3 calls each)")
    
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """Compute aggregated metrics from results."""
    n = len(results)
    
    if n == 0:
        return {}
    
    metrics = {
        "accuracy": sum(r["group_correct"] for r in results) / n,
        "unperturbed_accuracy": sum(r["correct_original"] for r in results) / n,
        "format_adherence_rate": sum(r["format_adherent"] for r in results) / n,
        "mismatch_rate": sum(not r["consistent"] for r in results) / n,
        "mean_output_tokens": sum(r["output_tokens"] for r in results) / n,
        "total_samples": n,
    }
    
    return metrics


def run_inference(cfg: DictConfig):
    """Main inference function."""
    print(f"Running inference for: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print(f"Method: {cfg.run.method.name}")
    print(f"Model: {cfg.run.model.name}")
    
    # Apply mode overrides
    if cfg.mode == "sanity_check":
        # Reduce to 10 samples for sanity check
        n_samples = 10
        # Use separate wandb project namespace for sanity checks
        wandb_project = f"{cfg.wandb.project}-sanity"
    else:
        n_samples = cfg.run.dataset.n_samples
        wandb_project = cfg.wandb.project
    
    # Initialize WandB
    wandb_mode = cfg.wandb.mode
    if wandb_mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=wandb_project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )
        print(f"WandB URL: {wandb.run.get_url()}")
    
    # Load dataset
    print(f"Loading {n_samples} examples from {cfg.run.dataset.name}...")
    examples = load_gsm8k(
        split=cfg.run.dataset.split,
        n_samples=n_samples,
        cache_dir=cfg.run.dataset.cache_dir,
    )
    print(f"Loaded {len(examples)} examples")
    
    # Initialize model client
    client = get_openai_client(cfg.run)
    
    # Run inference
    if cfg.run.method.single_call:
        print("Running single-call inference (IGV-CoT)...")
        results = run_inference_single_call(cfg, examples, client)
    else:
        print("Running multi-call inference (Standard CoT)...")
        results = run_inference_multi_call(cfg, examples, client)
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    print("\n=== Metrics ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Log to WandB
    if wandb_mode != "disabled":
        wandb.log(metrics)
        for key, value in metrics.items():
            wandb.summary[key] = value
        wandb.finish()
    
    # Sanity validation for sanity_check mode
    if cfg.mode == "sanity_check":
        print("\n=== Sanity Validation ===")
        passed = True
        reason = ""
        
        # Check: at least 5 samples processed
        if len(results) < 5:
            passed = False
            reason = "insufficient_samples"
        
        # Check: all metrics are finite
        elif any(not (isinstance(v, (int, float)) and abs(v) != float('inf')) 
                 for v in metrics.values() if isinstance(v, (int, float))):
            passed = False
            reason = "non_finite_metrics"
        
        # Check: at least some predictions are valid
        elif metrics.get("format_adherence_rate", 0) == 0:
            passed = False
            reason = "no_valid_predictions"
        
        # Print summary
        summary = {
            "samples": len(results),
            "accuracy": metrics.get("accuracy", 0),
            "format_adherence": metrics.get("format_adherence_rate", 0),
            "mean_tokens": metrics.get("mean_output_tokens", 0),
        }
        print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
        
        if passed:
            print("SANITY_VALIDATION: PASS")
        else:
            print(f"SANITY_VALIDATION: FAIL reason={reason}")
            sys.exit(1)
    
    return 0
