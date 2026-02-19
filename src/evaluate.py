"""Evaluation script for comparing multiple runs."""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def bootstrap_ci(values: List[float], n_bootstrap: int = 10000, ci: float = 0.95) -> tuple:
    """Compute bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    
    values = np.array(values)
    n = len(values)
    
    # Bootstrap resampling
    bootstrap_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Compute percentiles
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    mean = np.mean(values)
    
    return mean, lower, upper


def load_results_from_file(results_dir: Path, run_id: str) -> tuple[Dict, List[Dict]]:
    """Load results from JSON file."""
    run_dir = results_dir / run_id
    
    # Load metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    # Load detailed results
    results_file = run_dir / "results.json"
    with open(results_file, "r") as f:
        results = json.load(f)
    
    return metrics, results


def load_results_from_wandb(entity: str, project: str, run_id: str) -> tuple[Dict, List]:
    """Load results from WandB API."""
    api = wandb.Api()
    
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get summary metrics
        metrics = dict(run.summary)
        
        # Get run config
        config = dict(run.config)
        
        # Note: detailed per-sample results not available via WandB API
        # Return empty list
        results = []
        
        return metrics, results
    except Exception as e:
        print(f"Warning: Could not load run {run_id} from WandB: {e}")
        return {}, []


def create_per_run_plots(results_dir: Path, run_id: str, results: List[Dict]):
    """Create per-run visualization."""
    if not results:
        return
    
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    indices = [r["index"] for r in results]
    group_correct = [int(r["group_correct"]) for r in results]
    tokens = [r["output_tokens"] for r in results]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Correctness over samples
    axes[0].plot(indices, group_correct, marker='o', linestyle='-', alpha=0.6)
    axes[0].set_xlabel("Sample Index")
    axes[0].set_ylabel("Group Correct (0/1)")
    axes[0].set_title(f"Correctness: {run_id}")
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Token distribution
    axes[1].hist(tokens, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel("Output Tokens")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Token Distribution: {run_id}")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = run_dir / "per_run_plots.pdf"
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Created {plot_file}")


def create_comparison_plots(results_dir: Path, all_results: Dict[str, List[Dict]], all_metrics: Dict[str, Dict]):
    """Create comparison plots across runs."""
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Bar chart of accuracy with CI
    metric_names = ["accuracy", "unperturbed_accuracy", "format_adherence_rate", "mismatch_rate"]
    
    for metric_name in metric_names:
        if not any(metric_name in m for m in all_metrics.values()):
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        run_ids = []
        means = []
        cis_lower = []
        cis_upper = []
        
        for run_id, results in all_results.items():
            if not results:
                # Fall back to summary metric if no per-sample data
                if run_id in all_metrics and metric_name in all_metrics[run_id]:
                    run_ids.append(run_id)
                    means.append(all_metrics[run_id][metric_name])
                    cis_lower.append(all_metrics[run_id][metric_name])
                    cis_upper.append(all_metrics[run_id][metric_name])
                continue
            
            # Compute bootstrap CI from per-sample data
            if metric_name == "accuracy":
                values = [int(r["group_correct"]) for r in results]
            elif metric_name == "unperturbed_accuracy":
                values = [int(r["correct_original"]) for r in results]
            elif metric_name == "format_adherence_rate":
                values = [int(r["format_adherent"]) for r in results]
            elif metric_name == "mismatch_rate":
                values = [int(not r["consistent"]) for r in results]
            else:
                continue
            
            mean, lower, upper = bootstrap_ci(values)
            run_ids.append(run_id)
            means.append(mean)
            cis_lower.append(mean - lower)
            cis_upper.append(upper - mean)
        
        if not run_ids:
            continue
        
        x_pos = np.arange(len(run_ids))
        ax.bar(x_pos, means, yerr=[cis_lower, cis_upper], capsize=5, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"Comparison: {metric_name}")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = comparison_dir / f"comparison_{metric_name}.pdf"
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Created {plot_file}")
    
    # Plot 2: Token usage comparison
    if all(all_metrics[rid].get("mean_output_tokens") for rid in all_metrics):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        run_ids = list(all_metrics.keys())
        tokens = [all_metrics[rid]["mean_output_tokens"] for rid in run_ids]
        
        x_pos = np.arange(len(run_ids))
        ax.bar(x_pos, tokens, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(run_ids, rotation=45, ha='right')
        ax.set_ylabel("Mean Output Tokens")
        ax.set_title("Token Usage Comparison")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plot_file = comparison_dir / "comparison_tokens.pdf"
        plt.savefig(plot_file)
        plt.close()
        
        print(f"Created {plot_file}")


def compute_aggregated_metrics(all_metrics: Dict[str, Dict]) -> Dict:
    """Compute aggregated comparison metrics."""
    # Identify proposed vs baseline
    proposed_runs = [rid for rid in all_metrics if "proposed" in rid]
    baseline_runs = [rid for rid in all_metrics if "comparative" in rid]
    
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": all_metrics,
    }
    
    if proposed_runs and baseline_runs:
        # Get best of each
        best_proposed_acc = max(all_metrics[rid].get("accuracy", 0) for rid in proposed_runs)
        best_baseline_acc = max(all_metrics[rid].get("accuracy", 0) for rid in baseline_runs)
        
        aggregated["best_proposed"] = best_proposed_acc
        aggregated["best_baseline"] = best_baseline_acc
        aggregated["gap"] = best_proposed_acc - best_baseline_acc
    
    return aggregated


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate experiment runs")
    parser.add_argument("--results_dir", type=str, default=".research/results",
                        help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True,
                        help="JSON string list of run IDs to compare")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="WandB entity (optional)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="WandB project (optional)")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    run_ids = json.loads(args.run_ids)
    
    print(f"Evaluating runs: {run_ids}")
    print(f"Results directory: {results_dir}")
    
    # Load results for all runs
    all_metrics = {}
    all_results = {}
    
    for run_id in run_ids:
        print(f"\nLoading {run_id}...")
        
        # Try to load from file first
        try:
            metrics, results = load_results_from_file(results_dir, run_id)
            print(f"  Loaded from file: {len(results)} samples")
        except FileNotFoundError:
            # Fall back to WandB if available
            if args.wandb_entity and args.wandb_project:
                print(f"  File not found, trying WandB...")
                metrics, results = load_results_from_wandb(
                    args.wandb_entity, args.wandb_project, run_id
                )
            else:
                print(f"  WARNING: Could not load results for {run_id}")
                continue
        
        all_metrics[run_id] = metrics
        all_results[run_id] = results
        
        # Export per-run metrics
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_file = run_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved metrics to {metrics_file}")
        
        # Create per-run plots
        if results:
            create_per_run_plots(results_dir, run_id, results)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(results_dir, all_results, all_metrics)
    
    # Compute and save aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated = compute_aggregated_metrics(all_metrics)
    
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    agg_file = comparison_dir / "aggregated_metrics.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    print(f"Saved aggregated metrics to {agg_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for run_id, metrics in all_metrics.items():
        print(f"\n{run_id}:")
        for key in ["accuracy", "unperturbed_accuracy", "format_adherence_rate", "mean_output_tokens"]:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
    
    if "gap" in aggregated:
        print(f"\nProposed vs Baseline Gap: {aggregated['gap']:.4f}")
    
    print("\nAll files saved to:", results_dir)


if __name__ == "__main__":
    main()
