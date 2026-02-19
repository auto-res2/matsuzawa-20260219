"""Dataset loading and perturbation generation for GSM8K experiments."""

import re
import random
from typing import Dict, List, Tuple
from datasets import load_dataset


def load_gsm8k(split: str = "test", n_samples: int = 200, cache_dir: str = ".cache") -> List[Dict]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split (train/test)
        n_samples: Number of samples to load
        cache_dir: Cache directory for datasets
        
    Returns:
        List of examples with 'question', 'answer', 'gold_numeric' fields
    """
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    
    # Select first n_samples
    if n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))
    
    # Parse gold answers
    examples = []
    for ex in dataset:
        question = ex["question"]
        answer_text = ex["answer"]
        
        # Extract numeric answer after ####
        gold_numeric = answer_text.strip().split("####")[-1].strip()
        
        examples.append({
            "question": question,
            "answer": answer_text,
            "gold_numeric": gold_numeric,
        })
    
    return examples


def sentence_shuffle(text: str, seed: int = 0) -> str:
    """
    Shuffle sentences while keeping the last question sentence at the end.
    
    Args:
        text: Input text
        seed: Random seed for reproducibility
        
    Returns:
        Shuffled text
    """
    # Split on sentence boundaries
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    
    if len(parts) <= 1:
        return text
    
    # Keep last sentence (usually the question) at the end
    last = parts[-1]
    body = parts[:-1]
    
    # Shuffle body
    rnd = random.Random(seed)
    rnd.shuffle(body)
    
    return " ".join(body + [last])


def add_irrelevant(text: str, irrelevant_text: str = " Note: the sky is blue.") -> str:
    """
    Append an irrelevant sentence to the text.
    
    Args:
        text: Input text
        irrelevant_text: Irrelevant text to append
        
    Returns:
        Text with irrelevant detail added
    """
    return text + irrelevant_text


def generate_perturbations(question: str, index: int, perturbation_configs: List[Dict]) -> Dict[str, str]:
    """
    Generate perturbations for a question.
    
    Args:
        question: Original question
        index: Question index (used as seed)
        perturbation_configs: List of perturbation config dicts
        
    Returns:
        Dict with 'original', 'v1', 'v2' keys
    """
    variants = {"original": question}
    
    # Generate variants based on config
    for i, config in enumerate(perturbation_configs):
        variant_key = f"v{i+1}"
        
        if config["type"] == "sentence_shuffle":
            seed = index + config.get("seed_offset", 0)
            variants[variant_key] = sentence_shuffle(question, seed=seed)
        elif config["type"] == "add_irrelevant":
            irrelevant_text = config.get("text", " Note: the sky is blue.")
            variants[variant_key] = add_irrelevant(question, irrelevant_text)
        else:
            raise ValueError(f"Unknown perturbation type: {config['type']}")
    
    return variants


def extract_final_number(text: str) -> str:
    """
    Extract final numeric answer from model output.
    
    Looks for "FINAL: <number>" pattern first, falls back to last number.
    
    Args:
        text: Model output text
        
    Returns:
        Extracted number as string, or None if not found
    """
    # Try to find FINAL: <number> pattern
    final_match = re.search(r"FINAL:\s*([^\n]+)", text, re.IGNORECASE)
    if final_match:
        # Extract number from the FINAL line
        num_match = re.search(r"(-?\d+\.?\d*)", final_match.group(1))
        if num_match:
            return num_match.group(1)
    
    # Fall back to last number in text
    all_numbers = re.findall(r"(-?\d+\.?\d*)", text)
    if all_numbers:
        return all_numbers[-1]
    
    return None


def normalize_number(num_str: str) -> str:
    """
    Normalize numeric string for comparison.
    
    Converts to float, rounds if close to integer, returns canonical string.
    
    Args:
        num_str: Number as string
        
    Returns:
        Normalized number string, or None if invalid
    """
    if num_str is None:
        return None
    
    try:
        x = float(num_str)
        
        # If close to integer, return as integer string
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        
        # Otherwise return canonical float string
        return str(x)
    except (ValueError, TypeError):
        return None


def parse_igv_cot_output(text: str) -> Dict[str, str]:
    """
    Parse IGV-CoT output to extract answers for ORIGINAL, V1, V2.
    
    Args:
        text: Full model output
        
    Returns:
        Dict with 'original', 'v1', 'v2' keys containing extracted numbers
    """
    results = {}
    
    # The final FINAL: line is the adjudicated answer for ORIGINAL
    results["original"] = extract_final_number(text)
    
    # Try to extract V1 and V2 answers from metamorphic checks section
    # Look for patterns like "V1: <number>" or "Answer for V1: <number>"
    v1_match = re.search(r"V1[:\s]+.*?(-?\d+\.?\d*)", text, re.IGNORECASE)
    if v1_match:
        results["v1"] = v1_match.group(1)
    else:
        # Fall back: assume same as original if not explicitly different
        results["v1"] = results["original"]
    
    v2_match = re.search(r"V2[:\s]+.*?(-?\d+\.?\d*)", text, re.IGNORECASE)
    if v2_match:
        results["v2"] = v2_match.group(1)
    else:
        # Fall back: assume same as original if not explicitly different
        results["v2"] = results["original"]
    
    return results
