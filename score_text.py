import torch
import transformers
import numpy as np
from tqdm import tqdm
import argparse
import os
import tempfile

def load_models(base_model_name="gpt2-medium", mask_filling_model_name="t5-large", cache_dir=None, device="cuda"):
    """Load the base model and mask filling model."""
    if cache_dir is None:
        # Use a temporary directory if no cache directory is specified
        cache_dir = os.path.join(tempfile.gettempdir(), "detectgpt_cache")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f'Loading BASE model {base_model_name}...')
    try:
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            cache_dir=cache_dir,
            local_files_only=False  # Force download if needed
        )
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_name, 
            cache_dir=cache_dir,
            local_files_only=False
        )
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
    except Exception as e:
        print(f"Error loading base model: {e}")
        print("Trying to download with trust_remote_code=True...")
        base_model = transformers.AutoModelForCausalLM.from_pretrained(
            base_model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        base_tokenizer = transformers.AutoTokenizer.from_pretrained(
            base_model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        base_tokenizer.pad_token_id = base_tokenizer.eos_token_id

    print(f'Loading mask filling model {mask_filling_model_name}...')
    try:
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            mask_filling_model_name, 
            cache_dir=cache_dir,
            local_files_only=False
        )
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            mask_filling_model_name, 
            cache_dir=cache_dir,
            local_files_only=False
        )
    except Exception as e:
        print(f"Error loading mask model: {e}")
        print("Trying to download with trust_remote_code=True...")
        mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            mask_filling_model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        mask_tokenizer = transformers.AutoTokenizer.from_pretrained(
            mask_filling_model_name, 
            cache_dir=cache_dir,
            trust_remote_code=True
        )

    # Move models to device
    base_model.to(device)
    mask_model.to(device)

    return base_model, base_tokenizer, mask_model, mask_tokenizer

def get_ll(text, base_model, base_tokenizer, device="cuda"):
    """Get the log likelihood of a text under the base model."""
    with torch.no_grad():
        tokenized = base_tokenizer(text, return_tensors="pt").to(device)
        labels = tokenized.input_ids
        return -base_model(**tokenized, labels=labels).loss.item()

def perturb_text(text, span_length=2, pct=0.3, mask_model=None, mask_tokenizer=None, device="cuda"):
    """Create perturbed versions of the text."""
    tokens = text.split(' ')
    n_spans = int(pct * len(tokens) / (span_length + 2))
    
    perturbed_texts = []
    for _ in range(10):  # Create 10 perturbed versions
        current_tokens = tokens.copy()
        for _ in range(n_spans):
            start = np.random.randint(0, len(current_tokens) - span_length)
            current_tokens[start:start + span_length] = [f'<extra_id_{i}>' for i in range(span_length)]
        
        # Replace masks with T5 predictions
        masked_text = ' '.join(current_tokens)
        inputs = mask_tokenizer(masked_text, return_tensors="pt", padding=True).to(device)
        outputs = mask_model.generate(**inputs, max_length=150, do_sample=True, top_p=0.96)
        perturbed_text = mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
        perturbed_texts.append(perturbed_text)
    
    return perturbed_texts

def score_text(text, base_model, base_tokenizer, mask_model, mask_tokenizer, device="cuda"):
    """Score a text using DetectGPT's perturbation method."""
    # Get original text likelihood
    original_ll = get_ll(text, base_model, base_tokenizer, device)
    
    # Get perturbed texts and their likelihoods
    perturbed_texts = perturb_text(text, mask_model=mask_model, mask_tokenizer=mask_tokenizer, device=device)
    perturbed_lls = [get_ll(t, base_model, base_tokenizer, device) for t in perturbed_texts]
    
    # Calculate z-score
    mean_perturbed_ll = np.mean(perturbed_lls)
    std_perturbed_ll = np.std(perturbed_lls)
    z_score = (original_ll - mean_perturbed_ll) / std_perturbed_ll
    
    # Convert z-score to probability (higher z-score = higher probability of being machine-generated)
    probability = 1 / (1 + np.exp(-z_score))
    
    return probability

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Text to analyze')
    parser.add_argument('--base_model', type=str, default="gpt2-medium", help='Base model to use')
    parser.add_argument('--mask_model', type=str, default="t5-large", help='Mask filling model to use')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run on')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for models')
    args = parser.parse_args()

    try:
        # Load models
        base_model, base_tokenizer, mask_model, mask_tokenizer = load_models(
            args.base_model, args.mask_model, cache_dir=args.cache_dir, device=args.device
        )
        
        # Score the text
        probability = score_text(
            args.text, base_model, base_tokenizer, mask_model, mask_tokenizer, args.device
        )
        
        print(f"\nDetectGPT Score: {probability:.4f}")
        print(f"Interpretation: {probability:.1%} probability of being machine-generated")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure you have enough disk space")
        print("2. Check your internet connection")
        print("3. Try running with --cache_dir pointing to a different directory")
        print("4. If using CUDA, make sure you have enough GPU memory")

if __name__ == "__main__":
    main() 