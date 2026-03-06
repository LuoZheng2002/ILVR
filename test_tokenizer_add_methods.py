#!/usr/bin/env python3
"""
Compare effects of two tokenizer update methods:

- Method A (simulates current repo): call `tokenizer.add_tokens(new_tokens, special_tokens=True)`
  (falls back to `add_tokens(new_tokens)` if that signature isn't supported)
- Method B (recommended): call `tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})`

For each method this script:
 - loads a fresh processor/tokenizer from the specified model
 - applies the method to add latent tokens
 - reports tokenizer sizes and how the latent tokens tokenize
 - optionally loads the model, calls `model.resize_token_embeddings(len(tokenizer))`,
   and reports embedding/output shapes to see whether sizes align

Usage:
  python test_tokenizer_add_methods.py --model Qwen/Qwen2.5-VL-7B-Instruct --cache_dir /path/to/cache --load_model
"""

import argparse
import sys
from pprint import pprint


def report_token_info(tok, tokens):
    print(" tokenizer.vocab_size attr:", getattr(tok, "vocab_size", None))
    try:
        print(" len(tokenizer):", len(tok))
    except Exception:
        try:
            print(" len(tokenizer.get_vocab()):", len(tok.get_vocab()))
        except Exception as e:
            print(" could not get tokenizer length:", e)

    for t in tokens:
        try:
            out = tok(t, return_tensors="pt")
            ids = out["input_ids"][0].tolist()
            print(f"  {t!r} -> token ids: {ids}  (len {len(ids)})")
        except Exception as e:
            try:
                id_ = tok.convert_tokens_to_ids(t)
                print(f"  {t!r} -> convert_tokens_to_ids: {id_}")
            except Exception:
                print(f"  {t!r} -> error: {e}")


def try_add_tokens_variant(tok, new_tokens):
    """Try to mimic existing repo call: tokenizer.add_tokens(new_tokens, special_tokens=True)
    Fall back to tokenizer.add_tokens(new_tokens) if signature doesn't accept special_tokens.
    Return True if add_tokens changed vocab, False otherwise."""
    before_len = None
    try:
        before_len = len(tok)
    except Exception:
        try:
            before_len = getattr(tok, "vocab_size", None)
        except Exception:
            before_len = None

    changed = False
    try:
        # Some HF tokenizer implementations accept a special_tokens kwarg (older/newer variants).
        tok.add_tokens(new_tokens, special_tokens=True)
        changed = True
    except TypeError:
        # Common fallback
        try:
            tok.add_tokens(new_tokens)
            changed = True
        except Exception:
            changed = False
    except Exception:
        changed = False

    try:
        after_len = len(tok)
    except Exception:
        after_len = getattr(tok, "vocab_size", None)

    return changed, before_len, after_len


def try_add_special_tokens_variant(tok, new_tokens):
    before_len = None
    try:
        before_len = len(tok)
    except Exception:
        before_len = getattr(tok, "vocab_size", None)

    try:
        tok.add_special_tokens({"additional_special_tokens": new_tokens})
        changed = True
    except Exception:
        changed = False

    try:
        after_len = len(tok)
    except Exception:
        after_len = getattr(tok, "vocab_size", None)

    return changed, before_len, after_len


def run_test(model_id, cache_dir=None, load_model=False):
    try:
        from transformers import AutoProcessor, AutoConfig
    except Exception as e:
        print("ERROR: transformers not importable:", e, file=sys.stderr)
        sys.exit(2)

    cache = {"cache_dir": cache_dir} if cache_dir else {}
    new_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]

    print("=== Method A: add_tokens (simulate repo) ===")
    procA = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **cache)
    tokA = procA.tokenizer
    changedA, beforeA, afterA = try_add_tokens_variant(tokA, new_tokens)
    print(f" add_tokens changed: {changedA}; before_len={beforeA}; after_len={afterA}")
    report_token_info(tokA, new_tokens + ["<|im_start|>assistant", "<|endoftext|>", "<|vision_start|>"])

    if load_model:
        from transformers import Qwen2_5_VLForConditionalGeneration
        from transformers import AutoConfig as _AutoConfig
        cfg = _AutoConfig.from_pretrained(model_id, trust_remote_code=True, **cache)
        print(" model.config.vocab_size:", getattr(cfg, "vocab_size", None))
        print(" Loading model (may be large)...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **({} if cache_dir is None else {"cache_dir": cache_dir}))
        print(" Before resize: input_embeddings.num_embeddings:", model.get_input_embeddings().weight.shape[0])
        try:
            model.resize_token_embeddings(len(tokA))
            print(" After resize: input_embeddings.num_embeddings:", model.get_input_embeddings().weight.shape[0])
        except Exception as e:
            print(" resize_token_embeddings failed:", e)

    print("\n=== Method B: add_special_tokens (recommended) ===")
    procB = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **cache)
    tokB = procB.tokenizer
    changedB, beforeB, afterB = try_add_special_tokens_variant(tokB, new_tokens)
    print(f" add_special_tokens changed: {changedB}; before_len={beforeB}; after_len={afterB}")
    report_token_info(tokB, new_tokens + ["<|im_start|>assistant", "<|endoftext|>", "<|vision_start|>"])

    if load_model:
        # Re-load model (fresh) to avoid mixing states
        from transformers import Qwen2_5_VLForConditionalGeneration
        model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, trust_remote_code=True, **({} if cache_dir is None else {"cache_dir": cache_dir}))
        print(" Before resize (method B): input_embeddings.num_embeddings:", model2.get_input_embeddings().weight.shape[0])
        try:
            model2.resize_token_embeddings(len(tokB))
            print(" After resize (method B): input_embeddings.num_embeddings:", model2.get_input_embeddings().weight.shape[0])
        except Exception as e:
            print(" resize_token_embeddings failed (method B):", e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--load_model", action="store_true")
    args = parser.parse_args()
    run_test(args.model, cache_dir=args.cache_dir, load_model=args.load_model)


if __name__ == "__main__":
    main()
