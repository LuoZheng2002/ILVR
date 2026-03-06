#!/usr/bin/env python3
"""
Check whether `model.resize_token_embeddings(...)` updates model.config.vocab_size and embedding/output shapes.

Usage:
  python check_resize_behavior.py --model Qwen/Qwen2.5-VL-7B-Instruct --cache_dir /path/to/cache

This script will:
 - load processor/tokenizer
 - add special tokens if missing
 - load model
 - report config.vocab_size, input/output embedding shapes
 - call model.resize_token_embeddings(len(tokenizer)) and report shapes again
"""

import argparse
import sys
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cache_dir", type=str, default=None)
    args = parser.parse_args()

    try:
        from transformers import AutoProcessor
    except Exception as e:
        print("ERROR: transformers import failed:", e, file=sys.stderr)
        sys.exit(2)

    cache = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    print("Loading processor/tokenizer...")
    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, **cache)
    tok = proc.tokenizer

    new_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    # ensure they are present as additional special tokens
    try:
        existing = getattr(tok, "additional_special_tokens", []) or []
        missing = [t for t in new_tokens if t not in existing]
        if missing:
            print("Adding missing special tokens:", missing)
            tok.add_special_tokens({"additional_special_tokens": missing})
        else:
            print("All latent tokens already present in additional_special_tokens")
    except Exception as e:
        print("Could not add special tokens:", e)

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
    except Exception as e:
        print("ERROR: model class import failed:", e, file=sys.stderr)
        sys.exit(3)

    print("\nTokenizer:")
    try:
        print(" len(tokenizer):", len(tok))
    except Exception:
        print(" tokenizer.vocab_size:", getattr(tok, "vocab_size", None))

    print("\nLoading model (may be large)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, trust_remote_code=True, **({} if args.cache_dir is None else {"cache_dir": args.cache_dir}))

    def emb_info(m):
        info = {}
        try:
            ie = m.get_input_embeddings()
            info['input_embeddings_shape'] = tuple(ie.weight.shape)
            info['input_num_embeddings'] = ie.num_embeddings if hasattr(ie, 'num_embeddings') else ie.weight.shape[0]
        except Exception as e:
            info['input_embeddings_shape'] = f"error: {e}"
        try:
            oe = m.get_output_embeddings()
            if oe is not None:
                info['output_embeddings_shape'] = tuple(oe.weight.shape)
        except Exception as e:
            info['output_embeddings_shape'] = f"error: {e}"
        return info

    print("\nBefore resize:")
    print(" model.config.vocab_size:", getattr(model.config, 'vocab_size', None))
    pprint(emb_info(model))

    new_size = len(tok)
    print(f"\nCalling model.resize_token_embeddings({new_size})...")
    try:
        model.resize_token_embeddings(new_size)
    except Exception as e:
        print(" resize_token_embeddings raised:", e)

    print("\nAfter resize:")
    print(" model.config.vocab_size:", getattr(model.config, 'vocab_size', None))
    pprint(emb_info(model))

    print("\nSanity checks:")
    try:
        tok_len = len(tok)
        ie = model.get_input_embeddings()
        emb_num = ie.num_embeddings if hasattr(ie, 'num_embeddings') else ie.weight.shape[0]
        print(f" len(tokenizer)={tok_len}; input_embeddings.num_embeddings={emb_num}")
    except Exception as e:
        print("Could not compare tokenizer vs embedding:", e)

    print('\nDone')


if __name__ == '__main__':
    main()
