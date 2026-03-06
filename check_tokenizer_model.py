#!/usr/bin/env python3
"""
Check tokenizer vs model vocab/embeddings and special tokens for Qwen2.5-VL.

Usage:
  python check_tokenizer_model.py --model Qwen/Qwen2.5-VL-7B-Instruct --cache_dir /path/to/hf_cache
  # To also load the model (heavy), add --load_model
"""

import argparse
import sys
from pprint import pprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="model id/path")
    parser.add_argument("--cache_dir", type=str, default=None, help="HF cache dir (optional)")
    parser.add_argument("--load_model", action="store_true", help="If set, load the model (heavy).")
    args = parser.parse_args()

    try:
        from transformers import AutoProcessor, AutoConfig
    except Exception as e:
        print("ERROR: transformers not importable:", e, file=sys.stderr)
        sys.exit(2)

    model_id = args.model
    cache = {"cache_dir": args.cache_dir} if args.cache_dir else {}

    print("Loading processor (AutoProcessor.from_pretrained)...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **cache)
    tok = processor.tokenizer

    print("\nTokenizer overview:")
    try:
        vocab_size_prop = getattr(tok, "vocab_size", None)
        print(" tokenizer.vocab_size attribute:", vocab_size_prop)
    except Exception:
        print(" tokenizer.vocab_size not available")

    try:
        vocab_len = len(tok)
        print(" len(tokenizer):", vocab_len)
    except Exception:
        try:
            print(" len(tokenizer.get_vocab()):", len(tok.get_vocab()))
        except Exception as e:
            print(" cannot determine tokenizer length:", e)

    print("\nSpecial tokens / additional special tokens:")
    try:
        special = {
            "bos_token": getattr(tok, "bos_token", None),
            "eos_token": getattr(tok, "eos_token", None),
            "pad_token": getattr(tok, "pad_token", None),
            "unk_token": getattr(tok, "unk_token", None),
            "sep_token": getattr(tok, "sep_token", None),
            "cls_token": getattr(tok, "cls_token", None),
            "additional_special_tokens": getattr(tok, "additional_special_tokens", None),
        }
        pprint(special)
    except Exception as e:
        print("Could not read special tokens:", e)

    latent_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    other_test_tokens = ["<|im_start|>assistant", "<|endoftext|>", "<|vision_start|>", "<|vision_end|>"]

    print("\nChecking latent and other special tokens (convert to ids):")
    def show_token_info(s):
        try:
            out = tok(s, return_tensors="pt")
            ids = out["input_ids"][0].tolist()
            print(f"  {s!r} -> token ids: {ids}  (len {len(ids)})")
        except Exception as e:
            try:
                id_ = tok.convert_tokens_to_ids(s)
                print(f"  {s!r} -> convert_tokens_to_ids: {id_}")
            except Exception:
                print(f"  {s!r} -> error: {e}")

    for t in latent_tokens + other_test_tokens:
        show_token_info(t)

    print("\nLoading model config (AutoConfig.from_pretrained)...")
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True, **cache)
    print(" model.config.vocab_size:", getattr(cfg, "vocab_size", None))
    latent_fields = {k: getattr(cfg, k) for k in ["latent_size", "latent_token_id", "latent_start_id", "latent_end_id"] if hasattr(cfg, k)}
    if latent_fields:
        print(" latent-related config fields:")
        pprint(latent_fields)
    else:
        print(" no known latent-related fields in config")

    if args.load_model:
        print("\nLoading model weights (this may be large; be prepared)...")
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                trust_remote_code=True,
                **({} if args.cache_dir is None else {"cache_dir": args.cache_dir}),
            )
        except Exception as e:
            print("Failed to load model:", e, file=sys.stderr)
            sys.exit(3)

        print("Model loaded. Inspecting embedding/output shapes...")

        emb = model.get_input_embeddings()
        try:
            emb_shape = tuple(emb.weight.shape)
            print(" input_embeddings.weight.shape:", emb_shape)
        except Exception as e:
            print(" Could not read input embedding weight:", e)

        out_emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if out_emb is not None:
            try:
                out_shape = tuple(out_emb.weight.shape)
                print(" output_embeddings.weight.shape:", out_shape)
            except Exception as e:
                print(" Could not read output embedding weight:", e)
        else:
            if hasattr(model, "lm_head"):
                try:
                    print(" model.lm_head.weight.shape:", tuple(model.lm_head.weight.shape))
                except Exception:
                    pass

        try:
            tok_len = len(tok)
            emb_num = emb.num_embeddings if hasattr(emb, "num_embeddings") else emb.weight.shape[0]
            print(f"\nlen(tokenizer) == {tok_len}; input_embeddings.num_embeddings == {emb_num}")
            if tok_len != emb_num:
                print("Mismatch: tokenizer length != model embedding size")
                print("You may need to call model.resize_token_embeddings(len(tokenizer)) in training code.")
            else:
                print("Tokenizer length matches embedding size.")
        except Exception as e:
            print("Could not compare tokenizer vs embedding size:", e)

    print("\nDone.")


if __name__ == "__main__":
    main()
