#!/usr/bin/env python3
import argparse
import importlib

transformers = importlib.import_module("transformers")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check tokenizer/model vocab alignment.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model/tokenizer name or local path.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained calls.",
    )
    parser.add_argument(
        "--add-latent-tokens",
        action="store_true",
        help="Also add <|latent_pad|>, <|latent_start|>, <|latent_end|> before checking.",
    )
    args = parser.parse_args()

    processor_cls = getattr(transformers, "AutoProcessor")
    processor = processor_cls.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    if args.add_latent_tokens:
        processor.tokenizer.add_tokens(
            ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"],
            special_tokens=True,
        )

    tokenizer_len = len(processor.tokenizer)
    print(f"tokenizer_len={tokenizer_len}")
    print("model_not_loaded=True")


if __name__ == "__main__":
    main()
