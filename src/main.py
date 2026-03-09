import os
import re
import json
import copy
import logging
from tqdm import tqdm
from functools import partial

import torch
from torch.nn import functional as F
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLConfig,
    AutoProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import HfDeepSpeedConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

from qwen_vl_utils import process_vision_info

from utils_deepseed import *
from task_deepseed import *
from trainer import CustomTrainerStage1

# ==============================================================
# Collate Functions
# ==============================================================

def collate_fn_stage1(examples, processor, args):
    # Build chat-style prompts for each example. At this stage `texts` contains
    # full ChatML-style prompts (strings) with placeholders for images.
    # Example prompt (simplified): "<|im_start|>user ... <image> ... <|im_start|>assistant ... <output_image>"
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    texts = [place_input_image(t) for t in texts]
    texts = [place_output_image(t) for t in texts]
    texts = replace_visual_spectial_tokens(texts)

    image_inputs, _ = process_vision_info(examples)

    user_examples = remove_assistant_images(examples)
    user_text = [processor.apply_chat_template(example, tokenize=False) for example in user_examples]
    user_text = replace_visual_spectial_tokens(user_text)
    user_image_inputs, _ = process_vision_info(user_examples)
    user_batch = processor(text=user_text, images=user_image_inputs, return_tensors="pt", padding=True)

    assistant_examples = remove_user_images(examples)
    assistant_text = [processor.apply_chat_template(example, tokenize=False) for example in assistant_examples]
    assistant_text = replace_visual_spectial_tokens(assistant_text)
    assistant_image_inputs, _ = process_vision_info(assistant_examples)
    assistant_batch = processor(text=assistant_text, images=assistant_image_inputs, return_tensors="pt", padding=True)

    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
    
    if 'pixel_values' in user_batch:
        batch['pixel_values'] = user_batch['pixel_values']
        batch['image_grid_thw'] = user_batch['image_grid_thw']
    
    if 'pixel_values' in assistant_batch:
        batch['pixel_values_latent'] = assistant_batch['pixel_values']
        batch['image_grid_thw_latent'] = assistant_batch['image_grid_thw']

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    pad_token_idx    = processor.tokenizer("<|endoftext|>", return_tensors="pt")["input_ids"][0,0].item()

    new_input_ids, new_attention_mask = process_batch(
        batch["input_ids"], batch["attention_mask"],
        start_token=latent_start_idx, end_token=latent_end_idx,
        replacement_token=latent_token_idx, replacement_length=args.latent_size,
        pad_token=pad_token_idx
    )
    batch["input_ids"] = new_input_ids
    batch["attention_mask"] = new_attention_mask
    # At this point:
    # - `batch['input_ids']` is a tensor of shape (B, T') where latent spans between
    #   <|latent_start|> and <|latent_end|> have been replaced by `args.latent_size` copies
    #   of the single `latent_pad` token ID. This compresses variable-length visual
    #   segments into a fixed-length latent region per segment.
    # - `batch['attention_mask']` matches the new sequence length.
    
    answer_start_token_pattern = processor.tokenizer("<|im_start|>assistant", return_tensors="pt")["input_ids"][0]

    labels = generate_labels_with_latent_template(
        batch["input_ids"],
        answer_start_token_pattern,
        pad_token_idx,
        int(latent_start_idx),
        int(latent_end_idx),
        int(latent_token_idx),
        latent_ce_ratio=0,
    )
    batch["labels"] = labels

    image_out_mask = mask_latent_output_tokens_all_segments(
        batch["input_ids"], latent_start_idx, latent_end_idx, latent_token_idx
    )
    batch["image_out_mask"] = image_out_mask
    return batch

    

# ==============================================================
# Main Training Function
# ==============================================================
def main_train():
    seed_everything(seed=42)
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ],
    )
    logging.info('==' * 20)
    logging.info(args)
    logging.info('==' * 20)

    cache_dir = args.cache_dir
    os.environ['HF_HOME'] = cache_dir

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ds_config_path = os.path.join(repo_root, "configs", "config_stage3.json")
    per_device_train_batch_size = 1
    gradient_accumulation_steps = int(args.gradient_accumulation_steps)
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)

    with open(ds_config_path, "r", encoding="utf-8") as f:
        ds_config = json.load(f)
    zero_stage = int(ds_config.get("zero_optimization", {}).get("stage", 0))
    assert zero_stage == 3, (
        f"DeepSpeed ZeRO-3 is required, but config has zero_optimization.stage={zero_stage}"
    )

    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["gradient_accumulation_steps"] = gradient_accumulation_steps
    ds_config["train_batch_size"] = (
        per_device_train_batch_size * gradient_accumulation_steps * world_size
    )
    ds_init_config = copy.deepcopy(ds_config)
    ds_init_config["train_batch_size"] = (
        per_device_train_batch_size * gradient_accumulation_steps
    )

    logging.info(
        "Resolved DeepSpeed batch settings: micro_batch=%s, grad_accum=%s, train_batch=%s, world_size=%s",
        ds_config["train_micro_batch_size_per_gpu"],
        ds_config["gradient_accumulation_steps"],
        ds_config["train_batch_size"],
        world_size,
    )

    ds_init_helper = HfDeepSpeedConfig(ds_init_config)
    logging.info(f"Enabled HfDeepSpeedConfig for ZeRO init: {ds_config_path}")
    print(f"Enabled HfDeepSpeedConfig for ZeRO init: {ds_config_path}")
    
    logging.info(f"Loading processor from: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model, cache_dir=cache_dir, trust_remote_code=True)
    
    new_tokens = ["<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"]
    processor.tokenizer.add_tokens(new_tokens, special_tokens=True)


    logging.info(f"Loading model (Stage 1) from: {args.model}")
    model_path = args.model
    config = Qwen2_5_VLConfig.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)
    grad_checkpointing = True
    
    
    config.compress_strategy = args.compress_strategy
    config.latent_size = args.latent_size
    config.stage = args.stage

    latent_token_idx = processor.tokenizer("<|latent_pad|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_start_idx = processor.tokenizer("<|latent_start|>", return_tensors="pt")["input_ids"][0,0].item()
    latent_end_idx   = processor.tokenizer("<|latent_end|>", return_tensors="pt")["input_ids"][0,0].item()
    config.latent_token_id = int(latent_token_idx)
    config.latent_start_id = int(latent_start_idx)
    config.latent_end_id   = int(latent_end_idx)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # old code:
    # model.resize_token_embeddings(len(processor.tokenizer))
    
    # new code: to ensure vocab sync across tokenizer, model embeddings, and model config after adding latent tokens
    new_vocab = len(processor.tokenizer)
    model.resize_token_embeddings(new_vocab)
    model.config.vocab_size = new_vocab
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.vocab_size = new_vocab

    in_vocab = model.get_input_embeddings().weight.shape[0]
    out_vocab = model.get_output_embeddings().weight.shape[0]
    assert in_vocab == new_vocab, f"input embedding vocab mismatch: {in_vocab} != {new_vocab}"
    assert out_vocab == new_vocab, f"output embedding vocab mismatch: {out_vocab} != {new_vocab}"
    assert model.config.vocab_size == new_vocab, (
        f"config vocab mismatch: {model.config.vocab_size} != {new_vocab}"
    )
    logging.info(
        f"Vocab sync check passed: tokenizer={new_vocab}, input_emb={in_vocab}, "
        f"output_emb={out_vocab}, config_vocab={model.config.vocab_size}"
    )

    for param in model.visual.parameters():
        param.requires_grad = False
    
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        logging.info(f"Using CUDA device: {torch.cuda.current_device()}")
    

    preprocess_function = task_preporcess_config[args.task]
    train_dataset = load_jsonl_dataset(args.data_path)
    train_dataset = [preprocess_function(sample) for sample in train_dataset]

    
    CustomTrainer = CustomTrainerStage1
    collate_fn_raw = collate_fn_stage1
    
        
    collate_fn = partial(collate_fn_raw, processor=processor, args=args)

    peft_config = None
    if getattr(args, "use_lora", False):
        logging.info("Enabling LoRA...")
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )

    training_args = SFTConfig(
        output_dir=args.save_model_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=args.warm_up_steps,
        learning_rate=1e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        optim="adamw_torch_fused" if args.stage == 'stage1' else "adamw_torch",
        bf16=True,
        push_to_hub=False,
        remove_unused_columns=False,
        gradient_checkpointing=grad_checkpointing,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=[],
        logging_dir='./logs/',
        logging_strategy='steps',
        max_seq_length=32768 if args.stage == 'stage1' else args.max_seq_length_train,
        deepspeed=ds_config,
        ddp_find_unused_parameters=False if args.stage == 'stage2' else None,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
        peft_config=peft_config,
        sim_weight=getattr(args, "sim_weight", 1.0),
        ema_tau=getattr(args, "ema_tau", 0.999),
        coverage_p=getattr(args, "coverage_p", 0.9),
        image_pool_k=getattr(args, "image_pool_k", 8),
        helper_group_L=getattr(args, "helper_group_L", 256),
        ce_weight=getattr(args, "ce_weight", 1.0),
    )
    

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None:
            logging.info(f"last checkpoint: {last_checkpoint}，continue。")
        else:
            logging.info(f"output log {training_args.output_dir} exists but no checkpoint， train from start。")
    else:
        logging.info("no checkpoint，train from start。")

    logging.info("start training (DeepSpeed ZeRO-3 Mode)...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    final_model_path = training_args.output_dir

    if trainer.is_world_process_zero():
        logging.info(f"training finish，save model: {final_model_path}")
        processor.save_pretrained(final_model_path)

    trainer.save_model(final_model_path)

    if trainer.is_world_process_zero():
        logging.info("all saved。")
        
    logging.info("finish。")

if __name__ == "__main__":
    main_train()
