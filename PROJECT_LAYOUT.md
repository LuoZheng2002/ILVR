ILVR — Project layout

This document maps the repository structure and explains the purpose of the main files so you can find code and run training/evaluation quickly.

- Repository root: top-level entries you will use most
  - `README.md` — project overview, installation, dataset layout and examples (use this first).
  - `run_training.sh` — launcher script that calls `accelerate launch src/main.py` (edit its variables to point to your data, model and output paths).
  - `eval.sh` / `eval.py` / `evaluate_deepseed.py` — evaluation drivers (two variants provided). `eval.sh` wraps `eval.py`/`evaluate_deepseed.py` for batch evaluation.
  - `requirements.txt` — python dependencies used by the code.
  - `configs/` — DeepSpeed / accelerate config JSONs used by training (`config_stage1.json`, `config_stage2.json`, `config_stage3.json`).
  - `assets/` — images and PDFs for the README (e.g. `framework.png`, `framework.pdf`).

- `src/` — main implementation
  - `src/main.py` — training entry point. Key responsibilities:
    - builds processor and model (`AutoProcessor`, `Qwen2_5_VLForConditionalGeneration`), injects special latent tokens and sets model config flags for ILVR.
    - defines `collate_fn_stage1` (input formatting for interleaved latent templates) and calls `CustomTrainerStage1`.
    - configures training hyperparameters via `trl.SFTConfig` and optionally LoRA (`peft.LoraConfig`).
  - `src/trainer.py` — custom trainer logic (`CustomTrainerStage1`) built on `trl.SFTTrainer`.
    - Implements an EMA teacher (`_EMATeacher`) and methods to build teacher latents, compute the combined CE + similarity loss, and optimizer/EMA updates.
    - Several helper utilities for selecting latent patches and pooling visual embeddings.
  - `src/task_deepseed.py` — dataset preprocessing / task-specific helpers.
    - Several preprocess functions for tasks (e.g. `interleaved_latent_cot_preprocess_function`, `single_input_image_preprocess_function`) and mapping dictionaries: `task_preporcess_config` and `task_test_preporcess_config`.
    - Small environment defaults for dataset roots; also includes utility helpers used by VSP tasks (e.g. `simulate_vsp`).
  - `src/utils_deepseed.py` — collection of token / sequence utilities and dataset helpers used across training and eval.
    - Argument parsing (`get_args`, training defaults).
    - Dataset loader `load_jsonl_dataset`.
    - Token / sequence manipulation helpers: `place_input_image`, `place_output_image`, `process_batch`, functions to generate labels for the latent template (`generate_labels_with_latent_template`), masking helpers and `LatentTemplateLogitsProcessor` (used at generation time to constrain latent tokens).
  - `src/evaluate_deepseed.py` — evaluation script referencing the same preprocessing & decoding conventions as training.
    - Handles inference, answer extraction (`extract_final_answer`, `extract_assistant_content`) and task-specific scoring (e.g. VSP path simulation).

- Data expectations
  - The README specifies a `data/` layout; training/eval expects JSONL entries with keys like `text_input`, `image_input`, `sequence_plan`, `image_output`, `original_final_answer` depending on the task.
  - Example dataset root constants exist in code (`TRAIN_IMAGE_ROOT` in `src/task_deepseed.py` and `BASE_DATASET_DIR` in `src/evaluate_deepseed.py`) — update these or pass correct paths via the wrapper scripts.

- How to run
  1) Prepare environment (Python 3.11) and install deps: `pip install -r requirements.txt` and install the provided `transformers` local package as described in `README.md`.
  2) Edit `run_training.sh` and set `DATA_PATH`, `SAVE_MODEL_PATH`, `LOG_FILE` and `HF_HOME` (cache).
     - Then run: `bash run_training.sh` (the script uses `accelerate launch` internally).
  3) For evaluation, edit `eval.sh` variables (`MODEL_DIR`, `INPUT_DIR`, `OUTPUT_DIR`, `EVAL_SCRIPT_PATH`, `CACHE_DIR`, `GPU_ID`) and run `bash eval.sh`.
  4) Alternatively call `python src/evaluate_deepseed.py --model_dir <dir> --test_data_path <file> --task_name <task>` directly (see script for arg details).

- Key implementation notes to be aware of
  - Special tokens: the code inserts three special latent tokens `"<|latent_pad|>", "<|latent_start|>", "<|latent_end|>"` into the tokenizer (see `src/main.py`). Keep the processor saved after training so eval uses the same tokenizer.
  - The training loop expects that user vs assistant images are separated (see `collate_fn_stage1` and `remove_user_images` / `remove_assistant_images` helpers in `src/utils_deepseed.py`).
  - `CustomTrainerStage1.compute_loss` builds a teacher latent representation (EMA teacher) and, when available, combines a CE loss with a similarity loss between predicted latent hidden states and the teacher latents.
  - The repo includes two evaluation variants: `eval.py` (more recent, includes timing and summary) and `src/evaluate_deepseed.py` (older but full-featured). Either can be used; `eval.sh` expects you to choose which script to call via `EVAL_SCRIPT_PATH`.

- Useful file references
  - Training entry: `src/main.py`
  - Trainer implementation: `src/trainer.py` (class `CustomTrainerStage1`)
  - Preprocessing/task config: `src/task_deepseed.py`
  - Utilities and label/mask logic: `src/utils_deepseed.py`
  - Evaluation: `src/evaluate_deepseed.py`, `eval.py`
  - Launcher scripts: `run_training.sh`, `eval.sh`
  - DeepSpeed configs: `configs/config_stage1.json`, `configs/config_stage2.json`, `configs/config_stage3.json`

Next steps / recommendations
1. Verify dataset paths: update `TRAIN_IMAGE_ROOT` and `BASE_DATASET_DIR` or pass correct `--data_path`/script env vars before running.
2. Save the `processor` (`AutoProcessor`) after training (the training code already saves it when `trainer.is_world_process_zero()`), so evaluation loads the same tokenizer/special tokens.
3. If you plan to change tokenization or add new special tokens, ensure both training and evaluation use the same `processor` directory.

If you want, I can: (1) generate a shorter quickstart README snippet that shows the exact commands to run for your environment, or (2) open specific files and annotate important functions inline. Tell me which one you prefer.
