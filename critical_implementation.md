Critical algorithm → implementation mapping (ILVR)

This file maps the key algorithmic components shown in the paper (ILVR) to where they are implemented in the repository. For each component I list the most relevant files / functions and a short note about how the code implements it.

- Interleaved Latent Visual Reasoning (overall loop)
  - `src/main.py` — `main_train()` and `collate_fn_stage1`
    - `collate_fn_stage1` constructs the interleaved textual template with latent placeholders (`<|latent_start|>`, `<|latent_pad|>`, `<|latent_end|>`), inserts the latent tokens into the tokenizer and prepares `batch['labels']` using `generate_labels_with_latent_template`.
    - The training entry sets `config.latent_size` and model stage flags, loads processor and model and starts `CustomTrainerStage1`.

- Latent token insertion & sequence processing
  - `src/main.py` — adds special tokens: `processor.tokenizer.add_tokens(new_tokens, special_tokens=True)` (new_tokens = `['<|latent_pad|>','<|latent_start|>','<|latent_end|>']`).
  - `src/utils_deepseed.py` — `process_batch`, `replace_subsequent_image_parts_1d/2d`, `place_input_image`, `place_output_image`, `replace_visual_spectial_tokens` and `generate_labels_with_latent_template` implement the token-level rewrites and label masking required by the latent template.

- Momentum teacher (EMA) and teacher latents construction
  - `src/trainer.py` — `_EMATeacher` class (wraps a deep-copied, frozen teacher) and `CustomTrainerStage1._teacher_build_latents`.
    - `_EMATeacher.update` maintains the momentum teacher weights.
    - `_teacher_build_latents` runs the EMA teacher to obtain dense visual features for helper images, computes contextual queries, measures cosine similarity between query and candidate patch embeddings, applies top-p/top-k selection, and returns selected latent vectors plus a mask describing which latent-pad tokens are first-K selected. This corresponds to the paper's Momentum MLLM + Adaptive Selection block.
    - Related helpers used here: `_user_side_attn_pooling`, `_image_input_global_mean`, `_maybe_group_for_helper`, `_grouped_mean`, `_top_p_top_k`, `_extract_assistant_text_spans`.

- Adaptive selection (Top-k via cosine similarity)
  - `src/trainer.py` — inside `_teacher_build_latents`: compute `sim = F.cosine_similarity(cand, q_t.expand_as(cand), dim=-1)` and then `top_idx = self._top_p_top_k(sim, p=self.coverage_p, k=k)` to select features. The selection logic and top-p/k behavior is implemented in `_top_p_top_k` and the grouping logic in `_maybe_group_for_helper` / `_grouped_mean`.

- Image feature extraction / vision encoder
  - `CustomTrainerStage1._teacher_build_latents` calls `tea.visual(pv, grid_thw=thw)` to get patch embeddings from the teacher model's visual encoder. The same call is present in `src/trainer.py::_image_input_global_mean` and in `CustomTrainerStage1.compute_loss` where `self.model.visual(...)` is used to ensure visual layers are warmed/cached.

- Next-step latent alignment loss (similarity loss)
  - `src/trainer.py` — `CustomTrainerStage1.compute_loss`:
    - After building `teacher_latents` via `_teacher_build_latents`, the code creates `mod_inputs` where `latent_hidden_states` is set to the teacher latents and `image_out_mask` is set to the returned mask.
    - It calls `super().compute_loss` (CE) with `mod_inputs` to get `outputs` (which contains predicted hidden states), then computes a cosine-similarity loss between predicted hidden states at latent token positions and the `inputs_embeds` (ground-truth latent embeddings) and combines it with CE: `loss = self.ce_weight * ce_loss + self.sim_weight * sim_loss`.
    - This implements the paper's cross-entropy + latent alignment objective.

- Masking and label generation for latent template
  - `src/utils_deepseed.py` — `generate_labels_with_latent_template`, `mask_latent_output_tokens_all_segments`, `generate_labels_after_latent_tokens`, and `LatentTemplateLogitsProcessor`:
    - `generate_labels_with_latent_template` applies label masking so that only tokens after assistant start and selected latent pads contribute to CE (matches the template logic in figure).
    - `mask_latent_output_tokens_all_segments` returns boolean masks used by the trainer to mark which positions are latent output tokens.
    - `LatentTemplateLogitsProcessor` enforces generation constraints at decode time so model emits pad tokens for latent spans until K are filled then ends the latent span — used in `src/evaluate_deepseed.py`/`eval.py` when creating `LogitsProcessorList`.

- Generation-time constraints for latent tokens
  - `src/utils_deepseed.py` — class `LatentTemplateLogitsProcessor` is applied during generation (see `src/evaluate_deepseed.py` and `src/eval.py`) to restrict logits so latent token generation follows the template (force pad tokens / force end token once enough pads are generated).

- Collation and splitting user/assistant images
  - `src/main.py` — `collate_fn_stage1` relies on helpers `remove_assistant_images`, `remove_user_images` in `src/utils_deepseed.py` to separate image streams for (i) the user input images (full-resolution visual features) and (ii) assistant/helper images (candidate latent images). `process_vision_info` (imported from `qwen_vl_utils`) extracts pixel tensors and grid metadata expected by the model's visual encoder.

- Inference & evaluation utilities
  - `src/evaluate_deepseed.py` and `eval.py` — load `AutoProcessor` and `Qwen2_5_VLForConditionalGeneration`, build prompts via `processor.apply_chat_template`, run `model.generate(...)` and post-process outputs (`extract_final_answer`, `extract_assistant_content`, `extract_path_from_text`). They also use `LatentTemplateLogitsProcessor` to constrain latent generation where applicable.

- Practical anchors to inspect code quickly
  - Teacher + selection: `src/trainer.py::_teacher_build_latents`
  - Loss composition: `src/trainer.py::compute_loss`
  - Token/template preprocessing: `src/utils_deepseed.py::generate_labels_with_latent_template` and `src/utils_deepseed.py::process_batch`
  - Collation / training input building: `src/main.py::collate_fn_stage1`
  - Logits-time template constraints: `src/utils_deepseed.py::LatentTemplateLogitsProcessor`
  - EMA teacher wrapper: `src/trainer.py::_EMATeacher`

If you want, I can open and annotate the specific functions above with inline comments explaining control flow and tensor shapes, or produce a small diagram that points to these functions. Which would you prefer? Reply with "annotate" or "diagram" (or both).
