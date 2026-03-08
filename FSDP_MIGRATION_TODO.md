# FSDP + bf16 + Sharded EMA Migration TODO

## Scope
- [ ] Replace current DeepSpeed ZeRO-3 training path with PyTorch FSDP.
- [ ] Use `bf16` mixed precision for model forward/backward and parameter handling where supported.
- [ ] Implement Sharded EMA compatible with FSDP sharding semantics.
- [ ] Keep model behavior and training hyperparameters unchanged unless explicitly confirmed.

## 1) Audit Current Training Stack
- [ ] Identify all ZeRO-3/DeepSpeed entry points (`src/`, launch scripts, config files, SLURM scripts).
- [ ] Document current precision behavior (`fp16`/`bf16`, grad scaler, autocast usage).
- [ ] Document current EMA behavior and checkpoint format.
- [ ] Record distributed initialization path and launch assumptions (`torchrun`, env vars, world size/rank handling).

## 2) FSDP Integration
- [x] Add FSDP config surface (CLI/config file): sharding strategy, auto-wrap policy, mixed precision, state dict type.
- [ ] Replace DeepSpeed engine wrapping with FSDP module wrapping.
- [ ] Ensure gradient accumulation and `no_sync` semantics remain correct.
- [ ] Preserve optimizer step ordering and scheduler stepping behavior.
- [ ] Validate activation checkpointing compatibility under FSDP.

## 3) bf16 Enablement
- [x] Set FSDP mixed precision to `bf16` (params/reduce/buffer dtypes as appropriate).
- [ ] Remove/disable fp16-only codepaths (e.g., grad scaler if not needed for bf16).
- [ ] Ensure loss computation and numerically sensitive ops remain in stable dtype where required.
- [x] Add capability checks/fallback messaging for hardware that lacks bf16 support.

## 4) Sharded EMA
- [ ] Replace full-parameter EMA with FSDP-aware sharded EMA updates.
- [ ] Ensure EMA update happens with correct parameter view (local shard vs gathered full params).
- [ ] Add save/load logic for sharded EMA state alongside model/optimizer states.
- [ ] Verify EMA resume correctness across world sizes when supported.

## 5) Checkpointing & Resume
- [ ] Migrate checkpoint save/load from DeepSpeed format to FSDP-compatible format.
- [ ] Decide and implement state dict mode (`full`, `local`, or `sharded`) for train/eval/export.
- [ ] Provide conversion or compatibility path for existing checkpoints if needed.
- [ ] Verify strict resume parity: global step, optimizer, scheduler, EMA, RNG states.

## 6) Launch & Infrastructure Updates
- [x] Update training launch scripts (`run_training.sh`, SLURM files) to remove DeepSpeed launcher assumptions.
- [x] Add/adjust `torchrun` arguments and distributed environment setup.
- [x] Remove DeepSpeed config dependencies from runtime path.
- [ ] Update dependency list (`requirements.txt`) if DeepSpeed becomes optional or removed.

## 7) Validation
- [ ] Smoke test 1-GPU / small batch run.
- [ ] Smoke test multi-GPU FSDP run.
- [ ] Validate loss curve continuity versus pre-migration baseline (short run).
- [ ] Confirm checkpoint save/load and resumed training equivalence.
- [ ] Validate eval/inference path with and without EMA weights.
- [ ] Capture peak memory and throughput before/after migration.

## 8) Documentation
- [ ] Update README and training docs with FSDP usage and required PyTorch/CUDA versions.
- [ ] Document known caveats (bf16 hardware, checkpoint format, EMA behavior).
- [ ] Add troubleshooting section for common distributed/FSDP issues.

## Hyperparameter Change Policy
- [ ] Do **not** change any hyperparameter defaults (LR, batch size, grad accumulation, warmup, weight decay, EMA decay, clipping, etc.) without explicit user confirmation.
- [ ] If a hyperparameter adjustment is required for stability/performance after migration, propose a minimal diff and request approval first.
