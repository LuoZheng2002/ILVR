from trl import SFTTrainer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import contextlib
import logging
import os
import json
import random
from typing import Any, Dict, List, cast

import torch.distributed as dist
import numpy as np
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
except Exception:
    FSDP = None

try:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
except Exception:
    apply_multimodal_rotary_pos_emb = None

class _ShardedEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        self._backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name] = p.detach().clone().float()

    @torch.no_grad()
    def update(self, student: nn.Module):
        d = self.decay
        for name, p_s in student.named_parameters():
            if not p_s.requires_grad or name not in self.shadow:
                continue
            if p_s.numel() == 0:
                continue
            shadow = self.shadow[name]
            data = p_s.detach()
            if shadow.device != data.device:
                shadow = shadow.to(data.device)
                self.shadow[name] = shadow
            if shadow.shape != data.shape:
                self.shadow[name] = data.clone().float()
                continue
            shadow.mul_(d).add_(data.float(), alpha=(1.0 - d))

    @torch.no_grad()
    def begin_teacher(self, model: nn.Module):
        self._backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad or name not in self.shadow:
                continue
            self._backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].to(device=p.device, dtype=p.dtype))

    @torch.no_grad()
    def end_teacher(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name not in self._backup:
                continue
            p.data.copy_(self._backup[name].to(device=p.device, dtype=p.dtype))
        self._backup = {}

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {k: v.detach().cpu() for k, v in self.shadow.items()},
            "shadow_shapes": {k: tuple(v.shape) for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: Dict[str, object], model: nn.Module):
        decay_value = cast(Any, state_dict.get("decay", self.decay))
        try:
            self.decay = float(decay_value)
        except Exception:
            pass
        raw_shadow_obj = state_dict.get("shadow", {})
        raw_shadow: Dict[str, Any] = raw_shadow_obj if isinstance(raw_shadow_obj, dict) else {}
        self.shadow = {}
        loaded = 0
        skipped_shape = 0
        model_params = {
            name: p
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        for name, target in model_params.items():
            tensor_obj = raw_shadow.get(name)
            if not isinstance(tensor_obj, torch.Tensor):
                continue
            tensor = tensor_obj
            if tuple(tensor.shape) != tuple(target.shape):
                skipped_shape += 1
                continue
            self.shadow[name] = tensor.to(device=target.device, dtype=torch.float32)
            loaded += 1
        return {
            "loaded": loaded,
            "skipped_shape": skipped_shape,
            "target_trainable": len(model_params),
            "source_tensors": len(raw_shadow),
        }


class _EMAOptimizerStepCallback(TrainerCallback):
    def __init__(self, trainer):
        self._trainer = trainer

    def on_optimizer_step(self, args, state, control, **kwargs):
        self._trainer._update_ema_after_optimizer_step()
        return control


class CustomTrainerStage1(SFTTrainer):
    def __init__(
        self,
        *args,
        sim_weight: float = 1.0,
        ema_tau: float = 0.999,
        coverage_p: float = 0.9,
        image_pool_k: int = 8,
        ce_weight: float = 1.0,
        helper_group_L: int = 256,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.sim_weight = float(sim_weight)
        self.coverage_p = float(coverage_p)
        self.helper_group_L = int(helper_group_L)
        self.ce_weight = float(ce_weight)
        self.image_pool_k = int(image_pool_k)
        self._ema = None
        ema_tau = float(ema_tau)
        if ema_tau > 0.0:
            self._ema = _ShardedEMA(self.model, decay=ema_tau)
            logging.info("Initialized sharded EMA (tau=%s) for FSDP training.", ema_tau)
            self.add_callback(_EMAOptimizerStepCallback(self))

    def _update_ema_after_optimizer_step(self):
        if self._ema is None:
            return
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and bool(getattr(accelerator, "optimizer_step_was_skipped", False)):
            return
        with torch.no_grad():
            self._ema.update(self.model)

    def _dist_rank(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
        return 0

    def _dist_world_size(self) -> int:
        if dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
        return 1

    def _is_fsdp_wrapped(self, model: nn.Module) -> bool:
        return FSDP is not None and isinstance(model, FSDP)

    def _maybe_fsdp_summon_full_params(self, model: nn.Module):
        if FSDP is None:
            return contextlib.nullcontext()
        if isinstance(model, FSDP):
            return FSDP.summon_full_params(model, recurse=True, writeback=False)
        return contextlib.nullcontext()

    def _ema_shard_path(self, checkpoint_dir: str) -> str:
        rank = self._dist_rank()
        return os.path.join(checkpoint_dir, f"ema_shard_rank{rank:05d}.pt")

    def _checkpoint_meta_path(self, checkpoint_dir: str) -> str:
        return os.path.join(checkpoint_dir, "fsdp_checkpoint_meta.json")

    def _rng_shard_path(self, checkpoint_dir: str) -> str:
        rank = self._dist_rank()
        return os.path.join(checkpoint_dir, f"rng_state_rank{rank:05d}.pt")

    def _capture_rng_state(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            payload["torch_cuda_device"] = int(device_idx)
            payload["torch_cuda"] = torch.cuda.get_rng_state(device_idx)
        return payload

    def _restore_rng_state(self, payload: Dict[str, Any]):
        if not isinstance(payload, dict):
            return
        py_state = payload.get("python")
        np_state = payload.get("numpy")
        cpu_state = payload.get("torch_cpu")
        cuda_state = payload.get("torch_cuda")
        if py_state is not None:
            random.setstate(py_state)
        if np_state is not None:
            np.random.set_state(np_state)
        if cpu_state is not None:
            torch.random.set_rng_state(cpu_state)
        if cuda_state is not None and torch.cuda.is_available():
            device_idx = int(payload.get("torch_cuda_device", torch.cuda.current_device()))
            torch.cuda.set_rng_state(cuda_state, device_idx)

    def _save_rng_shard(self, checkpoint_dir: str):
        payload = self._capture_rng_state()
        torch.save(payload, self._rng_shard_path(checkpoint_dir))

    def _load_rng_shard(self, checkpoint_dir: str):
        path = self._rng_shard_path(checkpoint_dir)
        if not os.path.isfile(path):
            logging.warning("RNG shard checkpoint not found for this rank: %s", path)
            return False
        payload = torch.load(path, map_location="cpu")
        self._restore_rng_state(payload)
        logging.info("Loaded RNG shard checkpoint: %s", path)
        return True

    def _save_ema_shard(self):
        if self._ema is None:
            return
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        ema_path = self._ema_shard_path(checkpoint_dir)
        payload = self._ema.state_dict()
        payload["_ema_meta"] = {
            "rank": self._dist_rank(),
            "world_size": self._dist_world_size(),
            "global_step": int(self.state.global_step),
        }
        torch.save(payload, ema_path)

    def _save_checkpoint_meta(self, checkpoint_dir: str):
        if not self.is_world_process_zero():
            return
        fsdp_config = getattr(self.args, "fsdp_config", {})
        fsdp_state_dict_type = None
        if isinstance(fsdp_config, dict):
            fsdp_state_dict_type = fsdp_config.get("state_dict_type")
        payload = {
            "global_step": int(self.state.global_step),
            "world_size": self._dist_world_size(),
            "rank": self._dist_rank(),
            "ema_enabled": self._ema is not None,
            "ema_shard_pattern": "ema_shard_rank%05d.pt",
            "rng_shard_pattern": "rng_state_rank%05d.pt",
            "fsdp": getattr(self.args, "fsdp", None),
            "fsdp_state_dict_type": fsdp_state_dict_type,
        }
        with open(self._checkpoint_meta_path(checkpoint_dir), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _load_checkpoint_meta(self, checkpoint_dir: str):
        meta_path = self._checkpoint_meta_path(checkpoint_dir)
        if not os.path.isfile(meta_path):
            logging.warning("Checkpoint metadata file not found: %s", meta_path)
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            logging.info("Loaded checkpoint metadata: %s", payload)
            return payload
        except Exception as e:
            logging.warning("Failed to load checkpoint metadata (%s): %s", meta_path, e)
            return None

    def _select_best_ema_shard(self, checkpoint_dir: str):
        if not os.path.isdir(checkpoint_dir):
            return None, None
        trainable = {
            name: tuple(p.shape)
            for name, p in self.model.named_parameters()
            if p.requires_grad
        }
        best_path = None
        best_payload = None
        best_score = -1
        for fname in os.listdir(checkpoint_dir):
            if not (fname.startswith("ema_shard_rank") and fname.endswith(".pt")):
                continue
            path = os.path.join(checkpoint_dir, fname)
            try:
                payload = torch.load(path, map_location="cpu")
            except Exception:
                continue
            raw_shadow = payload.get("shadow", {})
            score = 0
            for name, shape in trainable.items():
                t = raw_shadow.get(name)
                if t is not None and tuple(t.shape) == shape:
                    score += 1
            if score > best_score:
                best_score = score
                best_path = path
                best_payload = payload
        return best_path, best_payload

    def _load_ema_shard(self, checkpoint_dir: str):
        if self._ema is None:
            return {"enabled": False, "loaded": False}
        ema_path = self._ema_shard_path(checkpoint_dir)
        payload = None
        source_path = ema_path
        if os.path.isfile(ema_path):
            payload = torch.load(ema_path, map_location="cpu")
        else:
            best_path, best_payload = self._select_best_ema_shard(checkpoint_dir)
            if best_payload is None:
                logging.warning("EMA shard checkpoint not found for rank or fallback: %s", checkpoint_dir)
                return {"enabled": True, "loaded": False}
            source_path = best_path
            payload = best_payload
            logging.warning(
                "EMA shard for current rank missing; falling back to best available shard: %s",
                best_path,
            )
        stats = self._ema.load_state_dict(payload, self.model)
        logging.info("Loaded EMA shard checkpoint: %s (stats=%s)", source_path, stats)
        return {"enabled": True, "loaded": True, "source": source_path, "stats": stats}

    def _resume_parity_report(self, checkpoint_dir: str, meta_payload, ema_report, rng_loaded: bool):
        optim_state_size = 0
        if getattr(self, "optimizer", None) is not None:
            try:
                optim_state_size = len(self.optimizer.state_dict().get("state", {}))
            except Exception:
                optim_state_size = -1

        scheduler_last_epoch = None
        if getattr(self, "lr_scheduler", None) is not None:
            scheduler_last_epoch = getattr(self.lr_scheduler, "last_epoch", None)

        report = {
            "checkpoint_dir": checkpoint_dir,
            "global_step": int(self.state.global_step),
            "optimizer_state_entries": optim_state_size,
            "scheduler_last_epoch": scheduler_last_epoch,
            "ema": ema_report,
            "rng_loaded": bool(rng_loaded),
        }

        if isinstance(meta_payload, dict):
            expected_step = meta_payload.get("global_step")
            expected_world_size = meta_payload.get("world_size")
            if expected_step is not None and int(expected_step) != int(self.state.global_step):
                logging.warning(
                    "Resume parity warning: global_step mismatch (meta=%s, loaded=%s)",
                    expected_step,
                    self.state.global_step,
                )
            if expected_world_size is not None and int(expected_world_size) != int(self._dist_world_size()):
                logging.warning(
                    "Resume world-size differs from checkpoint metadata (meta=%s, current=%s)",
                    expected_world_size,
                    self._dist_world_size(),
                )

        logging.info("Resume parity report: %s", report)

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial, metrics)
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        self._save_ema_shard()
        self._save_rng_shard(checkpoint_dir)
        self._save_checkpoint_meta(checkpoint_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        meta_payload = self._load_checkpoint_meta(resume_from_checkpoint)
        ema_report = self._load_ema_shard(resume_from_checkpoint)
        rng_loaded = self._load_rng_shard(resume_from_checkpoint)
        self._resume_parity_report(resume_from_checkpoint, meta_payload, ema_report, rng_loaded)

    def _find_latent_segments(self, input_ids, latent_start_id, latent_end_id, latent_pad_id):
        ids = input_ids[0].tolist()
        segments = []
        t = 0
        T = len(ids)
        while t < T:
            if ids[t] == latent_start_id:
                s = t
                e = s + 1
                while e < T and ids[e] != latent_end_id:
                    e += 1
                pad_pos = [i for i in range(s, e+1) if ids[i] == latent_pad_id]
                segments.append(pad_pos)
                t = e + 1
            else:
                t += 1
        return segments

    def _build_firstK_mask(self, input_ids, segments, K_list):
        B, T = input_ids.shape
        mask = torch.zeros(B, T, dtype=torch.bool, device=input_ids.device)
        assert len(segments) == len(K_list)
        for pads, K in zip(segments, K_list):
            if K > 0 and len(pads) > 0:
                take = min(K, len(pads))
                for i in pads[:take]:
                    mask[0, i] = True
        return mask

    def _top_p_top_k(self, scores: torch.Tensor, p: float, k: int):
        if scores.numel() == 0 or k <= 0:
            return []
        probs = torch.softmax(scores, dim=0)
        sorted_probs, idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=0)
        pos = torch.nonzero(cum >= p, as_tuple=False)
        if pos.numel() > 0:
            Kp = int(pos[0].item()) + 1
        else:
            Kp = int(probs.numel())
        Kstar = min(Kp, int(k))
        return idx[:Kstar].tolist()

    def _grouped_mean(self, ei: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 0:
            return ei.new_zeros((0, ei.shape[-1]))
        P = ei.shape[0]
        if P >= k and (P % k) != 0:
            newP = P - (P % k)
            if newP > 0:
                ei = ei[:newP]
            else:
                return ei.new_zeros((k, ei.shape[-1]))
        if ei.shape[0] >= k and ei.shape[0] % k == 0:
            g = ei.view(k, ei.shape[0]//k, ei.shape[-1]).mean(dim=1)
            return g
        if ei.shape[0] == 0:
            return ei.new_zeros((k, ei.shape[-1]))
        rep = math.ceil(k / ei.shape[0])
        g = ei.repeat(rep, 1)[:k]
        return g

    def _maybe_group_for_helper(self, ei: torch.Tensor, L_group: int) -> torch.Tensor:
        if L_group is None or L_group <= 0:
            return ei
        P = int(ei.shape[0])
        if P < L_group:
            return ei
        return self._grouped_mean(ei, L_group)

    def _prefix_text_mean_from_embeds(self, token_embeds: torch.Tensor, input_ids: torch.Tensor,
                                      upto_idx: int, special_ids: set) -> torch.Tensor:
        H = token_embeds.shape[-1]
        t_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if upto_idx > 0:
            t_mask[:, :upto_idx] = 1
        for sid in special_ids:
            t_mask &= (input_ids != sid)
        feats = token_embeds[0, t_mask[0]]
        if feats.numel() == 0:
            feats = token_embeds[0, :max(1, upto_idx)]
        return feats.mean(dim=0, keepdim=True)

    @torch.no_grad()
    def _image_input_global_mean(self, tea, inputs) -> torch.Tensor:
        pv  = inputs.get("pixel_values", None)
        thw = inputs.get("image_grid_thw", None)
        if pv is None or thw is None:
            return None
        pv  = pv.to(next(tea.parameters()).device).type(tea.visual.dtype)
        thw = thw.to(next(tea.parameters()).device)
        with self._maybe_fsdp_summon_full_params(tea):
            patches = tea.visual(pv, grid_thw=thw)
        num_imgs = thw.shape[0]
        if num_imgs == 0 or patches.numel() == 0:
            return None

        s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
        thw_long = thw.to(dtype=torch.long)
        tokens_per_img = thw_long[:,0] * (thw_long[:,1]//s_merge) * (thw_long[:,2]//s_merge)
        ends = torch.cumsum(tokens_per_img, dim=0).tolist()
        starts = [0] + ends[:-1]
        outs = []
        for st, ed in zip(starts, ends):
            ei = patches[st:ed, :]
            gk = self._grouped_mean(ei, self.image_pool_k)
            outs.append(gk)
        all_groups = torch.cat(outs, dim=0)
        return all_groups.mean(dim=0, keepdim=True)

    def _get_special_ids(self):
        tok = self.processing_class
        get_id = lambda s: tok(s, return_tensors="pt")["input_ids"][0,0].item()
        latent_pad_id   = get_id("<|latent_pad|>")
        latent_start_id = get_id("<|latent_start|>")
        latent_end_id   = get_id("<|latent_end|>")
        special_ids = {latent_pad_id, latent_start_id, latent_end_id}
        try:
            vision_start_id = get_id("<|vision_start|>")
            vision_end_id   = get_id("<|vision_end|>")
            special_ids.update({vision_start_id, vision_end_id})
        except Exception:
            pass
        img_token_id = getattr(self.model.config, "image_token_id", None)
        if img_token_id is None:
            img_token_id = 151655
        return special_ids, int(img_token_id), int(latent_start_id), int(latent_end_id), int(latent_pad_id)

    @torch.no_grad()
    def _user_side_attn_pooling(self, tea, inputs, ids, attn) -> torch.Tensor:
        device = ids.device
        B, T = ids.shape
        cand_texts = ["<|im_start|>assistant", "<|im_start|>assistant\n"]
        cand_patterns = [
            self.processing_class(s, return_tensors="pt")["input_ids"][0].to(device) for s in cand_texts
        ]
        start_assistant = -1
        for pat in cand_patterns:
            for i in range(0, T - pat.size(0) + 1):
                if torch.all(ids[0, i:i+pat.size(0)] == pat):
                    start_assistant = i
                    break
            if start_assistant != -1:
                break
        if start_assistant <= 0:
            return None

        special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()
        id_row = ids[0, :start_assistant]
        prompt_idx = [i for i in range(id_row.size(0))
                    if (int(id_row[i].item()) not in special_ids) and (int(id_row[i].item()) != image_token_id)]
        image_idx  = (id_row == image_token_id).nonzero().view(-1).tolist()
        if len(prompt_idx) == 0 or len(image_idx) == 0:
            return None

        old_flag = bool(getattr(tea.config, "output_hidden_states", False))
        tea.config.output_hidden_states = True
        try:
            out = tea(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                output_hidden_states=True,
                return_dict=True,
            )
            hs = out.hidden_states
        finally:
            tea.config.output_hidden_states = old_flag

        if isinstance(hs, (tuple, list)):
            H_last = hs[-1]
            H_pre  = hs[-2] if len(hs) > 1 else hs[-1]
        else:
            H_last = hs
            H_pre  = hs

        try:
            layer_last = tea.model.layers[-1].self_attn
            q = layer_last.q_proj(H_pre)
            k = layer_last.k_proj(H_pre)
            num_heads = layer_last.num_heads
            dh = q.shape[-1] // num_heads
            q = q.view(1, T, num_heads, dh).transpose(1, 2)
            kv = layer_last.num_key_value_heads
            k = k.view(1, T, kv, dh).transpose(1, 2)
            if kv != num_heads:
                rep = num_heads // kv
                k = k[:, :, None, :, :].expand(1, kv, rep, T, dh).reshape(1, num_heads, T, dh)

            q_sel = q[:, :, prompt_idx, :]
            k_sel = k[:, :, image_idx, :]
            logits = torch.einsum("bhpd,bhqd->bhpq", q_sel, k_sel) / (dh ** 0.5)
            probs_per_prompt = torch.softmax(logits, dim=-1)
            w = probs_per_prompt.mean(dim=2).mean(dim=1).squeeze(0)
            w = w / (w.sum() + 1e-9)

            V_img = H_last[0, :start_assistant, :][image_idx, :]
            V_prm = H_last[0, :start_assistant, :][prompt_idx, :]
            r_img = (w.unsqueeze(0) @ V_img).squeeze(0)
            r_txt = V_prm.mean(dim=0)
            u = torch.stack([r_img, r_txt], dim=0).mean(dim=0, keepdim=True)
            return u
        except Exception:
            H_sub = H_last[0, :start_assistant, :]
            V_img = H_sub[image_idx, :]
            V_prm = H_sub[prompt_idx, :]
            sim = (V_prm @ V_img.t()) / (V_img.shape[-1] ** 0.5)
            w = torch.softmax(sim, dim=-1).mean(dim=0)
            w = w / (w.sum() + 1e-9)
            r_img = (w.unsqueeze(0) @ V_img).squeeze(0)
            r_txt = V_prm.mean(dim=0)
            u = torch.stack([r_img, r_txt], dim=0).mean(dim=0, keepdim=True)
            return u

    def _extract_assistant_text_spans(self, ids: torch.Tensor,
                                      latent_starts: List[int], latent_ends: List[int],
                                      assistant_start: int, special_ids: set) -> List[List[int]]:
        spans = []
        prev = assistant_start
        for s, e in zip(latent_starts, latent_ends):
            span_idx = []
            for t in range(prev, s):
                tid = int(ids[t].item())
                if (tid not in special_ids):
                    span_idx.append(t)
            spans.append(span_idx)
            prev = e + 1
        return spans

    @torch.no_grad()
    def _teacher_build_latents_fsdp_safe(self, inputs):
        tea = self.model
        device = self.model.device
        ids = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)

        old_flag = bool(getattr(tea.config, "output_hidden_states", False))
        tea.config.output_hidden_states = True
        try:
            out = tea(
                input_ids=ids,
                attention_mask=attn,
                pixel_values=inputs.get("pixel_values", None),
                image_grid_thw=inputs.get("image_grid_thw", None),
                output_hidden_states=True,
                return_dict=True,
            )
        finally:
            tea.config.output_hidden_states = old_flag

        hs = out.hidden_states
        h_last = hs[-1] if isinstance(hs, (tuple, list)) else hs
        image_out_mask = inputs.get("image_out_mask", None)
        if image_out_mask is None:
            return None, torch.zeros_like(ids, dtype=torch.bool)
        image_out_mask = image_out_mask.to(h_last.device).bool()
        if not image_out_mask.any():
            return None, image_out_mask

        selected = h_last[image_out_mask]
        if selected.numel() == 0:
            return None, image_out_mask
        teacher_latents = selected.unsqueeze(0)
        return teacher_latents, image_out_mask

    @torch.no_grad()
    def _teacher_build_latents(self, inputs, k, p):
        # Inputs: `inputs` contains token tensors and (optionally) two image streams:
        #  - `pixel_values` / `image_grid_thw`: visual inputs corresponding to user images
        #  - `pixel_values_latent` / `image_grid_thw_latent`: helper images used to construct latents
        # `k` is the requested latent size, `p` is coverage probability for top-p selection.
        device = self.model.device
        ids  = inputs["input_ids"].to(device)
        attn = inputs["attention_mask"].to(device)
        tea = self.model

        if self._is_fsdp_wrapped(tea):
            return self._teacher_build_latents_fsdp_safe(inputs)

        restore_training = bool(tea.training)
        ema_ctx = contextlib.nullcontext()
        if self._ema is not None:
            ema_ctx = self._ema_teacher_context()
        tea.eval()

        with ema_ctx:
            try:
                special_ids, image_token_id, latent_start_id, latent_end_id, latent_pad_id = self._get_special_ids()

                # Find positions (indices) of latent-pad tokens within each latent segment.
                seg_pad_indices = self._find_latent_segments(ids, latent_start_id, latent_end_id, latent_pad_id)
                if len(seg_pad_indices) == 0:
                    return None, torch.zeros_like(ids, dtype=torch.bool)

                idlist = ids[0].tolist()
                starts = []
                ends = []
                t = 0
                T = len(idlist)
                while t < T:
                    if idlist[t] == latent_start_id:
                        s = t
                        e = s + 1
                        while e < T and idlist[e] != latent_end_id:
                            e += 1
                        starts.append(s)
                        ends.append(e)
                        t = e + 1
                    else:
                        t += 1

                pat = self.processing_class("<|im_start|>assistant", return_tensors="pt")["input_ids"][0].to(device)
                start_assistant = -1
                for i in range(0, ids.size(1) - pat.size(0) + 1):
                    if torch.all(ids[0, i:i+pat.size(0)] == pat):
                        start_assistant = i
                        break
                if start_assistant <= 0:
                    return None, torch.zeros_like(ids, dtype=torch.bool)

                # Build the contextual query `u` used to score visual patches.
                # Preferred strategy: use `_user_side_attn_pooling` which leverages
                # cross-attention patterns between prompt tokens and image tokens in the teacher's last layer.
                # Fallback: mean of prefix text embeddings and global image embedding.
                u = self._user_side_attn_pooling(tea, inputs, ids, attn)
                if u is None:
                    token_embeds = tea.get_input_embeddings()(ids)
                    u_text = self._prefix_text_mean_from_embeds(token_embeds, ids, start_assistant, special_ids)
                    u_img = self._image_input_global_mean(tea, inputs)
                    parts = [u_text] + ([u_img] if u_img is not None else [])
                    # `u` shape: (1, H) where H is hidden dim of the model's text/vision representation
                    u = torch.stack([x.squeeze(0) for x in parts]).mean(dim=0, keepdim=True)

                pv = inputs.get("pixel_values_latent", None)
                thw = inputs.get("image_grid_thw_latent", None)
                if pv is None or thw is None:
                    return None, torch.zeros_like(ids, dtype=torch.bool)
                pv = pv.to(device).to(tea.visual.dtype)
                thw = thw.to(device)
                # Extract dense visual patch embeddings from the teacher visual encoder.
                # `patch_all` shape: (num_patches_total, D), where patches are concatenated across helper images.
                with self._maybe_fsdp_summon_full_params(tea):
                    patch_all = tea.visual(pv, grid_thw=thw)
                num_imgs = int(thw.shape[0])

                s_merge = int(getattr(tea.visual, "spatial_merge_size", 2))
                thw_long = thw.to(dtype=torch.long)
                tokens_per_img = (thw_long[:, 0] * (thw_long[:, 1] // s_merge) * (thw_long[:, 2] // s_merge))
                ends_img = torch.cumsum(tokens_per_img, dim=0).tolist()
                starts_img = [0] + ends_img[:-1]
                slices_per_img = [(int(st), int(ed)) for st, ed in zip(starts_img, ends_img)]
                assert len(slices_per_img) == num_imgs

                text_spans = self._extract_assistant_text_spans(ids[0], starts, ends, start_assistant, special_ids)
                L_group = getattr(self, "helper_group_L", None)
                if L_group is None:
                    L_group = int(self.image_pool_k)

                latents_list = []
                Kstars = []
                prev_sel_mean = None

                # Iterate latent segments (one per helper image/segment). For each
                # segment we compute similarities between candidate patch embeddings
                # and the contextual query `q_t`, then select top patches by top-p/k.
                for seg_idx, pad_pos in enumerate(seg_pad_indices):
                    if len(pad_pos) == 0:
                        Kstars.append(0)
                        continue

                    text_idx = text_spans[seg_idx] if seg_idx < len(text_spans) else []
                    old_flag = bool(getattr(tea.config, "output_hidden_states", False))
                    tea.config.output_hidden_states = True
                    try:
                        out_assist = tea(
                            input_ids=ids,
                            attention_mask=attn,
                            pixel_values=inputs.get("pixel_values", None),
                            image_grid_thw=inputs.get("image_grid_thw", None),
                            output_hidden_states=True,
                            return_dict=True,
                        )
                    finally:
                        tea.config.output_hidden_states = old_flag

                    hs = out_assist.hidden_states
                    x = hs[-1] if isinstance(hs, (tuple, list)) else hs
                    H_last_2d = x[0] if x.dim() == 3 else (x if x.dim() == 2 else tea.get_input_embeddings()(ids)[0])

                    q_parts = [u]

                    if len(text_idx) > 0:
                        idx_tensor = torch.tensor(text_idx, device=H_last_2d.device, dtype=torch.long)
                        q_parts.append(H_last_2d.index_select(0, idx_tensor).mean(dim=0, keepdim=True))

                    if prev_sel_mean is not None:
                        q_parts.append(prev_sel_mean)

                    q_t = torch.stack([x.squeeze(0) for x in q_parts], dim=0).mean(dim=0, keepdim=True)

                    assert seg_idx < num_imgs, "latent 段数量与 helper images 数量不一致"
                    st_img, ed_img = slices_per_img[seg_idx]
                    # `ei` are patch vectors for this helper image: shape (P, D)
                    ei = patch_all[st_img:ed_img, :]
                    # Optionally group/aggregate patch vectors to reduce P to a target L_group
                    cand = self._maybe_group_for_helper(ei, L_group)

                    # Compute cosine similarity between each candidate patch (P') and query (1, D)
                    # `sim` shape: (P',)
                    sim = F.cosine_similarity(cand, q_t.expand_as(cand), dim=-1)
                    # Select indices via top-p/top-k strategy
                    top_idx = self._top_p_top_k(sim, p=self.coverage_p, k=k)
                    Kstar = len(top_idx)
                    Kstars.append(Kstar)

                    if Kstar > 0:
                        chosen = cand[top_idx[:Kstar], :]
                        latents_list.append(chosen)
                        prev_sel_mean = chosen.mean(dim=0, keepdim=True)

                if len(latents_list) == 0:
                    return None, torch.zeros_like(ids, dtype=torch.bool)
                latents = torch.cat(latents_list, dim=0).unsqueeze(0)

                firstK_mask = self._build_firstK_mask(ids, seg_pad_indices, Kstars)
                return latents, firstK_mask
            finally:
                if restore_training:
                    tea.train()

    @contextlib.contextmanager
    def _ema_teacher_context(self):
        if self._ema is None:
            yield
            return
        self._ema.begin_teacher(self.model)
        try:
            yield
        finally:
            self._ema.end_teacher(self.model)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # High-level compute_loss flow:
        # 1) Build teacher latents via `_teacher_build_latents(inputs)` (returns latents, firstK_mask)
        # 2) If teacher latents are available, create `mod_inputs` containing `latent_hidden_states`
        #    and `image_out_mask` and call the parent's `compute_loss` to obtain CE loss and outputs.
        # 3) If `sim_weight` > 0, compute similarity loss between predicted hidden states and
        #    the inputs_embeds at latent positions using `firstK_mask`, and combine with CE.
        k = getattr(self.model.config, "latent_size", 8)
        teacher_latents, firstK_mask = self._teacher_build_latents(inputs, k=k, p=self.coverage_p)
        
        with torch.no_grad():
            pv_lat = inputs.get("pixel_values_latent", None)
            thw_lat = inputs.get("image_grid_thw_latent", None)
            if (not self._is_fsdp_wrapped(self.model)) and pv_lat is not None and thw_lat is not None:
                with self._maybe_fsdp_summon_full_params(self.model):
                    _ = self.model.visual(
                        pv_lat.to(self.model.device).to(self.model.visual.dtype),
                        grid_thw=thw_lat.to(self.model.device)
                    )
        
        if teacher_latents is None:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return (ce_loss, outputs) if return_outputs else ce_loss

        if teacher_latents.dim() == 2:
            teacher_latents = teacher_latents.unsqueeze(0)
        S = int(firstK_mask.sum().item())
        if teacher_latents.shape[1] != S:
            ce_loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
            return (ce_loss, outputs) if return_outputs else ce_loss

        mod_inputs = dict(inputs)
        mod_inputs.pop("pixel_values_latent", None)
        mod_inputs["latent_hidden_states"] = teacher_latents.to(self.model.device).to(self.model.dtype)
        mod_inputs["image_out_mask"] = firstK_mask

        # Request hidden states explicitly because newer HF output objects don't
        # expose `inputs_embeds` and return hidden states as a tuple by layer.
        mod_inputs["output_hidden_states"] = True
        ce_loss, outputs = super().compute_loss(
            model, mod_inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )
        if self.sim_weight == 0.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred_h = outputs.hidden_states
        if isinstance(pred_h, (tuple, list)):
            pred_h = pred_h[-1] if len(pred_h) > 0 else None
        if pred_h is None:
            return (ce_loss, outputs) if return_outputs else ce_loss

        input_ids = mod_inputs.get("input_ids", None)
        if input_ids is None:
            return (ce_loss, outputs) if return_outputs else ce_loss
        inp_h = self.model.get_input_embeddings()(input_ids.to(pred_h.device))
        B, T, H = pred_h.shape
        if T <= 1:
            return (ce_loss, outputs) if return_outputs else ce_loss

        mask = mod_inputs["image_out_mask"][:, -(T - 1):].to(pred_h.device).bool()
        if not mask.any():
            return (ce_loss, outputs) if return_outputs else ce_loss

        pred = pred_h[..., :-1, :][mask].contiguous().float()
        gt   = inp_h[...,  1:, :][mask].contiguous().detach().float()
        gt = gt + 0.01 * torch.randn_like(gt)

        cos = F.cosine_similarity(gt, pred, dim=-1).mean()
        sim_loss = 1.0 - cos
        loss = self.ce_weight * ce_loss + self.sim_weight * sim_loss
        return (loss, outputs) if return_outputs else loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self._update_ema_after_optimizer_step()
